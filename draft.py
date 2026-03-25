import os
import re
import csv
import json
import time
import math
import argparse
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from monai.config import print_config
from monai.utils import set_determinism
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, MeanIoU
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandAffined,
    RandShiftIntensityd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
)

from networks.model import VNet
from networks.UNet3D import UNet3D
from networks.basic import VNet


CLASS_LABELS = {
    "0": "background",
    "1": "spleen",
    "2": "rkid",
    "3": "lkid",
    "4": "gall",
    "5": "eso",
    "6": "liver",
    "7": "sto",
    "8": "aorta",
    "9": "IVC",
    "10": "veins",
    "11": "pancreas",
    "12": "rad",
    "13": "lad",
}

RUN_SEED = 123
SPLIT_SEED = 123
rot = np.deg2rad(30.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/btcv")
    parser.add_argument("--split_json", type=str, default="dataset_0.json")
    parser.add_argument("--output_root", type=str, default="./outputs_basic")
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--pool_keys", type=str, nargs="+", default=["training", "validation"])
    parser.add_argument("--train_num", type=int, default=18)
    parser.add_argument("--val_num", type=int, default=6)
    parser.add_argument("--test_num", type=int, default=6)
    parser.add_argument("--split_seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--exclude_cases", type=str, nargs="*", default=[])

    parser.add_argument("--pixdim", type=float, nargs=3, default=[1.5, 1.5, 2.0])
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=2)

    parser.add_argument("--cache_num_train", type=int, default=18)
    parser.add_argument("--cache_num_val", type=int, default=6)
    parser.add_argument("--cache_num_test", type=int, default=6)
    parser.add_argument("--cache_rate", type=float, default=1.0)

    parser.add_argument("--num_workers_train", type=int, default=8)
    parser.add_argument("--num_workers_val", type=int, default=4)
    parser.add_argument("--num_workers_test", type=int, default=4)

    parser.add_argument("--max_iterations", type=int, default=38000)
    parser.add_argument("--eval_num", type=int, default=400)
    parser.add_argument("--val_start_iter", type=int, default=10000)
    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--sw_overlap", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--early_stop_patience", type=int, default=11)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    parser.add_argument("--early_stop_warmup", type=int, default=0)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def csv_safe(x: Any) -> Any:
    v = safe_float(x)
    return "" if v is None else v


def finite_values(values: List[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        sv = safe_float(v)
        if sv is not None:
            out.append(float(sv))
    return out


def mean_or_nan(values: List[Any]) -> float:
    vals = finite_values(values)
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


def std_or_nan(values: List[Any], ddof: int = 1) -> float:
    vals = finite_values(values)
    if len(vals) <= ddof:
        return float("nan")
    return float(np.std(vals, ddof=ddof))


def tensor_mean_of_finite(t: torch.Tensor) -> float:
    vals = [safe_float(x) for x in t.detach().float().cpu().tolist()]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


def get_class_names(num_classes: int) -> List[str]:
    return [CLASS_LABELS.get(str(i), f"class_{i}") for i in range(num_classes)]


def path_to_case_id(path: str) -> str:
    base = os.path.basename(str(path))
    if base.endswith(".nii.gz"):
        base = base[:-7]
    else:
        base = os.path.splitext(base)[0]
    return base if base else "unknown_case"


def extract_case_id_from_record(record: Dict[str, Any]) -> str:
    for key in ["image", "label"]:
        value = record.get(key, "")
        if value:
            return path_to_case_id(value)
    return "unknown_case"


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def deduplicate_records(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in files:
        key = (d.get("image", ""), d.get("label", ""))
        if key[0] and key[1]:
            uniq[key] = d
    return list(uniq.values())


def _normalize_case_id(cid: str) -> str:
    cid = str(cid).strip()
    if cid.isdigit():
        return cid.zfill(4)
    return cid


def _extract_case_ids_from_basename(base: str) -> List[str]:
    ids4 = re.findall(r"(?<!\d)(\d{4})(?!\d)", base)
    if ids4:
        return list(ids4)

    ids: List[str] = []
    for g in re.findall(r"\d+", base):
        if len(g) <= 4:
            ids.append(g.zfill(4))
    return ids


def _match_excluded_case(d: Dict[str, Any], excluded_set: set) -> bool:
    for k in ("image", "label"):
        p = str(d.get(k, "") or "")
        base = os.path.basename(p)
        cand = _extract_case_ids_from_basename(base)
        if any(c in excluded_set for c in cand):
            return True
    return False


def load_pool_from_json(json_path: str, pool_keys: List[str], exclude_cases: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    counts_by_key: Dict[str, int] = {}

    for k in pool_keys:
        part = load_decathlon_datalist(json_path, True, k)
        counts_by_key[k] = len(part)
        pool.extend(part)

    pool = deduplicate_records(pool)

    excluded_set = set()
    if exclude_cases:
        excluded_set = {_normalize_case_id(x) for x in exclude_cases}

    if excluded_set:
        before = len(pool)
        pool = [d for d in pool if not _match_excluded_case(d, excluded_set)]
        after = len(pool)
        print(f"[POOL] excluded={sorted(list(excluded_set))} | removed={before - after} | remain={after}")

    print(f"[POOL] source_counts={counts_by_key} | unique_total={len(pool)}")
    return pool


def build_single_split_from_json(
    json_path: str,
    pool_keys: List[str],
    train_num: int,
    val_num: int,
    test_num: int,
    split_seed: int,
    exclude_cases: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    按 split_seed 对 pool 打乱后，划分为互斥的：
      1) train: 前 train_num 个
      2) val:   接下来的 val_num 个
      3) test:  接下来的 test_num 个
    """
    pool = load_pool_from_json(
        json_path=json_path,
        pool_keys=pool_keys,
        exclude_cases=exclude_cases,
    )

    required = int(train_num) + int(val_num) + int(test_num)
    if len(pool) < required:
        raise ValueError(
            f"Not enough cases after pooling/exclusion: need {required} "
            f"(train={train_num}, val={val_num}, test={test_num}), got {len(pool)}"
        )

    rng = random.Random(int(split_seed))
    rng.shuffle(pool)

    train_end = int(train_num)
    val_end = train_end + int(val_num)
    test_end = val_end + int(test_num)

    train_files = pool[:train_end]
    val_files = pool[train_end:val_end]
    test_files = pool[val_end:test_end]

    print(
        f"[SPLIT] split_seed={split_seed} | "
        f"train={len(train_files)} | val={len(val_files)} | test={len(test_files)} | val_is_test=False"
    )

    return train_files, val_files, test_files, pool


def get_transforms(args):
    pixdim = tuple(float(x) for x in args.pixdim)
    roi_size = tuple(int(x) for x in args.roi_size)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                allow_smaller=True,
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=int(args.num_samples),
                image_key="image",
                image_threshold=0,
                allow_smaller=False,
            ),
            RandAffined(
                keys=["image", "label"],
                prob=0.20,
                rotate_range=(rot, rot, rot),
                scale_range=(0.10, 0.10, 0.10),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
            RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.25),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
        ]
    )

    eval_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                allow_smaller=True,
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], track_meta=True),
        ]
    )

    return train_transforms, eval_transforms


def get_dataloaders(
    args,
    train_files: List[Dict[str, Any]],
    val_files: List[Dict[str, Any]],
    test_files: List[Dict[str, Any]],
    seed: int,
):
    train_transforms, eval_transforms = get_transforms(args)

    g = torch.Generator()
    g.manual_seed(seed)

    cache_num_train = len(train_files) if args.cache_num_train <= 0 else min(args.cache_num_train, len(train_files))
    cache_num_val = len(val_files) if args.cache_num_val <= 0 else min(args.cache_num_val, len(val_files))
    cache_num_test = len(test_files) if args.cache_num_test <= 0 else min(args.cache_num_test, len(test_files))

    pin_memory = torch.cuda.is_available()

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=cache_num_train,
        cache_rate=float(args.cache_rate),
        num_workers=int(args.num_workers_train),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers_train),
        pin_memory=pin_memory,
        collate_fn=list_data_collate,
        persistent_workers=(int(args.num_workers_train) > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=eval_transforms,
        cache_num=cache_num_val,
        cache_rate=float(args.cache_rate),
        num_workers=int(args.num_workers_val),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers_val),
        pin_memory=pin_memory,
        persistent_workers=(int(args.num_workers_val) > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_ds = CacheDataset(
        data=test_files,
        transform=eval_transforms,
        cache_num=cache_num_test,
        cache_rate=float(args.cache_rate),
        num_workers=int(args.num_workers_test),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers_test),
        pin_memory=pin_memory,
        persistent_workers=(int(args.num_workers_test) > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader, test_loader


def build_model(args, device: torch.device) -> torch.nn.Module:
    model = VNet(n_channels=1, n_classes=int(args.num_classes)).to(device)
    return model


def build_optimizer(args, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(args.lr),
        momentum=0.9,
        weight_decay=float(args.wd),
    )
    return optimizer


@torch.no_grad()
def evaluate_val_mean_fg_dice(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    sw_overlap: float,
    num_classes: int,
    post_label: AsDiscrete,
    post_pred: AsDiscrete,
    use_amp: bool = False,
) -> float:
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True)

    pbar = tqdm(loader, desc="VAL", dynamic_ncols=True, leave=False)
    for batch in pbar:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=sw_overlap,
                mode="gaussian",
            )

        labels_convert = [post_label(x) for x in decollate_batch(labels)]
        outputs_convert = [post_pred(x) for x in decollate_batch(outputs)]
        dice_metric(y_pred=outputs_convert, y=labels_convert)

    dice_agg = dice_metric.aggregate()
    dice_metric.reset()

    dice_per_class = dice_agg[0] if isinstance(dice_agg, (tuple, list)) else dice_agg
    dice_per_class = dice_per_class.detach().float().cpu()

    if num_classes > 1:
        dice_mean_fg = float(torch.nanmean(dice_per_class[1:]).item())
    else:
        dice_mean_fg = float(torch.nanmean(dice_per_class).item())

    return dice_mean_fg


@torch.no_grad()
def evaluate_loader_with_case_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    files: List[Dict[str, Any]],
    device: torch.device,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    sw_overlap: float,
    num_classes: int,
    class_names: List[str],
    post_label: AsDiscrete,
    post_pred: AsDiscrete,
    split_name: str,
    seed: int,
    spacing: Tuple[float, float, float],
    use_amp: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()

    per_case_rows: List[Dict[str, Any]] = []

    per_organ_dice: Dict[str, List[float]] = {class_names[c]: [] for c in range(1, num_classes)}
    per_organ_hd95: Dict[str, List[float]] = {class_names[c]: [] for c in range(1, num_classes)}
    per_organ_assd: Dict[str, List[float]] = {class_names[c]: [] for c in range(1, num_classes)}
    per_organ_iou: Dict[str, List[float]] = {class_names[c]: [] for c in range(1, num_classes)}

    pbar = tqdm(loader, desc=f"{split_name.upper()} EVAL(seed={seed})", dynamic_ncols=True)
    for case_idx, batch in enumerate(pbar):
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=sw_overlap,
                mode="gaussian",
            )

        labels_convert = [post_label(x) for x in decollate_batch(labels)]
        outputs_convert = [post_pred(x) for x in decollate_batch(outputs)]

        dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True)
        hd95_metric = HausdorffDistanceMetric(
            include_background=False,
            reduction="mean_batch",
            percentile=95,
            get_not_nans=True,
        )
        assd_metric = SurfaceDistanceMetric(
            include_background=False,
            symmetric=True,
            reduction="mean_batch",
            get_not_nans=True,
        )
        iou_metric = MeanIoU(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=True,
        )

        dice_metric(y_pred=outputs_convert, y=labels_convert)
        hd95_metric(y_pred=outputs_convert, y=labels_convert, spacing=spacing)
        assd_metric(y_pred=outputs_convert, y=labels_convert, spacing=spacing)
        iou_metric(y_pred=outputs_convert, y=labels_convert)

        case_dice_agg = dice_metric.aggregate()
        case_hd95_agg = hd95_metric.aggregate()
        case_assd_agg = assd_metric.aggregate()
        case_iou_agg = iou_metric.aggregate()

        dice_metric.reset()
        hd95_metric.reset()
        assd_metric.reset()
        iou_metric.reset()

        case_dice_per_class = case_dice_agg[0] if isinstance(case_dice_agg, (tuple, list)) else case_dice_agg
        case_hd95_per_class = case_hd95_agg[0] if isinstance(case_hd95_agg, (tuple, list)) else case_hd95_agg
        case_assd_per_class = case_assd_agg[0] if isinstance(case_assd_agg, (tuple, list)) else case_assd_agg
        case_iou_per_class = case_iou_agg[0] if isinstance(case_iou_agg, (tuple, list)) else case_iou_agg

        case_dice_per_class = case_dice_per_class.detach().float().cpu()
        case_hd95_per_class = case_hd95_per_class.detach().float().cpu()
        case_assd_per_class = case_assd_per_class.detach().float().cpu()
        case_iou_per_class = case_iou_per_class.detach().float().cpu()

        case_dice_mean_fg = float(torch.nanmean(case_dice_per_class[1:]).item()) if num_classes > 1 else float(
            torch.nanmean(case_dice_per_class).item()
        )
        case_hd95_mean_fg = tensor_mean_of_finite(case_hd95_per_class)
        case_assd_mean_fg = tensor_mean_of_finite(case_assd_per_class)
        case_iou_mean_fg = tensor_mean_of_finite(case_iou_per_class)

        case_id = extract_case_id_from_record(files[case_idx]) if case_idx < len(files) else f"case_{case_idx:03d}"

        row = {
            "split": split_name,
            "seed": seed,
            "case_id": case_id,
            "dice_mean_fg": csv_safe(case_dice_mean_fg),
            "hd95_mean_fg": csv_safe(case_hd95_mean_fg),
            "assd_mean_fg": csv_safe(case_assd_mean_fg),
            "iou_mean_fg": csv_safe(case_iou_mean_fg),
        }

        for c in range(1, num_classes):
            organ_name = class_names[c]

            dice_val = safe_float(case_dice_per_class[c].item())
            hd_val = safe_float(case_hd95_per_class[c - 1].item())
            assd_val = safe_float(case_assd_per_class[c - 1].item())
            iou_val = safe_float(case_iou_per_class[c - 1].item())

            row[f"dice_{organ_name}"] = "" if dice_val is None else dice_val
            row[f"hd95_{organ_name}"] = "" if hd_val is None else hd_val
            row[f"assd_{organ_name}"] = "" if assd_val is None else assd_val
            row[f"iou_{organ_name}"] = "" if iou_val is None else iou_val

            if dice_val is not None:
                per_organ_dice[organ_name].append(dice_val)
            if hd_val is not None:
                per_organ_hd95[organ_name].append(hd_val)
            if assd_val is not None:
                per_organ_assd[organ_name].append(assd_val)
            if iou_val is not None:
                per_organ_iou[organ_name].append(iou_val)

        per_case_rows.append(row)

    summary_row: Dict[str, Any] = {
        "split": split_name,
        "seed": seed,
        "num_eval_cases": len(per_case_rows),
        "dice_mean_fg_all_cases": csv_safe(mean_or_nan([r.get("dice_mean_fg") for r in per_case_rows])),
        "hd95_mean_fg_all_cases": csv_safe(mean_or_nan([r.get("hd95_mean_fg") for r in per_case_rows])),
        "assd_mean_fg_all_cases": csv_safe(mean_or_nan([r.get("assd_mean_fg") for r in per_case_rows])),
        "iou_mean_fg_all_cases": csv_safe(mean_or_nan([r.get("iou_mean_fg") for r in per_case_rows])),
    }

    for organ_name in class_names[1:]:
        summary_row[f"dice_{organ_name}_mean"] = csv_safe(mean_or_nan(per_organ_dice[organ_name]))
        summary_row[f"hd95_{organ_name}_mean"] = csv_safe(mean_or_nan(per_organ_hd95[organ_name]))
        summary_row[f"assd_{organ_name}_mean"] = csv_safe(mean_or_nan(per_organ_assd[organ_name]))
        summary_row[f"iou_{organ_name}_mean"] = csv_safe(mean_or_nan(per_organ_iou[organ_name]))

    return summary_row, per_case_rows


def train_single_run(
    args,
    output_dir: str,
    class_names: List[str],
) -> None:
    json_path = os.path.join(args.data_dir, args.split_json)

    train_files, val_files, test_files, pool_files = build_single_split_from_json(
        json_path=json_path,
        pool_keys=list(args.pool_keys),
        train_num=int(args.train_num),
        val_num=int(args.val_num),
        test_num=int(args.test_num),
        split_seed=int(args.split_seed),
        exclude_cases=list(args.exclude_cases),
    )

    save_json(os.path.join(output_dir, "pool_files_list.json"), pool_files)
    save_json(os.path.join(output_dir, "train_files_list.json"), train_files)
    save_json(os.path.join(output_dir, "val_files_list.json"), val_files)
    save_json(os.path.join(output_dir, "test_files_list.json"), test_files)

    save_json(
        os.path.join(output_dir, "split_config.json"),
        {
            "run_seed": RUN_SEED,
            "split_seed": int(args.split_seed),
            "data_dir": args.data_dir,
            "split_json": args.split_json,
            "pool_keys": list(args.pool_keys),
            "exclude_cases": list(args.exclude_cases),
            "pool_size": len(pool_files),
            "train_num": int(args.train_num),
            "val_num": int(args.val_num),
            "test_num": int(args.test_num),
            "effective_train_size": len(train_files),
            "effective_val_size": len(val_files),
            "effective_test_size": len(test_files),
            "val_is_test": False,
            "split_logic": "shuffle pooled cases with split_seed, then slice: train first, val second, test third",
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 80}")
    print(f"[RUN] seed={RUN_SEED} | split_seed={args.split_seed} | device={device} | output_dir={output_dir}")
    print(f"{'=' * 80}")

    seed_everything(RUN_SEED)

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("[WARN] cudnn_benchmark=True may reduce reproducibility.")

    pixdim = tuple(float(x) for x in args.pixdim)
    roi_size = tuple(int(x) for x in args.roi_size)
    num_classes = int(args.num_classes)
    use_amp = bool(args.amp and device.type == "cuda")

    train_loader, val_loader, test_loader = get_dataloaders(
        args=args,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        seed=RUN_SEED,
    )

    print(
        f"[DATA] train={len(train_files)} | val={len(val_files)} | test={len(test_files)} | val_is_test=False"
    )

    model = build_model(args, device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = build_optimizer(args, model)
    scaler = GradScaler(device=device.type, enabled=use_amp)

    print(f"[OPT] SGD | lr={args.lr} | wd={args.wd} | momentum=0.9")
    print("[MODEL] VSNet")
    print(f"[PATCH] roi_size={roi_size} | num_samples={args.num_samples} | batch_size={args.batch_size}")

    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    global_step = 0
    best_val_dice_fg = -1.0
    best_step = -1
    best_path = os.path.join(output_dir, "best.pt")

    num_evals = 0
    bad_evals = 0
    stop_training = False
    train_start_time = time.time()

    eval_history_rows: List[Dict[str, Any]] = []

    pbar = tqdm(total=int(args.max_iterations), desc="Training", dynamic_ncols=True)

    while (global_step < int(args.max_iterations)) and (not stop_training):
        model.train()

        for batch in train_loader:
            if global_step >= int(args.max_iterations) or stop_training:
                break

            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = loss_function(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            pbar.update(1)
            pbar.set_description(f"Training iter={global_step}/{args.max_iterations} loss={loss.item():.5f}")

            do_eval_now = ((global_step % int(args.eval_num) == 0) or (global_step == int(args.max_iterations)))
            if do_eval_now and (global_step >= int(args.val_start_iter)):
                prev_best = float(best_val_dice_fg)

                val_score = evaluate_val_mean_fg_dice(
                    model=model,
                    loader=val_loader,
                    device=device,
                    roi_size=roi_size,
                    sw_batch_size=int(args.sw_batch_size),
                    sw_overlap=float(args.sw_overlap),
                    num_classes=num_classes,
                    post_label=post_label,
                    post_pred=post_pred,
                    use_amp=use_amp,
                )
                model.train()

                eval_history_rows.append(
                    {
                        "iter": global_step,
                        "train_loss_at_eval": float(loss.item()),
                        "val_dice_mean_fg": csv_safe(val_score),
                        "val_is_test": False,
                    }
                )

                if val_score > best_val_dice_fg:
                    best_val_dice_fg = float(val_score)
                    best_step = int(global_step)
                    torch.save(model.state_dict(), best_path)
                    print(f"[SAVE] iter={best_step} | best_val_dice_fg={best_val_dice_fg:.6f}")
                else:
                    print(f"[NOSAVE] best={best_val_dice_fg:.6f} | current={val_score:.6f}")

                num_evals += 1
                improved = val_score > (prev_best + float(args.early_stop_min_delta))
                if improved:
                    bad_evals = 0
                else:
                    if num_evals > int(args.early_stop_warmup):
                        bad_evals += 1

                if (num_evals > int(args.early_stop_warmup)) and (bad_evals >= int(args.early_stop_patience)):
                    stop_training = True
                    print(
                        f"[EARLY STOP] iter={global_step} | "
                        f"bad_evals={bad_evals} | patience={args.early_stop_patience}"
                    )
                    break

    pbar.close()
    train_minutes = (time.time() - train_start_time) / 60.0
    print(f"[DONE] train_minutes={train_minutes:.2f}")

    if not os.path.isfile(best_path):
        torch.save(model.state_dict(), best_path)
        best_step = global_step
        print("[WARN] no validation checkpoint was saved; final weights are used for evaluation.")

    write_csv(os.path.join(output_dir, "eval_history.csv"), eval_history_rows)
    save_json(os.path.join(output_dir, "eval_history.json"), eval_history_rows)

    best_sd = torch.load(best_path, map_location=device)
    model.load_state_dict(best_sd, strict=True)
    model.eval()

    val_summary_row, val_per_case_rows = evaluate_loader_with_case_metrics(
        model=model,
        loader=val_loader,
        files=val_files,
        device=device,
        roi_size=roi_size,
        sw_batch_size=int(args.sw_batch_size),
        sw_overlap=float(args.sw_overlap),
        num_classes=num_classes,
        class_names=class_names,
        post_label=post_label,
        post_pred=post_pred,
        split_name="val",
        seed=RUN_SEED,
        spacing=tuple(float(x) for x in pixdim),
        use_amp=use_amp,
    )

    test_summary_row, test_per_case_rows = evaluate_loader_with_case_metrics(
        model=model,
        loader=test_loader,
        files=test_files,
        device=device,
        roi_size=roi_size,
        sw_batch_size=int(args.sw_batch_size),
        sw_overlap=float(args.sw_overlap),
        num_classes=num_classes,
        class_names=class_names,
        post_label=post_label,
        post_pred=post_pred,
        split_name="test",
        seed=RUN_SEED,
        spacing=tuple(float(x) for x in pixdim),
        use_amp=use_amp,
    )

    val_summary_row["num_train_cases"] = len(train_files)
    val_summary_row["best_val_dice_fg"] = csv_safe(best_val_dice_fg)
    val_summary_row["best_step"] = best_step
    val_summary_row["val_is_test"] = False

    test_summary_row["num_train_cases"] = len(train_files)
    test_summary_row["best_val_dice_fg"] = csv_safe(best_val_dice_fg)
    test_summary_row["best_step"] = best_step
    test_summary_row["val_is_test"] = False

    write_csv(os.path.join(output_dir, "val_summary.csv"), [val_summary_row])
    write_csv(os.path.join(output_dir, "val_per_case.csv"), val_per_case_rows)
    write_csv(os.path.join(output_dir, "test_summary.csv"), [test_summary_row])
    write_csv(os.path.join(output_dir, "test_per_case.csv"), test_per_case_rows)

    final_summary = {
        "run_seed": RUN_SEED,
        "split_seed": int(args.split_seed),
        "train_size": len(train_files),
        "val_size": len(val_files),
        "test_size": len(test_files),
        "val_is_test": False,
        "best_step": best_step,
        "best_val_dice_fg": csv_safe(best_val_dice_fg),
        "train_minutes": csv_safe(train_minutes),
        "val_dice_mean_fg_all_cases": val_summary_row.get("dice_mean_fg_all_cases", ""),
        "val_hd95_mean_fg_all_cases": val_summary_row.get("hd95_mean_fg_all_cases", ""),
        "val_assd_mean_fg_all_cases": val_summary_row.get("assd_mean_fg_all_cases", ""),
        "val_iou_mean_fg_all_cases": val_summary_row.get("iou_mean_fg_all_cases", ""),
        "test_dice_mean_fg_all_cases": test_summary_row.get("dice_mean_fg_all_cases", ""),
        "test_hd95_mean_fg_all_cases": test_summary_row.get("hd95_mean_fg_all_cases", ""),
        "test_assd_mean_fg_all_cases": test_summary_row.get("assd_mean_fg_all_cases", ""),
        "test_iou_mean_fg_all_cases": test_summary_row.get("iou_mean_fg_all_cases", ""),
        "selection_metric": "val foreground Dice",
        "evaluation_metrics": ["Dice", "HD95", "ASSD", "IoU"],
        "split_logic": "shuffle pooled cases with split_seed, then slice: train first, val second, test third",
    }

    write_csv(os.path.join(output_dir, "single_run_summary.csv"), [final_summary])
    save_json(os.path.join(output_dir, "single_run_summary.json"), final_summary)

    print("\nSaved artifacts:")
    print(" -", os.path.join(output_dir, "pool_files_list.json"))
    print(" -", os.path.join(output_dir, "split_config.json"))
    print(" -", os.path.join(output_dir, "train_files_list.json"))
    print(" -", os.path.join(output_dir, "val_files_list.json"))
    print(" -", os.path.join(output_dir, "test_files_list.json"))
    print(" -", os.path.join(output_dir, "best.pt"))
    print(" -", os.path.join(output_dir, "eval_history.csv"))
    print(" -", os.path.join(output_dir, "eval_history.json"))
    print(" -", os.path.join(output_dir, "val_summary.csv"))
    print(" -", os.path.join(output_dir, "val_per_case.csv"))
    print(" -", os.path.join(output_dir, "test_summary.csv"))
    print(" -", os.path.join(output_dir, "test_per_case.csv"))
    print(" -", os.path.join(output_dir, "single_run_summary.csv"))
    print(" -", os.path.join(output_dir, "single_run_summary.json"))

    print("\nSingle-run summary:")
    print(final_summary)


def main():
    args = parse_args()

    run_id = args.run_name.strip() if args.run_name.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_root,
        f"btcv_single_run_seed{RUN_SEED}_splitseed{args.split_seed}_{run_id}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print_config()
    print(f"output_dir = {output_dir}")
    print(f"run_seed = {RUN_SEED}")
    print(f"split_seed = {args.split_seed}")
    print(f"train_num = {args.train_num}")
    print(f"val_num = {args.val_num}")
    print(f"test_num = {args.test_num}")
    print("model = basic")
    print("optimizer = SGD")

    num_classes = int(args.num_classes)
    class_names = get_class_names(num_classes)

    train_single_run(
        args=args,
        output_dir=output_dir,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()