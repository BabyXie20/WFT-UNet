import os
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
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
)
from networks import VNet

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
CV_SPLIT_SEED = 123
rot = np.deg2rad(30.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/btcv")
    parser.add_argument("--split_json", type=str, default="dataset_0.json")
    parser.add_argument("--output_root", type=str, default="./outputs_VNet")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--pixdim", type=float, nargs=3, default=[1.5, 1.5, 2.0])
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--cache_num_train", type=int, default=24)
    parser.add_argument("--cache_num_eval", type=int, default=6)
    parser.add_argument("--cache_rate", type=float, default=1.0)
    parser.add_argument("--num_workers_train", type=int, default=8)
    parser.add_argument("--num_workers_eval", type=int, default=4)
    parser.add_argument("--max_iterations", type=int, default=34000)
    parser.add_argument("--eval_num", type=int, default=400)
    parser.add_argument("--val_start_iter", type=int, default=12000)
    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--sw_overlap", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=10)
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


def load_cv_pool_from_json(json_path: str) -> List[Dict[str, Any]]:
    train_files = load_decathlon_datalist(json_path, True, "training")
    val_files = load_decathlon_datalist(json_path, True, "validation")
    pool = deduplicate_records(train_files + val_files)

    print(f"[POOL] training={len(train_files)} | validation={len(val_files)} | unique_total={len(pool)}")
    if len(pool) < 5:
        raise ValueError(f"Need at least 5 cases for 5-fold CV, but got {len(pool)}")
    return pool


def build_kfold_splits(
    files: List[Dict[str, Any]],
    n_splits: int,
    split_seed: int,
) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, but got {n_splits}")
    if len(files) < n_splits:
        raise ValueError(f"Number of files ({len(files)}) must be >= n_splits ({n_splits})")

    indices = list(range(len(files)))
    rng = random.Random(int(split_seed))
    rng.shuffle(indices)

    fold_sizes = [len(files) // n_splits] * n_splits
    for i in range(len(files) % n_splits):
        fold_sizes[i] += 1

    splits: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    current = 0
    for fold_idx, fold_size in enumerate(fold_sizes):
        val_idx = indices[current: current + fold_size]
        val_idx_set = set(val_idx)
        train_idx = [i for i in indices if i not in val_idx_set]
        current += fold_size

        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        splits.append((train_files, val_files))

        print(f"[FOLD {fold_idx + 1}] train={len(train_files)} | val={len(val_files)}")

    return splits


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
    eval_files: List[Dict[str, Any]],
    seed: int,
):
    train_transforms, eval_transforms = get_transforms(args)

    g = torch.Generator()
    g.manual_seed(seed)

    cache_num_train = len(train_files) if args.cache_num_train <= 0 else min(args.cache_num_train, len(train_files))
    cache_num_eval = len(eval_files) if args.cache_num_eval <= 0 else min(args.cache_num_eval, len(eval_files))

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

    eval_ds = CacheDataset(
        data=eval_files,
        transform=eval_transforms,
        cache_num=cache_num_eval,
        cache_rate=float(args.cache_rate),
        num_workers=int(args.num_workers_eval),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers_eval),
        pin_memory=pin_memory,
        persistent_workers=(int(args.num_workers_eval) > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, eval_loader


def build_model(args, device: torch.device) -> torch.nn.Module:
    model = VNet(
        n_channels=1,
        n_classes=int(args.num_classes),
    ).to(device)
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
) -> float:
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True)

    pbar = tqdm(loader, desc="VAL", dynamic_ncols=True, leave=False)
    for batch in pbar:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

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
    fold_id: int,
    seed: int,
    spacing: Tuple[float, float, float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()

    per_case_rows: List[Dict[str, Any]] = []

    per_organ_dice: Dict[str, List[float]] = {class_names[c]: [] for c in range(1, num_classes)}
    per_organ_hd95: Dict[str, List[float]] = {class_names[c]: [] for c in range(1, num_classes)}
    per_organ_assd: Dict[str, List[float]] = {class_names[c]: [] for c in range(1, num_classes)}
    per_organ_iou: Dict[str, List[float]] = {class_names[c]: [] for c in range(1, num_classes)}

    pbar = tqdm(loader, desc=f"FOLD-{fold_id} EVAL(seed={seed})", dynamic_ncols=True)
    for case_idx, batch in enumerate(pbar):
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

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

        case_dice_mean_fg = float(torch.nanmean(case_dice_per_class[1:]).item())
        case_hd95_mean_fg = tensor_mean_of_finite(case_hd95_per_class)
        case_assd_mean_fg = tensor_mean_of_finite(case_assd_per_class)
        case_iou_mean_fg = tensor_mean_of_finite(case_iou_per_class)

        case_id = extract_case_id_from_record(files[case_idx]) if case_idx < len(files) else f"case_{case_idx:03d}"

        row = {
            "fold": fold_id,
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
        "fold": fold_id,
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


def train_one_fold(
    args,
    fold_dir: str,
    fold_id: int,
    fold_seed: int,
    train_files: List[Dict[str, Any]],
    eval_files: List[Dict[str, Any]],
    class_names: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    os.makedirs(fold_dir, exist_ok=True)
    save_json(os.path.join(fold_dir, "train_files_list.json"), train_files)
    save_json(os.path.join(fold_dir, "val_files_list.json"), eval_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 80}")
    print(f"[FOLD {fold_id}] seed={fold_seed} | device={device} | fold_dir={fold_dir}")
    print(f"{'=' * 80}")

    seed_everything(fold_seed)

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("[WARN] cudnn_benchmark=True may reduce reproducibility.")

    pixdim = tuple(float(x) for x in args.pixdim)
    roi_size = tuple(int(x) for x in args.roi_size)
    num_classes = int(args.num_classes)
    use_amp = bool(args.amp and device.type == "cuda")

    train_loader, eval_loader = get_dataloaders(
        args=args,
        train_files=train_files,
        eval_files=eval_files,
        seed=fold_seed,
    )

    print(f"[DATA] fold={fold_id} | train={len(train_files)} | eval={len(eval_files)}")

    model = build_model(args, device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = build_optimizer(args, model)
    scaler = GradScaler(device=device.type, enabled=use_amp)

    print(f"[OPT] SGD | lr={args.lr} | wd={args.wd} | momentum=0.9")
    print("[MODEL] VNet")

    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    global_step = 0
    best_val_dice_fg = -1.0
    best_step = -1
    best_path = os.path.join(fold_dir, "best.pt")

    num_evals = 0
    bad_evals = 0
    stop_training = False

    pbar = tqdm(total=int(args.max_iterations), desc=f"Fold-{fold_id} Training", dynamic_ncols=True)
    t0 = time.time()

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
            pbar.set_description(
                f"Fold-{fold_id} Training iter={global_step}/{args.max_iterations} loss={loss.item():.5f}"
            )

            do_eval_now = ((global_step % int(args.eval_num) == 0) or (global_step == int(args.max_iterations)))
            if do_eval_now and (global_step >= int(args.val_start_iter)):
                prev_best = float(best_val_dice_fg)

                val_score = evaluate_val_mean_fg_dice(
                    model=model,
                    loader=eval_loader,
                    device=device,
                    roi_size=roi_size,
                    sw_batch_size=int(args.sw_batch_size),
                    sw_overlap=float(args.sw_overlap),
                    num_classes=num_classes,
                    post_label=post_label,
                    post_pred=post_pred,
                )
                model.train()

                if val_score > best_val_dice_fg:
                    best_val_dice_fg = float(val_score)
                    best_step = int(global_step)
                    torch.save(model.state_dict(), best_path)
                    print(
                        f"[SAVE] fold={fold_id} | iter={best_step} | best_val_dice_fg={best_val_dice_fg:.6f}"
                    )
                else:
                    print(
                        f"[NOSAVE] fold={fold_id} | best={best_val_dice_fg:.6f} | current={val_score:.6f}"
                    )

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
                        f"[EARLY STOP] fold={fold_id} | iter={global_step} | "
                        f"bad_evals={bad_evals} | patience={args.early_stop_patience}"
                    )
                    break

    pbar.close()
    print(f"[DONE] fold={fold_id} | train_minutes={(time.time() - t0) / 60.0:.2f}")

    if not os.path.isfile(best_path):
        torch.save(model.state_dict(), best_path)
        best_step = global_step
        print("[WARN] no validation checkpoint was saved; final weights are used for fold evaluation.")

    best_sd = torch.load(best_path, map_location=device)
    model.load_state_dict(best_sd, strict=True)
    model.eval()

    summary_row, per_case_rows = evaluate_loader_with_case_metrics(
        model=model,
        loader=eval_loader,
        files=eval_files,
        device=device,
        roi_size=roi_size,
        sw_batch_size=int(args.sw_batch_size),
        sw_overlap=float(args.sw_overlap),
        num_classes=num_classes,
        class_names=class_names,
        post_label=post_label,
        post_pred=post_pred,
        fold_id=fold_id,
        seed=fold_seed,
        spacing=tuple(float(x) for x in pixdim),
    )

    summary_row["num_train_cases"] = len(train_files)
    summary_row["best_val_dice_fg"] = csv_safe(best_val_dice_fg)
    summary_row["best_step"] = best_step

    write_csv(os.path.join(fold_dir, "fold_summary.csv"), [summary_row])
    write_csv(os.path.join(fold_dir, "fold_per_case.csv"), per_case_rows)

    return summary_row, per_case_rows


def summarize_crossval_results(
    fold_summary_rows: List[Dict[str, Any]],
    all_case_rows: List[Dict[str, Any]],
    class_names: List[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_folds": len(fold_summary_rows),
        "num_all_cases": len(all_case_rows),
        "dice_mean_fg_fold_mean": csv_safe(mean_or_nan([r.get("dice_mean_fg_all_cases") for r in fold_summary_rows])),
        "dice_mean_fg_fold_std": csv_safe(std_or_nan([r.get("dice_mean_fg_all_cases") for r in fold_summary_rows])),
        "hd95_mean_fg_fold_mean": csv_safe(mean_or_nan([r.get("hd95_mean_fg_all_cases") for r in fold_summary_rows])),
        "hd95_mean_fg_fold_std": csv_safe(std_or_nan([r.get("hd95_mean_fg_all_cases") for r in fold_summary_rows])),
        "assd_mean_fg_fold_mean": csv_safe(mean_or_nan([r.get("assd_mean_fg_all_cases") for r in fold_summary_rows])),
        "assd_mean_fg_fold_std": csv_safe(std_or_nan([r.get("assd_mean_fg_all_cases") for r in fold_summary_rows])),
        "iou_mean_fg_fold_mean": csv_safe(mean_or_nan([r.get("iou_mean_fg_all_cases") for r in fold_summary_rows])),
        "iou_mean_fg_fold_std": csv_safe(std_or_nan([r.get("iou_mean_fg_all_cases") for r in fold_summary_rows])),
        "dice_mean_fg_case_mean": csv_safe(mean_or_nan([r.get("dice_mean_fg") for r in all_case_rows])),
        "dice_mean_fg_case_std": csv_safe(std_or_nan([r.get("dice_mean_fg") for r in all_case_rows])),
        "hd95_mean_fg_case_mean": csv_safe(mean_or_nan([r.get("hd95_mean_fg") for r in all_case_rows])),
        "hd95_mean_fg_case_std": csv_safe(std_or_nan([r.get("hd95_mean_fg") for r in all_case_rows])),
        "assd_mean_fg_case_mean": csv_safe(mean_or_nan([r.get("assd_mean_fg") for r in all_case_rows])),
        "assd_mean_fg_case_std": csv_safe(std_or_nan([r.get("assd_mean_fg") for r in all_case_rows])),
        "iou_mean_fg_case_mean": csv_safe(mean_or_nan([r.get("iou_mean_fg") for r in all_case_rows])),
        "iou_mean_fg_case_std": csv_safe(std_or_nan([r.get("iou_mean_fg") for r in all_case_rows])),
        "best_val_dice_fg_fold_mean": csv_safe(mean_or_nan([r.get("best_val_dice_fg") for r in fold_summary_rows])),
        "best_val_dice_fg_fold_std": csv_safe(std_or_nan([r.get("best_val_dice_fg") for r in fold_summary_rows])),
    }

    for organ_name in class_names[1:]:
        summary[f"dice_{organ_name}_fold_mean"] = csv_safe(
            mean_or_nan([r.get(f"dice_{organ_name}_mean") for r in fold_summary_rows])
        )
        summary[f"dice_{organ_name}_fold_std"] = csv_safe(
            std_or_nan([r.get(f"dice_{organ_name}_mean") for r in fold_summary_rows])
        )
        summary[f"hd95_{organ_name}_fold_mean"] = csv_safe(
            mean_or_nan([r.get(f"hd95_{organ_name}_mean") for r in fold_summary_rows])
        )
        summary[f"hd95_{organ_name}_fold_std"] = csv_safe(
            std_or_nan([r.get(f"hd95_{organ_name}_mean") for r in fold_summary_rows])
        )
        summary[f"assd_{organ_name}_fold_mean"] = csv_safe(
            mean_or_nan([r.get(f"assd_{organ_name}_mean") for r in fold_summary_rows])
        )
        summary[f"assd_{organ_name}_fold_std"] = csv_safe(
            std_or_nan([r.get(f"assd_{organ_name}_mean") for r in fold_summary_rows])
        )
        summary[f"iou_{organ_name}_fold_mean"] = csv_safe(
            mean_or_nan([r.get(f"iou_{organ_name}_mean") for r in fold_summary_rows])
        )
        summary[f"iou_{organ_name}_fold_std"] = csv_safe(
            std_or_nan([r.get(f"iou_{organ_name}_mean") for r in fold_summary_rows])
        )

    return summary


def run_cross_validation(args, output_dir: str, class_names: List[str]) -> None:
    json_path = os.path.join(args.data_dir, args.split_json)
    pool_files = load_cv_pool_from_json(json_path)
    cv_splits = build_kfold_splits(pool_files, int(args.n_splits), CV_SPLIT_SEED)

    save_json(os.path.join(output_dir, "cv_pool_files_list.json"), pool_files)
    save_json(
        os.path.join(output_dir, "cv_split_config.json"),
        {
            "run_seed": RUN_SEED,
            "cv_split_seed": CV_SPLIT_SEED,
            "n_splits": int(args.n_splits),
            "data_dir": args.data_dir,
            "split_json": args.split_json,
            "pool_source_keys": ["training", "validation"],
            "pool_size": len(pool_files),
            "pixdim": args.pixdim,
            "roi_size": args.roi_size,
            "num_classes": args.num_classes,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "cache_num_train": args.cache_num_train,
            "cache_num_eval": args.cache_num_eval,
            "cache_rate": args.cache_rate,
            "num_workers_train": args.num_workers_train,
            "num_workers_eval": args.num_workers_eval,
            "max_iterations": args.max_iterations,
            "eval_num": args.eval_num,
            "val_start_iter": args.val_start_iter,
            "sw_batch_size": args.sw_batch_size,
            "sw_overlap": args.sw_overlap,
            "lr": args.lr,
            "wd": args.wd,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "early_stop_warmup": args.early_stop_warmup,
            "amp": args.amp,
            "cudnn_benchmark": args.cudnn_benchmark,
            "metrics": ["Dice", "HD95", "ASSD", "IoU"],
            "model_config": {
                "name": "VNet",
                "n_channels": 1,
                "n_classes": int(args.num_classes),
                "patch_size": 96,
                "n_filters": 16,
                "normalization": "instancenorm",
                "has_dropout": False,
                "has_residual": False,
            },
            "optimizer_config": {
                "name": "SGD",
                "lr": args.lr,
                "momentum": 0.9,
                "weight_decay": args.wd,
            },
            "note": "Standard 5-fold CV: each fold's held-out split is used for checkpoint selection and fold evaluation.",
        },
    )

    fold_summary_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_files, eval_files) in enumerate(cv_splits):
        fold_id = fold_idx + 1
        fold_seed = RUN_SEED + fold_idx
        fold_dir = os.path.join(output_dir, f"fold_{fold_id}")

        summary_row, per_case_rows = train_one_fold(
            args=args,
            fold_dir=fold_dir,
            fold_id=fold_id,
            fold_seed=fold_seed,
            train_files=train_files,
            eval_files=eval_files,
            class_names=class_names,
        )
        fold_summary_rows.append(summary_row)
        all_case_rows.extend(per_case_rows)

    overall_summary = summarize_crossval_results(
        fold_summary_rows=fold_summary_rows,
        all_case_rows=all_case_rows,
        class_names=class_names,
    )

    write_csv(os.path.join(output_dir, "crossval_fold_summary.csv"), fold_summary_rows)
    write_csv(os.path.join(output_dir, "crossval_all_cases.csv"), all_case_rows)
    write_csv(os.path.join(output_dir, "crossval_overall_summary.csv"), [overall_summary])
    save_json(os.path.join(output_dir, "crossval_overall_summary.json"), overall_summary)

    print("\nSaved artifacts:")
    print(" -", os.path.join(output_dir, "cv_pool_files_list.json"))
    print(" -", os.path.join(output_dir, "cv_split_config.json"))
    for fold_idx in range(len(cv_splits)):
        fold_id = fold_idx + 1
        print(" -", os.path.join(output_dir, f"fold_{fold_id}", "train_files_list.json"))
        print(" -", os.path.join(output_dir, f"fold_{fold_id}", "val_files_list.json"))
        print(" -", os.path.join(output_dir, f"fold_{fold_id}", "best.pt"))
        print(" -", os.path.join(output_dir, f"fold_{fold_id}", "fold_summary.csv"))
        print(" -", os.path.join(output_dir, f"fold_{fold_id}", "fold_per_case.csv"))
    print(" -", os.path.join(output_dir, "crossval_fold_summary.csv"))
    print(" -", os.path.join(output_dir, "crossval_all_cases.csv"))
    print(" -", os.path.join(output_dir, "crossval_overall_summary.csv"))
    print(" -", os.path.join(output_dir, "crossval_overall_summary.json"))

    print("\nCross-validation overall summary:")
    print(overall_summary)
    print(f"num_all_cases = {len(all_case_rows)}")


def main():
    args = parse_args()

    run_id = args.run_name.strip() if args.run_name.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_root,
        f"btcv_vnet_{args.n_splits}foldcv_runseed{RUN_SEED}_splitseed{CV_SPLIT_SEED}_{run_id}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print_config()
    print(f"output_dir = {output_dir}")
    print(f"run_seed = {RUN_SEED}")
    print(f"cv_split_seed = {CV_SPLIT_SEED}")
    print(f"n_splits = {args.n_splits}")
    print("model = VNet")
    print("optimizer = SGD")

    num_classes = int(args.num_classes)
    class_names = get_class_names(num_classes)

    run_cross_validation(
        args=args,
        output_dir=output_dir,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()