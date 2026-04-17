import os
import re
import json
import time
import math
import csv
import argparse
import tempfile
import random
import shutil
import sys
import platform
import hashlib
import inspect
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

import torch
from torch.utils.tensorboard import SummaryWriter
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
    EnsureChannelFirstd
)

from networks.model import WFT_UNet

rot = np.deg2rad(30.0)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/btcv")
    parser.add_argument("--split_json", type=str, default="dataset_0.json")
    parser.add_argument("--output_root", type=str, default="./outputs_btcv")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--pixdim", type=float, nargs=3, default=[1.5, 1.5, 2.0])
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--cache_num_train", type=int, default=18)
    parser.add_argument("--cache_num_val", type=int, default=12)
    parser.add_argument("--cache_num_test", type=int, default=12)
    parser.add_argument("--cache_rate", type=float, default=1.0)
    parser.add_argument("--num_workers_train", type=int, default=8)
    parser.add_argument("--num_workers_val", type=int, default=6)
    parser.add_argument("--num_workers_test", type=int, default=6)
    parser.add_argument("--max_iterations", type=int, default=34000)
    parser.add_argument("--eval_num", type=int, default=400)
    parser.add_argument("--val_start_iter", type=int, default=8800)
    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--sw_overlap", type=float, default=0.5)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9) 
    parser.add_argument("--sgd_nesterov", action="store_true")
    parser.add_argument("--adamw_betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--adamw_eps", type=float, default=1e-8)
    parser.add_argument("--use_poly", action="store_true")
    parser.add_argument("--poly_power", type=float, default=0.9)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stop_patience", type=int, default=11)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    parser.add_argument("--early_stop_warmup", type=int, default=0)
    parser.add_argument(
        "--pool_keys",
        type=str,
        nargs="+",
        default=["training", "validation"],
    )
    parser.add_argument("--train_num", type=int, default=18)
    parser.add_argument("--val_num", type=int, default=12)
    parser.add_argument("--test_num", type=int, default=12)
    parser.add_argument("--val_separate", action="store_true")
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument(
        "--exclude_cases",
        type=str,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--snapshot_extra",
        type=str,
        nargs="*",
        default=["networks"],
    )
    parser.add_argument("--seed", type=int, default=123)
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
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError("safe_float only accepts scalar tensors.")
        x = x.item()
    if isinstance(x, (np.floating, float)):
        if math.isnan(float(x)) or math.isinf(float(x)):
            return None
        return float(x)
    if isinstance(x, (np.integer, int)):
        return float(x)
    return float(x)


def tensor_to_float_list(t: torch.Tensor) -> List[Optional[float]]:
    t = t.detach().float().cpu()
    out: List[Optional[float]] = []
    for v in t.tolist():
        out.append(safe_float(v))
    return out


def get_class_names(num_classes: int) -> List[str]:
    names = []
    for i in range(num_classes):
        key = str(i)
        names.append(CLASS_LABELS.get(key, f"class_{i}"))
    return names


def get_foreground_class_names(class_names: List[str]) -> List[str]:
    return class_names[1:] if len(class_names) > 1 else []


def _safe_run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 5) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        return int(p.returncode), p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 999, "", f"{type(e).__name__}: {e}"


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ignore_snapshot_files(dirpath: str, names: List[str]) -> set:
    ignore = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        ".idea",
        ".vscode",
        "wandb",
        ".DS_Store",
    }
    out = set()
    for n in names:
        if n in ignore:
            out.add(n)
        elif n.endswith((".pyc", ".pyo", ".so")):
            out.add(n)
    return out


def save_code_snapshot(output_dir: str, extra_rel_paths: Optional[List[str]] = None) -> None:
    snapshot_dir = os.path.join(output_dir, "snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)

    script_path = None
    try:
        script_path = os.path.abspath(__file__)
    except Exception:
        script_path = None

    if script_path and os.path.isfile(script_path):
        base_dir = os.path.dirname(script_path)
    else:
        base_dir = os.getcwd()

    copied: List[Dict[str, Any]] = []

    if script_path and os.path.isfile(script_path):
        dst = os.path.join(snapshot_dir, os.path.basename(script_path))
        shutil.copy2(script_path, dst)
        copied.append({"type": "file", "src": script_path, "dst": dst, "sha256": _sha256_file(dst)})
    else:
        try:
            src_text = inspect.getsource(sys.modules[__name__])
            dst = os.path.join(snapshot_dir, "train_script_snapshot.py")
            with open(dst, "w", encoding="utf-8") as f:
                f.write(src_text)
            copied.append({"type": "file_generated", "src": "<inspect.getsource>", "dst": dst, "sha256": _sha256_file(dst)})
        except Exception as e:
            print(f"[SNAPSHOT][WARN] cannot save script source: {type(e).__name__}: {e}")

    extra_rel_paths = extra_rel_paths or []
    for rel in extra_rel_paths:
        if not rel:
            continue
        src = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        if not os.path.exists(src):
            print(f"[SNAPSHOT][WARN] extra path not found: {src}")
            continue

        dst = os.path.join(snapshot_dir, os.path.basename(src))
        try:
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst, ignore=_ignore_snapshot_files)
                copied.append({"type": "dir", "src": src, "dst": dst})
            else:
                shutil.copy2(src, dst)
                copied.append({"type": "file", "src": src, "dst": dst, "sha256": _sha256_file(dst)})
        except Exception as e:
            print(f"[SNAPSHOT][WARN] failed copying {src} -> {dst}: {type(e).__name__}: {e}")

    git_commit = ""
    git_dirty = None
    git_root = None
    rc, out, _ = _safe_run_cmd(["git", "rev-parse", "--show-toplevel"], cwd=base_dir)
    if rc == 0 and out:
        git_root = out
        rc2, out2, _ = _safe_run_cmd(["git", "rev-parse", "HEAD"], cwd=git_root)
        if rc2 == 0:
            git_commit = out2.strip()
        rc3, out3, _ = _safe_run_cmd(["git", "status", "--porcelain"], cwd=git_root)
        if rc3 == 0:
            git_dirty = (len(out3.strip()) > 0)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": os.path.abspath(output_dir),
        "snapshot_dir": os.path.abspath(snapshot_dir),
        "script_path": os.path.abspath(script_path) if script_path else "",
        "base_dir": os.path.abspath(base_dir) if base_dir else "",
        "cmdline": sys.argv,
        "python": {"version": sys.version.replace("\n", " "), "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": {
            "torch": getattr(torch, "__version__", ""),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", ""),
            "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
            "monai": "",
        },
        "git": {"root": git_root or "", "commit": git_commit, "dirty": git_dirty},
        "copied": copied,
    }

    try:
        import monai  # noqa: F401
        meta["packages"]["monai"] = getattr(monai, "__version__", "")
    except Exception:
        meta["packages"]["monai"] = ""

    with open(os.path.join(snapshot_dir, "snapshot_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SNAPSHOT] saved to: {snapshot_dir}")
    print(f"[SNAPSHOT] items: {len(copied)} (meta written: snapshot_meta.json)")


def _normalize_case_id(cid: str) -> str:
    cid = str(cid).strip()
    if cid.isdigit():
        return cid.zfill(4)
    return cid


def _extract_case_ids_from_basename(base: str) -> List[str]:
    ids: List[str] = []
    ids4 = re.findall(r"(?<!\d)(\d{4})(?!\d)", base)
    if ids4:
        return [x for x in ids4]
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


def build_custom_splits_from_json(
    json_path: str,
    pool_keys: List[str],
    train_num: int,
    val_num: int,
    test_num: int,
    split_seed: int,
    exclude_cases: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pool: List[Dict[str, Any]] = []
    for k in pool_keys:
        part = load_decathlon_datalist(json_path, True, k)
        pool.extend(part)

    uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in pool:
        key = (d.get("image", ""), d.get("label", ""))
        if not key[0] or not key[1]:
            continue
        uniq[key] = d
    pool = list(uniq.values())

    excluded_set = set()
    if exclude_cases:
        excluded_set = {_normalize_case_id(x) for x in exclude_cases}

    if excluded_set:
        before = len(pool)
        pool = [d for d in pool if not _match_excluded_case(d, excluded_set)]
        after = len(pool)
        print(f"[SPLIT] excluded cases={sorted(list(excluded_set))} | removed={before - after} | remain={after}")

    rng = random.Random(int(split_seed))
    rng.shuffle(pool)

    total = len(pool)
    val_num = max(0, int(val_num))
    test_num = max(0, int(test_num))
    train_num = int(train_num)

    if val_num + test_num > total:
        overflow = val_num + test_num - total
        test_num = max(0, test_num - overflow)

    remain = total - (val_num + test_num)
    if train_num < 0:
        train_num = remain
    else:
        train_num = min(train_num, remain)

    val_files = pool[:val_num]
    test_files = pool[val_num: val_num + test_num]
    train_files = pool[val_num + test_num: val_num + test_num + train_num]
    return train_files, val_files, test_files


def _get_case_id(batch: Dict[str, Any]) -> str:
    try:
        md = batch.get("image_meta_dict", None)
        if md and "filename_or_obj" in md:
            fn = md["filename_or_obj"][0]
            return str(fn)
    except Exception:
        pass
    return "unknown_case"


def dice_per_class_onehot(
    y_pred_1hot: torch.Tensor,
    y_true_1hot: torch.Tensor,
    eps: float = 1e-8,
    ignore_empty: bool = True,
) -> torch.Tensor:
    if y_pred_1hot.dim() == 5:
        y_pred_1hot = y_pred_1hot[0]
    if y_true_1hot.dim() == 5:
        y_true_1hot = y_true_1hot[0]

    y_pred = y_pred_1hot.float()
    y_true = y_true_1hot.float()

    dims = tuple(range(1, y_pred.dim()))
    inter = (y_pred * y_true).sum(dim=dims)
    pred_sum = y_pred.sum(dim=dims)
    true_sum = y_true.sum(dim=dims)
    denom = pred_sum + true_sum

    dice = (2.0 * inter) / (denom + eps)

    if ignore_empty:
        nan = torch.tensor(float("nan"), device=dice.device, dtype=dice.dtype)
        dice = torch.where(true_sum > 0, dice, nan)
    else:
        dice = torch.where(denom > 0, dice, torch.ones_like(dice))
    return dice


def iou_per_class_onehot(
    y_pred_1hot: torch.Tensor,
    y_true_1hot: torch.Tensor,
    eps: float = 1e-8,
    ignore_empty: bool = True,
) -> torch.Tensor:
    if y_pred_1hot.dim() == 5:
        y_pred_1hot = y_pred_1hot[0]
    if y_true_1hot.dim() == 5:
        y_true_1hot = y_true_1hot[0]

    y_pred = y_pred_1hot.float()
    y_true = y_true_1hot.float()

    dims = tuple(range(1, y_pred.dim()))
    inter = (y_pred * y_true).sum(dim=dims)
    pred_sum = y_pred.sum(dim=dims)
    true_sum = y_true.sum(dim=dims)
    union = pred_sum + true_sum - inter

    iou = inter / (union + eps)

    if ignore_empty:
        nan = torch.tensor(float("nan"), device=iou.device, dtype=iou.dtype)
        iou = torch.where(true_sum > 0, iou, nan)
    else:
        iou = torch.where(union > 0, iou, torch.ones_like(iou))
    return iou


def _compute_binary_hd95_mm(pred_mask: np.ndarray, true_mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    if (not pred.any()) or (not true.any()):
        return float("nan")

    conn = generate_binary_structure(rank=pred.ndim, connectivity=1)

    pred_eroded = binary_erosion(pred, structure=conn, border_value=0)
    true_eroded = binary_erosion(true, structure=conn, border_value=0)

    pred_surface = np.logical_xor(pred, pred_eroded)
    true_surface = np.logical_xor(true, true_eroded)

    if not pred_surface.any():
        pred_surface = pred
    if not true_surface.any():
        true_surface = true

    dt_to_true = distance_transform_edt(~true_surface, sampling=spacing)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    dist_pred_to_true = dt_to_true[pred_surface]
    dist_true_to_pred = dt_to_pred[true_surface]

    if dist_pred_to_true.size == 0 or dist_true_to_pred.size == 0:
        return float("nan")

    all_dist = np.concatenate([dist_pred_to_true, dist_true_to_pred], axis=0)
    if all_dist.size == 0:
        return float("nan")
    return float(np.percentile(all_dist, 95.0))


def hd95_per_class_onehot(
    y_pred_1hot: torch.Tensor,
    y_true_1hot: torch.Tensor,
    spacing: Tuple[float, float, float],
    include_background: bool = False,
    ignore_empty: bool = True,
) -> torch.Tensor:
    if y_pred_1hot.dim() == 5:
        y_pred_1hot = y_pred_1hot[0]
    if y_true_1hot.dim() == 5:
        y_true_1hot = y_true_1hot[0]

    y_pred_np = y_pred_1hot.detach().cpu().numpy() > 0.5
    y_true_np = y_true_1hot.detach().cpu().numpy() > 0.5

    c_start = 0 if include_background else 1
    values: List[float] = []

    for c in range(c_start, y_pred_np.shape[0]):
        pred_c = y_pred_np[c]
        true_c = y_true_np[c]

        if ignore_empty and (not true_c.any()):
            values.append(float("nan"))
            continue

        values.append(_compute_binary_hd95_mm(pred_c, true_c, spacing=spacing))

    return torch.tensor(values, dtype=torch.float32)


@torch.no_grad()
def run_evaluation_single_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    sw_overlap: float,
    num_classes: int,
    class_names: List[str],
    post_label: AsDiscrete,
    post_pred: AsDiscrete,
    global_step: int,
    spacing: Tuple[float, float, float],
    writer: Optional[SummaryWriter] = None,
    prefix: str = "val",
    compute_hd95: bool = False,
    compute_iou: bool = True,
) -> Dict[str, Any]:
    model.eval()

    sum_dice = torch.zeros((num_classes,), dtype=torch.float64)
    cnt_dice = torch.zeros((num_classes,), dtype=torch.float64)

    sum_iou = torch.zeros((num_classes,), dtype=torch.float64)
    cnt_iou = torch.zeros((num_classes,), dtype=torch.float64)

    if compute_hd95:
        sum_hd95 = torch.zeros((max(num_classes - 1, 0),), dtype=torch.float64)
        cnt_hd95 = torch.zeros((max(num_classes - 1, 0),), dtype=torch.float64)
    else:
        sum_hd95 = None
        cnt_hd95 = None

    pbar = tqdm(loader, desc=f"{prefix.upper()}@{global_step}", dynamic_ncols=True)
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

        labels_list = decollate_batch(labels)
        labels_convert = [post_label(x) for x in labels_list]

        outputs_list = decollate_batch(outputs)
        outputs_convert = [post_pred(x) for x in outputs_list]

        for pred_1hot, true_1hot in zip(outputs_convert, labels_convert):
            dice_c = dice_per_class_onehot(pred_1hot, true_1hot, ignore_empty=True).detach().cpu().double()
            dice_mask = ~torch.isnan(dice_c)
            sum_dice += torch.nan_to_num(dice_c, nan=0.0)
            cnt_dice += dice_mask.double()

            if compute_iou:
                iou_c = iou_per_class_onehot(pred_1hot, true_1hot, ignore_empty=True).detach().cpu().double()
                iou_mask = ~torch.isnan(iou_c)
                sum_iou += torch.nan_to_num(iou_c, nan=0.0)
                cnt_iou += iou_mask.double()

            if compute_hd95:
                hd95_c = hd95_per_class_onehot(
                    pred_1hot,
                    true_1hot,
                    spacing=spacing,
                    include_background=False,
                    ignore_empty=True,
                ).detach().cpu().double()
                hd95_mask = ~torch.isnan(hd95_c)
                sum_hd95 += torch.nan_to_num(hd95_c, nan=0.0)
                cnt_hd95 += hd95_mask.double()

    dice_per_class = torch.full((num_classes,), float("nan"), dtype=torch.float64)
    valid_dice = cnt_dice > 0
    dice_per_class[valid_dice] = sum_dice[valid_dice] / cnt_dice[valid_dice]
    dice_per_class = dice_per_class.float()

    iou_per_class = torch.full((num_classes,), float("nan"), dtype=torch.float64)
    valid_iou = cnt_iou > 0
    iou_per_class[valid_iou] = sum_iou[valid_iou] / cnt_iou[valid_iou]
    iou_per_class = iou_per_class.float()

    if compute_hd95:
        hd95_per_class = torch.full((max(num_classes - 1, 0),), float("nan"), dtype=torch.float64)
        valid_hd95 = cnt_hd95 > 0
        hd95_per_class[valid_hd95] = sum_hd95[valid_hd95] / cnt_hd95[valid_hd95]
        hd95_per_class = hd95_per_class.float()
    else:
        hd95_per_class = torch.full((max(num_classes - 1, 0),), float("nan"), dtype=torch.float32)

    # 只保留前景类别（1~13），不保存背景类
    dice_per_class_fg = dice_per_class[1:] if num_classes > 1 else torch.empty(0, dtype=torch.float32)
    iou_per_class_fg = iou_per_class[1:] if num_classes > 1 else torch.empty(0, dtype=torch.float32)
    hd95_per_class_fg = hd95_per_class  # 已经不含背景

    dice_mean_fg = float(torch.nanmean(dice_per_class_fg).item()) if dice_per_class_fg.numel() > 0 else float("nan")
    iou_mean_fg = float(torch.nanmean(iou_per_class_fg).item()) if iou_per_class_fg.numel() > 0 else float("nan")
    hd95_mean_fg = float(torch.nanmean(hd95_per_class_fg).item()) if hd95_per_class_fg.numel() > 0 else float("nan")

    if writer is not None:
        writer.add_scalar(f"{prefix}/dice_mean_fg", dice_mean_fg, global_step)
        writer.add_scalar(f"{prefix}/iou_mean_fg", iou_mean_fg, global_step)

        if compute_hd95:
            writer.add_scalar(f"{prefix}/hd95_mean_fg", hd95_mean_fg, global_step)

        fg_class_names = get_foreground_class_names(class_names)
        for local_idx, class_name in enumerate(fg_class_names):
            writer.add_scalar(
                f"{prefix}/per_class_dice/{class_name}",
                float(dice_per_class_fg[local_idx].item()),
                global_step,
            )
            writer.add_scalar(
                f"{prefix}/per_class_iou/{class_name}",
                float(iou_per_class_fg[local_idx].item()),
                global_step,
            )

        if compute_hd95:
            for local_idx, class_name in enumerate(fg_class_names):
                writer.add_scalar(
                    f"{prefix}/per_class_hd95/{class_name}",
                    float(hd95_per_class_fg[local_idx].item()),
                    global_step,
                )

    return {
        "dice_mean_fg": dice_mean_fg,
        "iou_mean_fg": iou_mean_fg,
        "hd95_mean_fg": hd95_mean_fg,
        "dice_per_class_fg": dice_per_class_fg,
        "iou_per_class_fg": iou_per_class_fg,
        "hd95_per_class_fg": hd95_per_class_fg,
    }


def _csv_cell(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            v = v.item()
        else:
            raise ValueError("CSV cell must be scalar.")
    if isinstance(v, (np.floating, float)):
        return "" if (math.isnan(float(v)) or math.isinf(float(v))) else float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if v is None:
        return ""
    return v


def write_csv_rows(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k, "")) for k in fieldnames})


def build_run_overall_row(
    run_idx: int,
    seed: int,
    best_step: int,
    best_val_dice_fg: float,
    test_res: Dict[str, Any],
    train_minutes: float,
    early_stopped: bool,
    early_stop_reason: str,
) -> Dict[str, Any]:
    return {
        "run_idx": int(run_idx),
        "seed": int(seed),
        "best_iter": int(best_step),
        "best_val_dice_fg": safe_float(best_val_dice_fg),
        "test_dice_mean_fg": safe_float(test_res["dice_mean_fg"]),
        "test_iou_mean_fg": safe_float(test_res["iou_mean_fg"]),
        "test_hd95_mean_fg_mm": safe_float(test_res["hd95_mean_fg"]),
        "train_minutes": safe_float(train_minutes),
        "early_stopped": bool(early_stopped),
        "early_stop_reason": early_stop_reason,
    }


def build_run_per_class_rows(
    run_idx: int,
    seed: int,
    class_names: List[str],
    test_res: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    fg_class_names = get_foreground_class_names(class_names)

    dice_pc = test_res["dice_per_class_fg"]
    iou_pc = test_res["iou_per_class_fg"]
    hd95_pc = test_res["hd95_per_class_fg"]

    for local_idx, class_name in enumerate(fg_class_names):
        class_index = local_idx + 1  # 对应原始标签 1~13
        row = {
            "run_idx": int(run_idx),
            "seed": int(seed),
            "class_index": int(class_index),
            "class_name": class_name,
            "dice": safe_float(dice_pc[local_idx].item()) if local_idx < len(dice_pc) else None,
            "iou": safe_float(iou_pc[local_idx].item()) if local_idx < len(iou_pc) else None,
            "hd95_mm": safe_float(hd95_pc[local_idx].item()) if local_idx < len(hd95_pc) else None,
        }
        rows.append(row)
    return rows


def build_three_run_overall_mean(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [
        "best_val_dice_fg",
        "test_dice_mean_fg",
        "test_iou_mean_fg",
        "test_hd95_mean_fg_mm",
        "train_minutes",
    ]

    out: Dict[str, Any] = {"num_runs": int(len(rows))}
    for m in metrics:
        vals = [float(r[m]) for r in rows if r.get(m) not in ("", None)]
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{m}_mean"] = float(np.nanmean(arr)) if arr.size > 0 else None
        out[f"{m}_std"] = float(np.nanstd(arr)) if arr.size > 0 else None
    return [out]


def build_three_run_per_class_mean(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    class_indices = sorted({int(r["class_index"]) for r in rows})

    for c in class_indices:
        class_rows = [r for r in rows if int(r["class_index"]) == c]
        class_name = class_rows[0]["class_name"] if len(class_rows) > 0 else f"class_{c}"

        dice_vals = np.asarray([float(r["dice"]) for r in class_rows if r.get("dice") not in ("", None)], dtype=np.float64)
        iou_vals = np.asarray([float(r["iou"]) for r in class_rows if r.get("iou") not in ("", None)], dtype=np.float64)
        hd95_vals = np.asarray([float(r["hd95_mm"]) for r in class_rows if r.get("hd95_mm") not in ("", None)], dtype=np.float64)

        out_rows.append(
            {
                "class_index": int(c),
                "class_name": class_name,
                "dice_mean": float(np.nanmean(dice_vals)) if dice_vals.size > 0 else None,
                "dice_std": float(np.nanstd(dice_vals)) if dice_vals.size > 0 else None,
                "iou_mean": float(np.nanmean(iou_vals)) if iou_vals.size > 0 else None,
                "iou_std": float(np.nanstd(iou_vals)) if iou_vals.size > 0 else None,
                "hd95_mm_mean": float(np.nanmean(hd95_vals)) if hd95_vals.size > 0 else None,
                "hd95_mm_std": float(np.nanstd(hd95_vals)) if hd95_vals.size > 0 else None,
            }
        )
    return out_rows


def print_run_test_metrics(run_idx: int, seed: int, class_names: List[str], best_step: int, best_val_dice_fg: float, test_res: Dict[str, Any]) -> None:
    print("\n================ TEST (best.pt single model) ================")
    print(f"run_idx              : {run_idx}")
    print(f"seed                 : {seed}")
    print(f"best_iter            : {best_step}")
    print(f"best_val_dice_fg     : {best_val_dice_fg:.6f}")
    print(f"TEST dice_mean_fg    : {test_res['dice_mean_fg']:.6f}")
    print(f"TEST iou_mean_fg     : {test_res['iou_mean_fg']:.6f}")
    print(f"TEST hd95_mean_fg(mm): {test_res['hd95_mean_fg']:.6f}")
    print("=============================================================\n")

    print("Per-class metrics (labels 1~13 only):")
    fg_class_names = get_foreground_class_names(class_names)
    dice_pc = test_res["dice_per_class_fg"]
    iou_pc = test_res["iou_per_class_fg"]
    hd95_pc = test_res["hd95_per_class_fg"]

    for local_idx, class_name in enumerate(fg_class_names):
        class_index = local_idx + 1
        dice_v = safe_float(dice_pc[local_idx].item()) if local_idx < len(dice_pc) else None
        iou_v = safe_float(iou_pc[local_idx].item()) if local_idx < len(iou_pc) else None
        hd95_v = safe_float(hd95_pc[local_idx].item()) if local_idx < len(hd95_pc) else None

        dice_str = f"{dice_v:.6f}" if dice_v is not None else "NA"
        iou_str = f"{iou_v:.6f}" if iou_v is not None else "NA"
        hd95_str = f"{hd95_v:.6f}" if hd95_v is not None else "NA"
        print(f"  [{class_index:02d}] {class_name:>10s} | Dice={dice_str} | IoU={iou_str} | HD95(mm)={hd95_str}")
    print("")


def build_optimizer(args, model: torch.nn.Module) -> torch.optim.Optimizer:
    def build_param_groups(m: torch.nn.Module, weight_decay: float):
        decay, no_decay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() == 1 or n.endswith(".bias") or ("norm" in n.lower()) or ("bn" in n.lower()):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    param_groups = build_param_groups(model, weight_decay=float(args.wd))
    opt_name = str(args.optimizer).lower()

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=float(args.lr),
            momentum=float(args.momentum),
            nesterov=bool(args.sgd_nesterov),
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=float(args.lr),
            betas=(float(args.adamw_betas[0]), float(args.adamw_betas[1])),
            eps=float(args.adamw_eps),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    return optimizer


def train_and_evaluate_one_run(
    args,
    root_output_dir: str,
    run_idx: int,
    run_seed: int,
    device: torch.device,
    json_path: str,
    pixdim: Tuple[float, float, float],
    roi_size: Tuple[int, int, int],
    num_classes: int,
    class_names: List[str],
    train_files: List[Dict[str, Any]],
    val_files: List[Dict[str, Any]],
    test_files: List[Dict[str, Any]],
    val_source: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    output_dir = os.path.join(root_output_dir, f"run_{run_idx:02d}_seed_{run_seed}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[RUN {run_idx}] seed={run_seed} | output_dir={output_dir}")

    seed_everything(run_seed)
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("[WARN] cudnn_benchmark=True may reduce reproducibility even with fixed seeds.")
    else:
        torch.backends.cudnn.benchmark = False

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
    writer.add_text("meta/output_dir", output_dir, 0)
    writer.add_text("meta/json_path", json_path, 0)
    writer.add_text("meta/sw_infer_mode", "gaussian", 0)
    writer.add_text("meta/sw_overlap", str(args.sw_overlap), 0)
    writer.add_text("meta/val_source", val_source, 0)
    writer.add_text("meta/val_start_iter", str(args.val_start_iter), 0)
    writer.add_text("meta/run_seed", str(run_seed), 0)
    writer.add_text("meta/split_seed", str(args.split_seed), 0)
    writer.add_text("meta/optimizer", str(args.optimizer), 0)

    run_cfg = vars(args).copy()
    run_cfg.update(
        {
            "run_seed": int(run_seed),
            "output_dir": output_dir,
            "json_path": json_path,
            "pixdim": pixdim,
            "roi_size": roi_size,
            "device": str(device),
            "class_names": class_names,
            "class_labels": CLASS_LABELS,
            "best_selection_metric": "val/dice_mean_fg (labels 1~13 only)",
            "test_report_metrics": [
                "test/dice_mean_fg",
                "test/iou_mean_fg",
                "test/hd95_mean_fg_mm",
            ],
            "sw_infer_mode": "gaussian",
            "exclude_cases": list(args.exclude_cases),
            "val_source": val_source,
            "val_start_iter": int(args.val_start_iter),
            "effective_split_sizes": {
                "train": int(len(train_files)),
                "val": int(len(val_files)),
                "test": int(len(test_files)),
            },
        }
    )
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(
                keys=["image", "label"],
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
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
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(
                keys=["image", "label"],
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        ]
    )

    with open(os.path.join(output_dir, "train_files_list.json"), "w", encoding="utf-8") as f:
        json.dump(train_files, f, indent=2)
    with open(os.path.join(output_dir, "val_files_list.json"), "w", encoding="utf-8") as f:
        json.dump(val_files, f, indent=2)
    with open(os.path.join(output_dir, "test_files_list.json"), "w", encoding="utf-8") as f:
        json.dump(test_files, f, indent=2)

    g = torch.Generator()
    g.manual_seed(run_seed)

    cache_num_train = len(train_files) if args.cache_num_train <= 0 else min(args.cache_num_train, len(train_files))
    cache_num_val = len(val_files) if args.cache_num_val <= 0 else min(args.cache_num_val, len(val_files))
    cache_num_test = len(test_files) if args.cache_num_test <= 0 else min(args.cache_num_test, len(test_files))

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=cache_num_train,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers_train,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers_train,
        pin_memory=True,
        collate_fn=list_data_collate,
        persistent_workers=(args.num_workers_train > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_ds = None
    val_loader = None
    test_ds = None
    test_loader = None

    def ensure_val_loader() -> DataLoader:
        nonlocal val_ds, val_loader
        if val_loader is not None:
            return val_loader

        val_ds = CacheDataset(
            data=val_files,
            transform=eval_transforms,
            cache_num=cache_num_val,
            cache_rate=args.cache_rate,
            num_workers=args.num_workers_val,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers_val,
            pin_memory=True,
            persistent_workers=(args.num_workers_val > 0),
            worker_init_fn=seed_worker,
            generator=g,
        )
        print(f"[VAL] loader built (lazy) | size={len(val_files)} | cache_num={cache_num_val}")
        return val_loader

    def ensure_test_loader() -> DataLoader:
        nonlocal test_ds, test_loader, val_ds, val_loader
        if test_loader is not None:
            return test_loader

        if (
            (val_source == "test_as_val")
            and (val_loader is not None)
            and (args.num_workers_test == args.num_workers_val)
            and (cache_num_test == cache_num_val)
        ):
            test_loader = val_loader
            test_ds = val_ds
            print("[TEST] reuse val_loader as test_loader (val=test)")
            return test_loader

        test_ds = CacheDataset(
            data=test_files,
            transform=eval_transforms,
            cache_num=cache_num_test,
            cache_rate=args.cache_rate,
            num_workers=args.num_workers_test,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers_test,
            pin_memory=True,
            persistent_workers=(args.num_workers_test > 0),
            worker_init_fn=seed_worker,
            generator=g,
        )
        print(f"[TEST] loader built (lazy) | size={len(test_files)} | cache_num={cache_num_test}")
        return test_loader

    model = WFT_UNet(n_channels=1,n_classes=14).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    optimizer = build_optimizer(args, model)

    scheduler = None
    if args.use_poly:
        base_lr = float(args.lr)

        def lr_lambda(step: int):
            t = min(max(step, 0), int(args.max_iterations))
            poly = (1.0 - t / float(args.max_iterations)) ** float(args.poly_power)
            return max(float(args.min_lr) / base_lr, poly)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    global_step = 0
    best_val_dice_fg = -1.0
    best_step = -1
    best_path = os.path.join(output_dir, "best.pt")

    num_evals = 0
    bad_evals = 0
    stop_training = False
    early_stop_reason = ""

    pbar = tqdm(total=args.max_iterations, desc=f"Training(seed={run_seed})", dynamic_ncols=True)
    t0 = time.time()

    while (global_step < args.max_iterations) and (not stop_training):
        model.train()
        for batch in train_loader:
            if global_step >= args.max_iterations or stop_training:
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

            if scheduler is not None:
                scheduler.step()

            global_step += 1
            pbar.update(1)

            lr_now = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/loss_total", float(loss.item()), global_step)
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/lr", float(lr_now), global_step)
            pbar.set_description(f"Training seed={run_seed} iter={global_step}/{args.max_iterations} loss={loss.item():.5f}")

            do_eval_now = ((global_step % args.eval_num == 0) or (global_step == args.max_iterations))
            if do_eval_now and (global_step >= int(args.val_start_iter)):
                prev_best = float(best_val_dice_fg)

                vloader = ensure_val_loader()
                val_res = run_evaluation_single_model(
                    model=model,
                    loader=vloader,
                    device=device,
                    roi_size=roi_size,
                    sw_batch_size=args.sw_batch_size,
                    sw_overlap=args.sw_overlap,
                    num_classes=num_classes,
                    class_names=class_names,
                    post_label=post_label,
                    post_pred=post_pred,
                    global_step=global_step,
                    spacing=pixdim,
                    writer=writer,
                    prefix="val",
                    compute_hd95=False,
                    compute_iou=True,
                )
                model.train()
                dice_fg = float(val_res["dice_mean_fg"])
                iou_fg = float(val_res["iou_mean_fg"])

                if dice_fg > best_val_dice_fg:
                    best_val_dice_fg = float(dice_fg)
                    best_step = int(global_step)
                    torch.save(model.state_dict(), best_path)
                    print(
                        f"[SAVE] best.pt @ iter={best_step} | "
                        f"best_val_dice_fg={best_val_dice_fg:.4f} | "
                        f"val_iou_fg={iou_fg:.4f}"
                    )
                else:
                    print(
                        f"[NOSAVE] best_val_dice_fg={best_val_dice_fg:.4f} | "
                        f"current_val_dice_fg={dice_fg:.4f} | "
                        f"val_iou_fg={iou_fg:.4f}"
                    )

                num_evals += 1
                improved_for_early_stop = (dice_fg > (prev_best + args.early_stop_min_delta))
                if improved_for_early_stop:
                    bad_evals = 0
                else:
                    if num_evals > args.early_stop_warmup:
                        bad_evals += 1

                writer.add_scalar("train/early_stop_bad_evals", bad_evals, global_step)
                writer.add_scalar("train/best_val_dice_mean_fg", float(best_val_dice_fg), global_step)

                if (num_evals > args.early_stop_warmup) and (bad_evals >= args.early_stop_patience):
                    stop_training = True
                    early_stop_reason = (
                        f"no improvement in val foreground Dice for {bad_evals} validations "
                        f"(patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta})"
                    )
                    print(f"[EARLY STOP] iter={global_step} | {early_stop_reason}")
                    writer.add_text("meta/early_stop_reason", early_stop_reason, global_step)
                    break

    pbar.close()
    dt = time.time() - t0
    train_minutes = dt / 60.0
    print(f"[RUN {run_idx}] Training finished. Time: {train_minutes:.1f} min")
    writer.add_scalar("meta/train_minutes", train_minutes, global_step)

    if stop_training:
        print(f"[RUN {run_idx}] Stopped early at iter={global_step}. Reason: {early_stop_reason}")

    if not os.path.isfile(best_path):
        torch.save(model.state_dict(), best_path)
        best_step = global_step
        print("[WARN] No validation checkpoint was saved. Falling back to final weights as best.pt.")

    best_sd = torch.load(best_path, map_location=device)
    model.load_state_dict(best_sd, strict=True)
    model.eval()

    tloader = ensure_test_loader()
    test_res = run_evaluation_single_model(
        model=model,
        loader=tloader,
        device=device,
        roi_size=roi_size,
        sw_batch_size=args.sw_batch_size,
        sw_overlap=args.sw_overlap,
        num_classes=num_classes,
        class_names=class_names,
        post_label=post_label,
        post_pred=post_pred,
        global_step=best_step if best_step > 0 else global_step,
        spacing=pixdim,
        writer=writer,
        prefix="test",
        compute_hd95=True,
        compute_iou=True,
    )

    print_run_test_metrics(
        run_idx=run_idx,
        seed=run_seed,
        class_names=class_names,
        best_step=best_step,
        best_val_dice_fg=best_val_dice_fg,
        test_res=test_res,
    )

    overall_row = build_run_overall_row(
        run_idx=run_idx,
        seed=run_seed,
        best_step=best_step,
        best_val_dice_fg=best_val_dice_fg,
        test_res=test_res,
        train_minutes=train_minutes,
        early_stopped=stop_training,
        early_stop_reason=early_stop_reason,
    )
    per_class_rows = build_run_per_class_rows(
        run_idx=run_idx,
        seed=run_seed,
        class_names=class_names,
        test_res=test_res,
    )

    run_overall_fields = [
        "run_idx",
        "seed",
        "best_iter",
        "best_val_dice_fg",
        "test_dice_mean_fg",
        "test_iou_mean_fg",
        "test_hd95_mean_fg_mm",
        "train_minutes",
        "early_stopped",
        "early_stop_reason",
    ]
    run_per_class_fields = [
        "run_idx",
        "seed",
        "class_index",
        "class_name",
        "dice",
        "iou",
        "hd95_mm",
    ]

    write_csv_rows(os.path.join(output_dir, "test_overall_metrics.csv"), [overall_row], run_overall_fields)
    write_csv_rows(os.path.join(output_dir, "test_per_class_metrics.csv"), per_class_rows, run_per_class_fields)

    writer.close()

    print("\nSaved artifacts:")
    print(" -", os.path.join(output_dir, "config.json"))
    print(" -", os.path.join(output_dir, "tb"))
    print(" -", best_path)
    print(" -", os.path.join(output_dir, "test_overall_metrics.csv"))
    print(" -", os.path.join(output_dir, "test_per_class_metrics.csv"))
    print(" -", os.path.join(output_dir, "train_files_list.json"))
    print(" -", os.path.join(output_dir, "val_files_list.json"))
    print(" -", os.path.join(output_dir, "test_files_list.json"))

    return overall_row, per_class_rows


def main():
    args = parse_args()

    fixed_seed = 123
    args.seed = fixed_seed
    all_run_seeds = [fixed_seed]

    run_id = args.run_name.strip() if args.run_name.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    root_output_dir = os.path.join(args.output_root, f"btcv_run_{run_id}")
    os.makedirs(root_output_dir, exist_ok=True)
    print("root_output_dir =", root_output_dir)

    save_code_snapshot(output_dir=root_output_dir, extra_rel_paths=list(args.snapshot_extra))

    print_config()
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print("MONAI root_dir =", root_dir)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    json_path = os.path.join(args.data_dir, args.split_json)

    pixdim = tuple(float(x) for x in args.pixdim)
    roi_size = tuple(int(x) for x in args.roi_size)
    num_classes = int(args.num_classes)
    class_names = get_class_names(num_classes)

    if args.val_separate:
        train_files, val_files, test_files = build_custom_splits_from_json(
            json_path=json_path,
            pool_keys=args.pool_keys,
            train_num=args.train_num,
            val_num=args.val_num,
            test_num=args.test_num,
            split_seed=args.split_seed,
            exclude_cases=args.exclude_cases,
        )
        val_source = "separate_val"
    else:
        train_files, _empty_val, test_files = build_custom_splits_from_json(
            json_path=json_path,
            pool_keys=args.pool_keys,
            train_num=args.train_num,
            val_num=0,
            test_num=args.test_num,
            split_seed=args.split_seed,
            exclude_cases=args.exclude_cases,
        )
        val_files = list(test_files)
        val_source = "test_as_val"

    print("\n================ SPLIT ================")
    print(f"[POOL] keys={args.pool_keys} | split_seed={args.split_seed}")
    print(f"[EXCL] exclude_cases={args.exclude_cases if args.exclude_cases else '[] (no exclusion)'}")
    print(f"[VAL ] val_source={val_source} | val_start_iter={args.val_start_iter}")
    print(f"[SPLIT] train={len(train_files)} | val={len(val_files)} | test={len(test_files)}")
    print(f"[SEED] train/test seeds={all_run_seeds} | split_seed(fixed)={args.split_seed}")
    print(f"[OPT ] optimizer={args.optimizer}")
    print("=======================================\n")

    root_cfg = vars(args).copy()
    root_cfg.update(
        {
            "root_output_dir": root_output_dir,
            "json_path": json_path,
            "pixdim": pixdim,
            "roi_size": roi_size,
            "device": str(device),
            "class_names": class_names,
            "class_labels": CLASS_LABELS,
            "run_seed": fixed_seed,
            "val_source": val_source,
            "effective_split_sizes": {
                "train": int(len(train_files)),
                "val": int(len(val_files)),
                "test": int(len(test_files)),
            },
        }
    )
    with open(os.path.join(root_output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(root_cfg, f, indent=2)

    with open(os.path.join(root_output_dir, "train_files_list.json"), "w", encoding="utf-8") as f:
        json.dump(train_files, f, indent=2)
    with open(os.path.join(root_output_dir, "val_files_list.json"), "w", encoding="utf-8") as f:
        json.dump(val_files, f, indent=2)
    with open(os.path.join(root_output_dir, "test_files_list.json"), "w", encoding="utf-8") as f:
        json.dump(test_files, f, indent=2)

    overall_row, per_class_rows = train_and_evaluate_one_run(
        args=args,
        root_output_dir=root_output_dir,
        run_idx=1,
        run_seed=fixed_seed,
        device=device,
        json_path=json_path,
        pixdim=pixdim,
        roi_size=roi_size,
        num_classes=num_classes,
        class_names=class_names,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        val_source=val_source,
    )

    overall_fields = [
        "run_idx",
        "seed",
        "best_iter",
        "best_val_dice_fg",
        "test_dice_mean_fg",
        "test_iou_mean_fg",
        "test_hd95_mean_fg_mm",
        "train_minutes",
        "early_stopped",
        "early_stop_reason",
    ]
    per_class_fields = [
        "run_idx",
        "seed",
        "class_index",
        "class_name",
        "dice",
        "iou",
        "hd95_mm",
    ]

    write_csv_rows(os.path.join(root_output_dir, "single_run_overall_metrics.csv"), [overall_row], overall_fields)
    write_csv_rows(os.path.join(root_output_dir, "single_run_per_class_metrics.csv"), per_class_rows, per_class_fields)

    print("\n================ SINGLE-RUN SUMMARY ================")
    print(f"run_idx                 : {overall_row['run_idx']}")
    print(f"seed                    : {overall_row['seed']}")
    print(f"best_val_dice_fg        : {overall_row['best_val_dice_fg']:.6f}")
    print(f"test_dice_mean_fg       : {overall_row['test_dice_mean_fg']:.6f}")
    print(f"test_iou_mean_fg        : {overall_row['test_iou_mean_fg']:.6f}")
    print(f"test_hd95_mean_fg(mm)   : {overall_row['test_hd95_mean_fg_mm']:.6f}")
    print("====================================================\n")

    print("Saved root artifacts:")
    print(" -", os.path.join(root_output_dir, "config.json"))
    print(" -", os.path.join(root_output_dir, "snapshot"))
    print(" -", os.path.join(root_output_dir, "train_files_list.json"))
    print(" -", os.path.join(root_output_dir, "val_files_list.json"))
    print(" -", os.path.join(root_output_dir, "test_files_list.json"))
    print(" -", os.path.join(root_output_dir, "single_run_overall_metrics.csv"))
    print(" -", os.path.join(root_output_dir, "single_run_per_class_metrics.csv"))


if __name__ == "__main__":
    main()