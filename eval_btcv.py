import os
import json
import csv
import glob
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
from tqdm import tqdm

from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

from networks.UNet3D import UNet3D


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
    parser = argparse.ArgumentParser("Evaluate IoU / Dice / HD95 for seeds 123/223/323 best.pt")
    parser.add_argument("--output_root", type=str, default="./outputs_UNet3D")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[123, 223, 323],
        help="random seeds to evaluate",
    )
    parser.add_argument(
        "--run_dirs",
        type=str,
        nargs="*",
        default=None,
        help="optional explicit run directories; if provided, auto-search is skipped",
    )
    parser.add_argument(
        "--same_test_list",
        type=str,
        default="",
        help="optional path to a fixed shared test_files_list.json; "
             "if empty, each run uses its own test_files_list.json",
    )
    parser.add_argument("--checkpoint_name", type=str, default="best.pt")

    parser.add_argument("--iou_csv_path", type=str, default="./iou_summary_123_223_323.csv")
    parser.add_argument("--dice_csv_path", type=str, default="./dice_summary_123_223_323.csv")
    parser.add_argument("--hd95_csv_path", type=str, default="./hd95_summary_123_223_323.csv")

    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--sw_overlap", type=float, default=0.5)
    parser.add_argument("--cache_num_test", type=int, default=0, help="<=0 means cache all")
    parser.add_argument("--cache_rate", type=float, default=1.0)
    parser.add_argument("--num_workers_test", type=int, default=6)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def get_class_names(num_classes: int) -> List[str]:
    names = []
    for i in range(num_classes):
        names.append(CLASS_LABELS.get(str(i), f"class_{i}"))
    return names


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def nanmean(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def build_eval_transforms(pixdim: Tuple[float, float, float]):
    return Compose(
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


def build_model(num_classes: int) -> torch.nn.Module:
    #model = UNet3D(in_channels=1,out_channels=14,f_maps=[24, 48, 96, 192]).to(device)
    return UNet3D(in_channels=1, out_channels=num_classes,f_maps=[24, 48, 96, 192])


def find_run_dir_by_seed(output_root: str, seed: int) -> str:
    pattern = os.path.join(output_root, f"btcv_run_*seed{seed}")
    candidates = glob.glob(pattern)
    if len(candidates) == 0:
        raise FileNotFoundError(f"No run directory found for seed={seed}, pattern={pattern}")

    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = candidates[0]
    if len(candidates) > 1:
        print(f"[WARN] multiple run dirs found for seed={seed}, choose latest:\n  {chosen}")
    return chosen


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_weights(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # 兼容 DDP/DP 的 module. 前缀
        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v
        model.load_state_dict(new_state, strict=True)


def to_case_chwd(x: torch.Tensor, n_cls: int) -> torch.Tensor:
    """
    Normalize one case tensor to [C, H, W, D].
    Accept:
      - [C, H, W, D]
      - [1, C, H, W, D]
      - [C, 1, H, W, D]
    """
    if x.ndim == 4:
        return x
    if x.ndim == 5:
        if x.shape[0] == 1 and x.shape[1] == n_cls:
            return x.squeeze(0)
        if x.shape[0] == n_cls and x.shape[1] == 1:
            return x.squeeze(1)
    raise ValueError(f"Expected one-case tensor in CHWD-like shape, got {tuple(x.shape)}")


def iou_per_class_onehot(
    y_pred_1hot: torch.Tensor,
    y_true_1hot: torch.Tensor,
    eps: float = 1e-8,
    ignore_empty: bool = True,
) -> torch.Tensor:
    """
    Returns IoU for each class, shape [C].
    ignore_empty=True:
      - GT empty class => NaN, excluded from averaging
    """
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


def dice_per_class_onehot(
    y_pred_1hot: torch.Tensor,
    y_true_1hot: torch.Tensor,
    eps: float = 1e-8,
    ignore_empty: bool = True,
) -> torch.Tensor:
    """
    Returns Dice for each class, shape [C].
    ignore_empty=True:
      - GT empty class => NaN, excluded from averaging
    """
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


@torch.no_grad()
def evaluate_one_run(
    run_dir: str,
    checkpoint_name: str,
    same_test_list: str,
    sw_batch_size: int,
    sw_overlap: float,
    cache_num_test: int,
    cache_rate: float,
    num_workers_test: int,
    device: torch.device,
) -> Dict[str, Any]:
    config_path = os.path.join(run_dir, "config.json")
    ckpt_path = os.path.join(run_dir, checkpoint_name)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    cfg = load_json(config_path)

    num_classes = int(cfg["num_classes"])
    pixdim = tuple(cfg["pixdim"])
    roi_size = tuple(cfg["roi_size"])
    class_names = cfg.get("class_names", get_class_names(num_classes))
    seed = cfg.get("seed", None)

    if same_test_list:
        test_list_path = same_test_list
    else:
        test_list_path = os.path.join(run_dir, "test_files_list.json")

    if not os.path.isfile(test_list_path):
        raise FileNotFoundError(f"Missing test file list: {test_list_path}")

    test_files = load_json(test_list_path)
    if not isinstance(test_files, list) or len(test_files) == 0:
        raise ValueError(f"test_files_list.json is empty or invalid: {test_list_path}")

    print("\n====================================================")
    print(f"[RUN] run_dir   : {run_dir}")
    print(f"[RUN] seed      : {seed}")
    print(f"[RUN] checkpoint: {ckpt_path}")
    print(f"[RUN] test_list : {test_list_path}")
    print(f"[RUN] test_size : {len(test_files)}")
    print("====================================================")

    test_transforms = build_eval_transforms(pixdim=pixdim)
    cache_num = len(test_files) if cache_num_test <= 0 else min(cache_num_test, len(test_files))

    test_ds = CacheDataset(
        data=test_files,
        transform=test_transforms,
        cache_num=cache_num,
        cache_rate=cache_rate,
        num_workers=num_workers_test,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers_test,
        pin_memory=True,
        persistent_workers=(num_workers_test > 0),
    )

    model = build_model(num_classes=num_classes).to(device)
    load_checkpoint_weights(model, ckpt_path, device)
    model.eval()

    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

    # mean-over-cases per class
    sum_iou = torch.zeros((num_classes,), dtype=torch.float64)
    cnt_iou = torch.zeros((num_classes,), dtype=torch.float64)

    sum_dice = torch.zeros((num_classes,), dtype=torch.float64)
    cnt_dice = torch.zeros((num_classes,), dtype=torch.float64)

    fg_classes = max(num_classes - 1, 0)
    sum_hd95 = torch.zeros((fg_classes,), dtype=torch.float64)
    cnt_hd95 = torch.zeros((fg_classes,), dtype=torch.float64)

    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        reduction="none",
        percentile=95,
        get_not_nans=False,
    )

    pbar = tqdm(test_loader, desc=f"Eval seed={seed}", dynamic_ncols=True)
    for batch in pbar:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = sliding_window_inference(
            inputs,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=sw_overlap,
            mode="gaussian",
        )

        labels_list = decollate_batch(labels)
        preds_list = decollate_batch(logits)

        true_case = to_case_chwd(post_label(labels_list[0]), num_classes)
        pred_case = to_case_chwd(post_pred(preds_list[0]), num_classes)

        if pred_case.shape != true_case.shape:
            raise ValueError(
                f"Prediction/label shape mismatch: pred={tuple(pred_case.shape)} vs true={tuple(true_case.shape)}"
            )

        # IoU
        iou_c = iou_per_class_onehot(pred_case, true_case, ignore_empty=True).detach().cpu().double()
        valid_iou = torch.isfinite(iou_c)
        sum_iou += torch.where(valid_iou, iou_c, torch.zeros_like(iou_c))
        cnt_iou += valid_iou.double()

        # Dice
        dice_c = dice_per_class_onehot(pred_case, true_case, ignore_empty=True).detach().cpu().double()
        valid_dice = torch.isfinite(dice_c)
        sum_dice += torch.where(valid_dice, dice_c, torch.zeros_like(dice_c))
        cnt_dice += valid_dice.double()

        # HD95 (foreground only)
        if fg_classes > 0:
            hd95_metric(y_pred=[pred_case], y=[true_case], spacing=pixdim)
            hd95_case = hd95_metric.aggregate()
            hd95_metric.reset()

            if isinstance(hd95_case, (tuple, list)):
                hd95_case = hd95_case[0]
            hd95_case = hd95_case.squeeze(0).detach().cpu().double()

            if hd95_case.numel() != fg_classes:
                raise ValueError(
                    f"HD95 result numel mismatch: expected {fg_classes}, got {hd95_case.numel()}"
                )

            valid_hd95 = torch.isfinite(hd95_case)
            sum_hd95 += torch.where(valid_hd95, hd95_case, torch.zeros_like(hd95_case))
            cnt_hd95 += valid_hd95.double()

        fg_iou = torch.nanmean(iou_c[1:]).item() if num_classes > 1 else torch.nanmean(iou_c).item()
        fg_dice = torch.nanmean(dice_c[1:]).item() if num_classes > 1 else torch.nanmean(dice_c).item()
        pbar.set_postfix({"fg_mIoU": f"{fg_iou:.4f}", "fg_Dice": f"{fg_dice:.4f}"})

    iou_per_class_mean = torch.full((num_classes,), float("nan"), dtype=torch.float64)
    valid = cnt_iou > 0
    iou_per_class_mean[valid] = sum_iou[valid] / cnt_iou[valid]

    dice_per_class_mean = torch.full((num_classes,), float("nan"), dtype=torch.float64)
    valid = cnt_dice > 0
    dice_per_class_mean[valid] = sum_dice[valid] / cnt_dice[valid]

    hd95_per_class_mean = torch.full((fg_classes,), float("nan"), dtype=torch.float64)
    if fg_classes > 0:
        valid = cnt_hd95 > 0
        hd95_per_class_mean[valid] = sum_hd95[valid] / cnt_hd95[valid]

    mean_iou_fg = (
        float(torch.nanmean(iou_per_class_mean[1:]).item())
        if num_classes > 1
        else float(torch.nanmean(iou_per_class_mean).item())
    )
    mean_dice_fg = (
        float(torch.nanmean(dice_per_class_mean[1:]).item())
        if num_classes > 1
        else float(torch.nanmean(dice_per_class_mean).item())
    )
    mean_hd95_excl_bg = (
        float(torch.nanmean(hd95_per_class_mean).item())
        if hd95_per_class_mean.numel() > 0
        else float("nan")
    )

    print(f"[RESULT] seed={seed} | mean_iou_fg={mean_iou_fg:.6f} | mean_dice_fg={mean_dice_fg:.6f} | mean_hd95_excl_bg={mean_hd95_excl_bg:.6f}")

    print("Per-organ IoU:")
    for c in range(1, num_classes):
        v = iou_per_class_mean[c].item()
        print(f"  [{c:02d}] {class_names[c]:>10s} : {v:.6f}")

    print("Per-organ Dice:")
    for c in range(1, num_classes):
        v = dice_per_class_mean[c].item()
        print(f"  [{c:02d}] {class_names[c]:>10s} : {v:.6f}")

    if fg_classes > 0:
        print("Per-organ HD95:")
        for c in range(1, num_classes):
            idx = c - 1
            v = hd95_per_class_mean[idx].item()
            print(f"  [{c:02d}] {class_names[c]:>10s} : {v:.6f}")

    return {
        "run_dir": run_dir,
        "seed": seed,
        "checkpoint": ckpt_path,
        "test_list_path": test_list_path,
        "num_classes": num_classes,
        "class_names": class_names,

        "mean_iou_fg": safe_float(mean_iou_fg),
        "iou_per_class_incl_bg": [
            safe_float(v.item()) if torch.isfinite(v) else None
            for v in iou_per_class_mean
        ],

        "mean_dice_fg": safe_float(mean_dice_fg),
        "dice_per_class_incl_bg": [
            safe_float(v.item()) if torch.isfinite(v) else None
            for v in dice_per_class_mean
        ],

        "mean_hd95_excl_bg": safe_float(mean_hd95_excl_bg),
        "hd95_per_class_excl_bg": [
            safe_float(v.item()) if torch.isfinite(v) else None
            for v in hd95_per_class_mean
        ],
    }


def save_metric_csv(
    csv_path: str,
    run_results: List[Dict[str, Any]],
    overall_key: str,
    per_class_key: str,
    overall_metric_name: str,
    per_class_is_fg_only: bool,
):
    if len(run_results) == 0:
        raise ValueError("run_results is empty")

    class_names = run_results[0]["class_names"]
    num_classes = run_results[0]["num_classes"]

    # 校验各 run 配置一致
    for r in run_results[1:]:
        if r["num_classes"] != num_classes:
            raise ValueError("num_classes mismatch across runs")
        if r["class_names"] != class_names:
            raise ValueError("class_names mismatch across runs")

    seed_cols = [f"seed_{r['seed']}" for r in run_results]
    header = ["metric"] + seed_cols + ["mean_over_runs"]

    rows = []

    # overall row
    vals = [r.get(overall_key) for r in run_results]
    rows.append([overall_metric_name] + vals + [nanmean(vals)])

    # per-organ rows (always exclude background)
    for c in range(1, num_classes):
        organ_name = class_names[c]
        vals = []
        for r in run_results:
            arr = r.get(per_class_key, [])
            if per_class_is_fg_only:
                idx = c - 1
            else:
                idx = c
            vals.append(arr[idx] if idx < len(arr) else None)
        rows.append([organ_name] + vals + [nanmean(vals)])

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[CSV] saved to: {csv_path}")


def main():
    args = parse_args()

    if args.device.strip():
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] device = {device}")

    if args.run_dirs is not None and len(args.run_dirs) > 0:
        run_dirs = args.run_dirs
    else:
        run_dirs = [find_run_dir_by_seed(args.output_root, s) for s in args.seeds]

    run_results = []
    for run_dir in run_dirs:
        res = evaluate_one_run(
            run_dir=run_dir,
            checkpoint_name=args.checkpoint_name,
            same_test_list=args.same_test_list,
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
            cache_num_test=args.cache_num_test,
            cache_rate=args.cache_rate,
            num_workers_test=args.num_workers_test,
            device=device,
        )
        run_results.append(res)

    # IoU CSV
    save_metric_csv(
        csv_path=args.iou_csv_path,
        run_results=run_results,
        overall_key="mean_iou_fg",
        per_class_key="iou_per_class_incl_bg",
        overall_metric_name="mean_iou_fg",
        per_class_is_fg_only=False,
    )

    # Dice CSV
    save_metric_csv(
        csv_path=args.dice_csv_path,
        run_results=run_results,
        overall_key="mean_dice_fg",
        per_class_key="dice_per_class_incl_bg",
        overall_metric_name="mean_dice_fg",
        per_class_is_fg_only=False,
    )

    # HD95 CSV
    save_metric_csv(
        csv_path=args.hd95_csv_path,
        run_results=run_results,
        overall_key="mean_hd95_excl_bg",
        per_class_key="hd95_per_class_excl_bg",
        overall_metric_name="mean_hd95_excl_bg",
        per_class_is_fg_only=True,
    )


if __name__ == "__main__":
    main()