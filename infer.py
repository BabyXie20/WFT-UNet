import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)
from networks.Att_UNet import AttU_Net3D


CLASS_LABELS = {
    0: "Background",
    1: "Spleen",
    2: "R. Kid",
    3: "L. Kid",
    4: "Gall.",
    5: "Eso.",
    6: "Liver",
    7: "Stom.",
    8: "Aorta",
    9: "IVC",
    10: "PSV",
    11: "Panc.",
    12: "RAG",
    13: "LAG",
}

CLASS_COLORS = [
    "#000000",  # 0 background
    "#1f77b4",  # 1 Spleen
    "#ff7f0e",  # 2 R. Kid
    "#2ca02c",  # 3 L. Kid
    "#d62728",  # 4 Gall.
    "#9467bd",  # 5 Eso.
    "#8c564b",  # 6 Liver
    "#e377c2",  # 7 Stom.
    "#63fa3d",  # 8 Aorta
    "#bcbd22",  # 9 IVC
    "#17becf",  # 10 PSV
    "#393b79",  # 11 Panc.
    "#637939",  # 12 RAG
    "#8c6d31",  # 13 LAG
]

# 按当前需求，仅可视化该病例的固定切片
VIS_CASE_ID = "img0022"
VIS_SLICE_INDEX = 165


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer BTCV cases and visualize only axial slice 165 of img0022 as two PNGs (GT and Pred)."
    )
    parser.add_argument("--data_dir", type=str, default="../data/btcv")
    parser.add_argument("--test_files_json", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--pred_output_dir", type=str, default="./eval_AttU_Net3D/predictions")
    parser.add_argument("--vis_output_dir", type=str, default="./eval_AttU_Net3D/visualizations")

    parser.add_argument("--pixdim", type=float, nargs=3, default=[1.5, 1.5, 2.0])
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--num_classes", type=int, default=14)

    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--sw_overlap", type=float, default=0.5)

    parser.add_argument("--hu_min", type=float, default=-175.0)
    parser.add_argument("--hu_max", type=float, default=250.0)

    parser.add_argument("--figure_dpi", type=int, default=800)
    parser.add_argument("--overlay_alpha", type=float, default=0.52)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_json_cases(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for k in ["test", "testing", "validation", "data"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]

    raise ValueError(f"无法从 {json_path} 解析测试样本列表。")


def resolve_path(data_dir: str, p: Optional[str]) -> Optional[str]:
    """
    忽略 JSON 中的旧目录前缀，只使用文件名并按 BTCV 标准目录重组。
    """
    if p is None:
        return None

    p_norm = os.path.normpath(p)
    filename = os.path.basename(p_norm)
    p_lower = p_norm.replace("/", "\\").lower()

    if os.path.isabs(p_norm) and os.path.isfile(p_norm):
        return p_norm

    if "label" in p_lower:
        candidate = os.path.normpath(os.path.join(data_dir, "labelsTr", filename))
    else:
        candidate = os.path.normpath(os.path.join(data_dir, "imagesTr", filename))

    return candidate


def get_case_id_from_path(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_affine(x) -> np.ndarray:
    if hasattr(x, "affine"):
        affine = x.affine
        if hasattr(affine, "detach"):
            affine = affine.detach().cpu().numpy()
        else:
            affine = np.asarray(affine)
        return affine.astype(np.float64)

    if hasattr(x, "meta") and isinstance(x.meta, dict):
        if "affine" in x.meta:
            affine = x.meta["affine"]
            if hasattr(affine, "detach"):
                affine = affine.detach().cpu().numpy()
            else:
                affine = np.asarray(affine)
            return affine.astype(np.float64)

    raise RuntimeError("无法从变换后的图像中提取 affine，请检查 MONAI 版本。")


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def build_infer_transform(pixdim: Tuple[float, float, float], hu_min: float, hu_max: float):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(
            keys=["image"],
            axcodes="RAS",
            labels=(("L", "R"), ("P", "A"), ("I", "S")),
        ),
        Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_min,
            a_max=hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image"], track_meta=True),
    ])


def build_vis_image_transform(pixdim: Tuple[float, float, float]):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(
            keys=["image"],
            axcodes="RAS",
            labels=(("L", "R"), ("P", "A"), ("I", "S")),
        ),
        Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
        EnsureTyped(keys=["image"], track_meta=True),
    ])


def build_label_transform(pixdim: Tuple[float, float, float]):
    return Compose([
        LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
        Orientationd(
            keys=["label"],
            axcodes="RAS",
            labels=(("L", "R"), ("P", "A"), ("I", "S")),
        ),
        Spacingd(keys=["label"], pixdim=pixdim, mode="nearest"),
        EnsureTyped(keys=["label"], track_meta=True),
    ])


@torch.no_grad()
def infer_one_case(
    model: torch.nn.Module,
    image_tensor,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    sw_overlap: float,
) -> np.ndarray:
    x = image_tensor
    if x.ndim == 4:  # [C, H, W, D]
        x = x.unsqueeze(0)  # [B, C, H, W, D]
    x = x.to(device=device, dtype=torch.float32)

    logits = sliding_window_inference(
        inputs=x,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=sw_overlap,
        mode="gaussian",
    )
    pred = torch.argmax(logits, dim=1)  # [B, H, W, D]
    pred_np = pred[0].detach().cpu().numpy().astype(np.uint8)
    return pred_np


def save_nifti(array3d: np.ndarray, affine: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(array3d, affine), out_path)


def rotate_axial_for_display(arr3d: np.ndarray, z: int) -> np.ndarray:
    return np.rot90(arr3d[:, :, z])


def draw_single_overlay_figure(
    ct3d_hu_clipped: np.ndarray,
    seg3d: np.ndarray,
    z: int,
    out_png: str,
    dpi: int = 800,
    alpha: float = 0.52,
):
    """
    仅输出一张无图注 PNG：CT 灰度底图 + 分割彩色叠加。
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    ct2d = rotate_axial_for_display(ct3d_hu_clipped, z)
    seg2d = rotate_axial_for_display(seg3d, z)

    cmap = ListedColormap(CLASS_COLORS)

    fig = plt.figure(figsize=(6, 6), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(ct2d, cmap="gray", vmin=np.min(ct2d), vmax=np.max(ct2d))

    masked_seg = np.ma.masked_where(seg2d == 0, seg2d)
    ax.imshow(masked_seg, cmap=cmap, alpha=alpha, vmin=0, vmax=13, interpolation="nearest")

    ax.axis("off")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()

    os.makedirs(args.pred_output_dir, exist_ok=True)
    os.makedirs(args.vis_output_dir, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    pixdim = tuple(float(x) for x in args.pixdim)
    roi_size = tuple(int(x) for x in args.roi_size)
    target_cases: Set[str] = {VIS_CASE_ID}

    model = AttU_Net3D(input_channel=1).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    cases = load_json_cases(args.test_files_json)

    infer_transform = build_infer_transform(
        pixdim=pixdim,
        hu_min=float(args.hu_min),
        hu_max=float(args.hu_max),
    )
    vis_image_transform = build_vis_image_transform(pixdim=pixdim)
    label_transform = build_label_transform(pixdim=pixdim)

    summary = []

    for item in cases:
        if not isinstance(item, dict):
            continue
        if "image" not in item:
            continue

        image_path = resolve_path(args.data_dir, item["image"])
        label_path = resolve_path(args.data_dir, item.get("label", None))

        if image_path is None or (not image_path.endswith(".nii") and not image_path.endswith(".nii.gz")):
            continue

        case_id = get_case_id_from_path(image_path)

        if case_id not in target_cases:
            continue

        if not os.path.isfile(image_path):
            print(f"[SKIP] image not found: {image_path}")
            print(f"       abs path      : {os.path.abspath(image_path)}")
            continue

        if label_path is None or (not os.path.isfile(label_path)):
            print(f"[SKIP] label not found for {case_id}: {label_path}")
            print(f"       abs path      : {os.path.abspath(label_path) if label_path is not None else 'None'}")
            continue

        print(f"[INFO] infer {case_id}")

        # 1) 推理图像：重采样 + HU 截断 + 归一化
        infer_data = infer_transform({"image": image_path})
        infer_img = infer_data["image"]

        # 2) 可视化 CT：重采样 + HU 截断（不归一化）
        vis_data = vis_image_transform({"image": image_path})
        vis_img = to_numpy(vis_data["image"]).astype(np.float32)[0]   # [H, W, D]
        vis_img_hu = np.clip(vis_img, float(args.hu_min), float(args.hu_max))

        # 3) GT：重采样到相同空间
        lab_data = label_transform({"label": label_path})
        gt_np = to_numpy(lab_data["label"]).astype(np.uint8)[0]

        # 4) 推理
        pred_np = infer_one_case(
            model=model,
            image_tensor=infer_img,
            device=device,
            roi_size=roi_size,
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
        )

        # 5) 形状检查
        if pred_np.shape != vis_img_hu.shape:
            raise RuntimeError(
                f"{case_id} shape mismatch: pred={pred_np.shape}, vis_ct={vis_img_hu.shape}"
            )
        if gt_np.shape != pred_np.shape:
            raise RuntimeError(
                f"{case_id} GT shape mismatch: gt={gt_np.shape}, pred={pred_np.shape}"
            )

        if VIS_SLICE_INDEX < 0 or VIS_SLICE_INDEX >= gt_np.shape[-1]:
            raise RuntimeError(
                f"{case_id} fixed slice index out of range: z={VIS_SLICE_INDEX}, depth={gt_np.shape[-1]}"
            )

        affine = extract_affine(vis_data["image"])

        # 6) 保存 NIfTI
        case_pred_dir = os.path.join(args.pred_output_dir, case_id)
        case_vis_dir = os.path.join(args.vis_output_dir, case_id)
        os.makedirs(case_pred_dir, exist_ok=True)
        os.makedirs(case_vis_dir, exist_ok=True)

        pred_out_path = os.path.join(case_pred_dir, f"{case_id}_pred.nii.gz")
        ct_out_path = os.path.join(case_vis_dir, f"{case_id}_ct_resampled_hu_clipped.nii.gz")
        gt_out_path = os.path.join(case_vis_dir, f"{case_id}_gt_resampled.nii.gz")

        save_nifti(pred_np.astype(np.uint8), affine, pred_out_path)
        save_nifti(vis_img_hu.astype(np.float32), affine, ct_out_path)
        save_nifti(gt_np.astype(np.uint8), affine, gt_out_path)

        # 7) 仅可视化固定第 165 个 axial 切片，且只导出两张图：GT 和 Pred
        z = int(VIS_SLICE_INDEX)
        gt_png = os.path.join(case_vis_dir, f"{case_id}_axial_z{z:03d}_gt.png")
        pred_png = os.path.join(case_vis_dir, f"{case_id}_axial_z{z:03d}_pred.png")

        draw_single_overlay_figure(
            ct3d_hu_clipped=vis_img_hu,
            seg3d=gt_np,
            z=z,
            out_png=gt_png,
            dpi=args.figure_dpi,
            alpha=args.overlay_alpha,
        )
        draw_single_overlay_figure(
            ct3d_hu_clipped=vis_img_hu,
            seg3d=pred_np,
            z=z,
            out_png=pred_png,
            dpi=args.figure_dpi,
            alpha=args.overlay_alpha,
        )

        summary.append({
            "case_id": case_id,
            "image": image_path,
            "label": label_path,
            "pred_nifti": pred_out_path,
            "ct_resampled_hu_clipped_nifti": ct_out_path,
            "gt_resampled_nifti": gt_out_path,
            "visualized_case_only": VIS_CASE_ID,
            "visualized_axial_slice_index": z,
            "gt_png": gt_png,
            "pred_png": pred_png,
        })

    summary_path = os.path.join(args.vis_output_dir, "selected_cases_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n================ DONE ================")
    print(f"Predictions saved to: {args.pred_output_dir}")
    print(f"Visualizations saved to: {args.vis_output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
