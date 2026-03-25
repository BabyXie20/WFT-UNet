import csv
import json
from collections import OrderedDict

import torch
from fvcore.nn import FlopCountAnalysis

# =========================
# 改这里：导入你自己的模型
# 例如：
# from models.unet import UNet3D
# from networks.my_model import MyModel
# =========================
from xx import xx


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        new_state_dict[new_k] = v

    incompatible = model.load_state_dict(new_state_dict, strict=False)
    print(f"[Checkpoint] Loaded from: {ckpt_path}")
    print(f"[Checkpoint] Missing keys: {len(incompatible.missing_keys)}")
    print(f"[Checkpoint] Unexpected keys: {len(incompatible.unexpected_keys)}")


def count_params_m(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def count_gflops(model, x):
    flops = FlopCountAnalysis(model, x)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    total_flops = flops.total()
    unsupported = flops.unsupported_ops()
    return total_flops / 1e9, unsupported


def save_csv(csv_path, row):
    fieldnames = [
        "model_name",
        "input_shape",
        "ckpt",
        "device",
        "total_params_m",
        "trainable_params_m",
        "gflops",
        "unsupported_ops_json",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main():
    # =========================
    # 1. 设备
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # 2. 你自己手动写参数
    # =========================
    ckpt_path = None
    # ckpt_path = "your_checkpoint.pth"

    output_csv = "model_stats.csv"

    batch_size = 1
    n_channels = 1
    roi_size = (96, 96, 96)

    # =========================
    # 3. 你自己手动实例化模型
    #    改成 model = xx(...).to(device)
    # =========================
    model = xx(
        # 这里写你模型需要的参数
        # 例如：
        # in_channels=n_channels,
        # out_channels=14,
        # feature_size=48,
    ).to(device)

    # 如果有 checkpoint 就加载
    if ckpt_path is not None:
        load_checkpoint(model, ckpt_path)

    model.eval()

    # =========================
    # 4. 构造输入
    #    如果你的模型输入尺寸不是这个形式，也在这里改
    # =========================
    x = torch.randn(batch_size, n_channels, *roi_size).to(device)

    total_params_m, trainable_params_m = count_params_m(model)

    try:
        with torch.no_grad():
            gflops, unsupported = count_gflops(model, x)
    except Exception as e:
        gflops, unsupported = None, {"flop_count_error": str(e)}
        print(f"[FLOPs] Failed to compute FLOPs: {e}")

    print("=" * 50)
    print(f"Model            : {model.__class__.__name__}")
    print(f"Input shape      : {tuple(x.shape)}")
    print(f"Total Params     : {total_params_m:.2f} M")
    print(f"Trainable Params : {trainable_params_m:.2f} M")
    if gflops is not None:
        print(f"GFLOPs           : {gflops:.2f} G")
    else:
        print("GFLOPs           : unavailable")
    print(f"CSV Saved        : {output_csv}")
    print("=" * 50)

    if unsupported:
        print("[Warning] Unsupported ops not counted by fvcore:")
        for k, v in unsupported.items():
            print(f"  {k}: {v}")

    row = {
        "model_name": model.__class__.__name__,
        "input_shape": str(tuple(x.shape)),
        "ckpt": "" if ckpt_path is None else ckpt_path,
        "device": str(device),
        "total_params_m": f"{total_params_m:.6f}",
        "trainable_params_m": f"{trainable_params_m:.6f}",
        "gflops": "" if gflops is None else f"{gflops:.6f}",
        "unsupported_ops_json": json.dumps(unsupported, ensure_ascii=False),
    }
    save_csv(output_csv, row)


if __name__ == "__main__":
    main()