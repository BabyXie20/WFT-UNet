import csv
from collections import OrderedDict

import torch
from fvcore.nn import FlopCountAnalysis
from monai.networks.nets import SwinUNETR, UNETR

from networks.Att_UNet import AttU_Net3D
from networks.TraBTS import BTS
from networks.UNet3D import UNet3D
from networks.UNetr_plusplus import UNETR_PP
from networks.UXNet import UXNET
from networks.VSNet import VSNet
from networks.model import VNet


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


def count_gparams(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e9


def count_gflops(model, x):
    flops = FlopCountAnalysis(model, x)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    total_flops = flops.total()
    return total_flops / 1e9


def format_4_sig(value):
    return f"{value:.4g}"


def save_csv(csv_path, rows):
    fieldnames = ["model", "GParams", "GFlops"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_models():
    return [
        lambda: UNETR(
            in_channels=1,
            out_channels=14,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
            spatial_dims=3,
        ),
        lambda: SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=14,
            feature_size=24,
            spatial_dims=3,
        ),
        lambda: BTS(
            img_dim=96,
            patch_dim=8,
            num_channels=1,
            num_classes=14,
            embedding_dim=512,
            num_heads=8,
            num_layers=4,
            hidden_dim=4096,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            positional_encoding_type="learned",
            apply_softmax=False,
        ),
        lambda: UNet3D(in_channels=1, out_channels=14, f_maps=24),
        lambda: AttU_Net3D(input_channel=1, num_classes=14),
        lambda: VSNet(in_channels=1, out_channels=14, img_size=96),
        lambda: VNet(n_channels=1, n_classes=14),
        lambda: UNETR_PP(in_channels=1, out_channels=14, img_size=(96, 96, 96)),
        lambda: UXNET(in_chans=1, out_chans=14),
    ]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_csv = "model_stats.csv"

    x = torch.randn(1, 1, 96, 96, 96).to(device)
    rows = []

    for create_model in build_models():
        model = create_model().to(device)
        model.eval()
        model_name = model.__class__.__name__

        gparams = count_gparams(model)

        try:
            with torch.no_grad():
                gflops = count_gflops(model, x)
            gflops_str = format_4_sig(gflops)
        except Exception as err:
            print(f"[FLOPs] {model_name} 计算失败: {err}")
            gflops_str = "N/A"

        rows.append(
            {
                "model": model_name,
                "GParams": format_4_sig(gparams),
                "GFlops": gflops_str,
            }
        )

        print(f"{model_name:12s} | GParams={format_4_sig(gparams)} | GFlops={gflops_str}")

    save_csv(output_csv, rows)
    print(f"结果已保存到: {output_csv}")


if __name__ == "__main__":
    main()
