#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock


class LayerNorm(nn.Module):
    """
    支持两种数据格式的 LayerNorm:
    - channels_last: (B, D, H, W, C)
    - channels_first: (B, C, D, H, W)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data_format: {data_format}")

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )

        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class UXBlock(nn.Module):
    """
    3D ConvNeXt-style block:
    DwConv -> LayerNorm -> grouped 1x1 Conv -> GELU -> grouped 1x1 Conv -> residual
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)

        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)   # (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)   # -> (B, C, D, H, W)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)

        x = residual + self.drop_path(x)
        return x


class UXNetConv(nn.Module):
    """
    UXNet 3D encoder backbone.
    """
    def __init__(
        self,
        in_chans=1,
        depths=(2, 2, 2, 2),
        dims=(48, 96, 192, 384),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=(0, 1, 2, 3),
    ):
        super().__init__()

        self.out_indices = out_indices

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[
                    UXBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i in range(4):
            self.add_module(f"norm{i}", norm_layer(dims[i]))

    def forward(self, x):
        outs = []

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                outs.append(norm_layer(x))

        return tuple(outs)


class UXNET(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=14,
        depths=(2, 2, 2, 2),
        feat_size=(48, 96, 192, 384),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        hidden_size=768,
        norm_name: Union[Tuple, str] = "instance",
        res_block=True,
        spatial_dims=3,
    ):
        super().__init__()

        out_indices = tuple(range(len(feat_size)))

        self.backbone = UXNetConv(
            in_chans=in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            out_indices=out_indices,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=out_chans,
        )

    def forward(self, x_in):
        outs = self.backbone(x_in)

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])

        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)

        return self.out(out)