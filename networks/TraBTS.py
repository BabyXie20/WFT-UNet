import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# IntermediateSequential
# =========================
class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, x):
        if not self.return_intermediate:
            return super().forward(x)

        intermediate_outputs = {}
        out = x
        for name, module in self.named_children():
            out = module(out)
            intermediate_outputs[name] = out
        return out, intermediate_outputs


# =========================
# Positional Encoding
# =========================
class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length):
        super().__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_length, embedding_dim]

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, N, C]
        return x + self.pe[:, : x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim):
        super().__init__()
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, max_position_embeddings, embedding_dim)
        )

    def forward(self, x):
        # x: [B, N, C]
        return x + self.position_embeddings[:, : x.size(1), :]


# =========================
# Transformer
# =========================
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        assert dim % heads == 0, "embedding dim must be divisible by num_heads"

        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # [3, B, heads, N, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()

        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim,
                                heads=heads,
                                dropout_rate=attn_dropout_rate,
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout_rate),
                        )
                    ),
                ]
            )

        self.net = IntermediateSequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# U-Net encoder with skip connections
# =========================
def normalization(planes, norm="gn"):
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    if norm == "gn":
        return nn.GroupNorm(8, planes)
    if norm == "in":
        return nn.InstanceNorm3d(planes)
    raise ValueError(f"normalization type {norm} is not supported")


class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout, training=self.training)
        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm="gn"):
        super().__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)

        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)

        return y + x


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels=4, base_channels=16):
        super().__init__()

        self.InitConv = InitConv(
            in_channels=in_channels, out_channels=base_channels, dropout=0.2
        )
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels * 2)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels * 2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels * 2)
        self.EnDown2 = EnDown(
            in_channels=base_channels * 2, out_channels=base_channels * 4
        )

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(
            in_channels=base_channels * 4, out_channels=base_channels * 8
        )

        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

    def forward(self, x):
        x = self.InitConv(x)

        x1_1 = self.EnBlock1(x)
        x1_2 = self.EnDown1(x1_1)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2 = self.EnDown2(x2_1)

        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_2 = self.EnDown3(x3_1)

        x4_1 = self.EnBlock4_1(x3_2)
        x4_2 = self.EnBlock4_2(x4_1)
        x4_3 = self.EnBlock4_3(x4_2)
        output = self.EnBlock4_4(x4_3)

        return x1_1, x2_1, x3_1, output


# =========================
# Decoder blocks
# =========================
class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 4

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x + residual


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(
            out_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv3 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat((prev, x), dim=1)
        x = self.conv3(x)
        return x


class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x + residual


# =========================
# Main network
# =========================
class TransformerBTS(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        positional_encoding_type="learned",
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        # 这个整理版只保留 conv_patch_representation=True 的路径
        # 由于 U-Net 下采样 3 次，因此 bottleneck 尺寸 = img_dim / 8
        # 所以这里 patch_dim 实际必须等于 8
        assert patch_dim == 8, "This simplified version only supports patch_dim=8"

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches

        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                max_position_embeddings=self.seq_length,
                embedding_dim=self.embedding_dim,
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                embedding_dim=self.embedding_dim,
                max_length=self.seq_length,
            )
        else:
            raise ValueError(
                f"Unsupported positional_encoding_type: {positional_encoding_type}"
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=hidden_dim,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        self.unet = Unet(in_channels=num_channels, base_channels=16)
        self.bn = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv_x = nn.Conv3d(
            128, self.embedding_dim, kernel_size=3, stride=1, padding=1
        )

    def encode(self, x):
        x1_1, x2_1, x3_1, x = self.unet(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_x(x)

        # [B, C, D, H, W] -> [B, N, C]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        return x1_1, x2_1, x3_1, x, intmd_x

    def _reshape_output(self, x):
        side = int(self.img_dim / self.patch_dim)
        x = x.view(
            x.size(0),
            side,
            side,
            side,
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def decode(self, x1_1, x2_1, x3_1, x, intmd_x, intmd_layers=(1, 2, 3, 4)):
        raise NotImplementedError("Should be implemented in child class.")

    def forward(self, x, auxiliary_output_layers=(1, 2, 3, 4)):
        x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs = self.encode(x)
        return self.decode(
            x1_1,
            x2_1,
            x3_1,
            encoder_output,
            intmd_encoder_outputs,
            auxiliary_output_layers,
        )


class BTS(TransformerBTS):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        positional_encoding_type="learned",
        apply_softmax=True,
    ):
        super().__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes
        self.apply_softmax = apply_softmax

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        self.DeUp4 = DeUp_Cat(
            in_channels=self.embedding_dim // 4,
            out_channels=self.embedding_dim // 8,
        )
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim // 8)

        self.DeUp3 = DeUp_Cat(
            in_channels=self.embedding_dim // 8,
            out_channels=self.embedding_dim // 16,
        )
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim // 16)

        self.DeUp2 = DeUp_Cat(
            in_channels=self.embedding_dim // 16,
            out_channels=self.embedding_dim // 32,
        )
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim // 32)

        self.endconv = nn.Conv3d(
            self.embedding_dim // 32, self.num_classes, kernel_size=1
        )

    def decode(self, x1_1, x2_1, x3_1, x, intmd_x, intmd_layers=(1, 2, 3, 4)):
        assert intmd_layers is not None and len(intmd_layers) > 0

        # 原始实现里虽然收集了多层，但最终实际只用到了最后一层
        last_layer_idx = max(intmd_layers)
        x8 = intmd_x[str(2 * last_layer_idx - 1)]

        x8 = self._reshape_output(x8)
        x8 = self.Enblock8_1(x8)
        x8 = self.Enblock8_2(x8)

        y4 = self.DeUp4(x8, x3_1)
        y4 = self.DeBlock4(y4)

        y3 = self.DeUp3(y4, x2_1)
        y3 = self.DeBlock3(y3)

        y2 = self.DeUp2(y3, x1_1)
        y2 = self.DeBlock2(y2)

        y = self.endconv(y2)

        # 若训练时使用 CrossEntropyLoss，通常应设 apply_softmax=False
        if self.apply_softmax:
            y = torch.softmax(y, dim=1)

        return y


def TransBTS(
    dataset="renal",
    _conv_repr=True,
    _pe_type="learned",
    apply_softmax=True,
):
    if not _conv_repr:
        raise ValueError(
            "This simplified single-file version only keeps conv patch representation."
        )

    dataset = dataset.lower()

    if dataset == "renal":
        img_dim = 96
        num_classes = 4
    elif dataset == "brats":
        img_dim = 128
        num_classes = 4
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Please manually set img_dim and num_classes in the factory."
        )

    num_channels = 4
    patch_dim = 8
    aux_layers = [1, 2, 3, 4]

    model = BTS(
        img_dim=img_dim,
        patch_dim=patch_dim,
        num_channels=num_channels,
        num_classes=num_classes,
        embedding_dim=512,
        num_heads=8,
        num_layers=4,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        positional_encoding_type=_pe_type,
        apply_softmax=apply_softmax,
    )

    return aux_layers, model
