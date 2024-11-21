# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class BLCModuleCompatibleBCHW(nn.Module):
    def forward_blc(self, x):
        raise NotImplementedError()

    def forward(self, x):
        is2d = x.ndim == 4
        if is2d:
            _, _, H, W = x.shape
            x = rearrange(x, "B C H W -> B (H W) C")

        x = self.forward_blc(x)

        if is2d:
            x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)

        return x


class FeatureEncoder(nn.Module):
    """Encoder + Feature extractor
    """
    def __init__(self, safe=True):
        super().__init__()
        self.safe = safe  # clone return features to protect it from after-modification
        self._features = []

    def hook(self, module, input, output):
        self._features.append(output)

    def clear_features(self):
        self._features.clear()

    def _encode(self, x):
        raise NotImplementedError()

    def forward(self, *args, ret_feats=False, **kwargs):
        self.clear_features()

        x = self._encode(*args, **kwargs)

        if ret_feats:
            if self.safe:
                features = [t.clone() for t in self._features]
                self.clear_features()
            else:
                features = self._features
            return x, features
        else:
            self.clear_features()
            return x


class Project2d(nn.Module):
    """2d projection by 1x1 conv

    Args:
        p: [C_in, C_out]
    """
    def __init__(self, p):
        # convert to 1x1 conv weight
        super().__init__()
        p = rearrange(p, "Cin Cout -> Cout Cin 1 1")
        self.p = nn.Parameter(p.detach().clone())

    def forward(self, x):
        return F.conv2d(x, self.p)  # 1x1 conv


def dispatcher(dispatch_fn):
    def decorated(key, *args):
        if callable(key):
            return key

        if key is None:
            key = "none"

        return dispatch_fn(key, *args)

    return decorated


@dispatcher
def activ_dispatch(activ):
    return {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
        "gelu": nn.GELU,
    }[activ.lower()]


def get_norm_fn(norm, C):
    """2d normalization layers
    """
    if norm is None or norm == "none":
        return nn.Identity()

    return {
        "bn": nn.BatchNorm2d(C),
        "syncbn": nn.SyncBatchNorm(C),
        "ln": LayerNorm2d(C),
        "gn": nn.GroupNorm(32, C),
    }[norm]


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps
        ).permute(0, 3, 1, 2)


class Gate(nn.Module):
    """Tanh gate"""
    def __init__(self, init=0.0):
        super().__init__()
        self.gate = nn.Parameter(torch.as_tensor(init))

    def forward(self, x):
        return torch.tanh(self.gate) * x


class ConvBlock(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size=3,
        stride=1,
        padding=1,
        norm="none",
        activ="relu",
        bias=True,
        upsample=False,
        downsample=False,
        pad_type="zeros",
        dropout=0.0,
        gate=False,
    ):
        super().__init__()
        if kernel_size == 1:
            assert padding == 0
        self.C_in = C_in
        self.C_out = C_out

        activ = activ_dispatch(activ)
        self.upsample = upsample
        self.downsample = downsample

        self.norm = get_norm_fn(norm, C_in)
        self.activ = activ()
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(p=dropout)
        self.conv = nn.Conv2d(
            C_in, C_out, kernel_size, stride, padding,
            bias=bias, padding_mode=pad_type
        )

        self.gate = Gate() if gate else None

    def forward(self, x):
        # pre-act
        x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.conv(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)

        if self.gate is not None:
            x = self.gate(x)

        return x


class ResConv(nn.Module):
    """Pre-activate residual block with single or double conv block"""

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size=3,
        stride=1,
        padding=1,
        norm="none",
        activ="relu",
        upsample=False,
        pad_type="zeros",
        dropout=0.0,
        gate=True,  # if True, use zero-init gate
        double=False,
        # norm2 and activ2 are only used when double is True
        norm2=None,  # if given, apply it to second conv
        activ2=None  # if given, apply it to second conv
    ):
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.double = double
        self.conv = ConvBlock(
            C_in, C_out, kernel_size, stride, padding, norm, activ,
            pad_type=pad_type, dropout=dropout, gate=gate,
        )
        if double:
            norm2 = norm2 or norm
            activ2 = activ2 or activ
            self.conv2 = ConvBlock(
                C_out, C_out, kernel_size, stride, padding, norm2, activ2,
                pad_type=pad_type, dropout=dropout, gate=gate
            )

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = x + self.conv(x)

        if self.double:
            x = x + self.conv2(x)

        return x
