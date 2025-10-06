# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
from typing import Dict, List, Any
from datetime import datetime
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

# ImageNet mean/std (from timm)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_MEAN = IMAGENET_DEFAULT_MEAN
DEFAULT_STD = IMAGENET_DEFAULT_STD

# NOTE Originally CLIP statistics should be used, but the legacy of ImageNet statistics
# from GroupViT is applied. Fortunately, CLIP is quite robust to slightly different
# normalization constants (https://github.com/openai/CLIP/issues/20#issuecomment-764985771).


def unnorm(x):
    mean = torch.as_tensor(DEFAULT_MEAN, device=x.device)[None, ..., None, None]
    std = torch.as_tensor(DEFAULT_STD, device=x.device)[None, ..., None, None]
    return x.mul(std).add(mean)


# DEBUG NaN
def check_nonfinite(x, name=""):
    rank = dist.get_rank()
    n_nan = x.isnan().sum()
    n_inf = x.isinf().sum()
    if n_nan or n_inf:
        print(f"[RANK {rank}] {name} is not finite: #nan={n_nan}, #inf={n_inf}")
        return True

    print(f"[RANK {rank}] {name} is OK ...")
    return False


def normalize(t, dim, eps=1e-6):
    """Large default eps for fp16"""
    return F.normalize(t, dim=dim, eps=eps)


def timestamp(fmt="%y%m%d-%H%M%S"):
    return datetime.now().strftime(fmt)


def merge_dicts_by_key(dics: List[Dict]) -> Dict[Any, List]:
    """Merge dictionaries by key. All of dicts must have same keys."""
    ret = {key: [] for key in dics[0].keys()}
    for dic in dics:
        for key, value in dic.items():
            ret[key].append(value)

    return ret


def flatten_2d_list(list2d):
    return list(chain.from_iterable(list2d))


def num_params(module):
    return sum(p.numel() for p in module.parameters())


def param_trace(name, module, depth=0, max_depth=999, threshold=0, printf=print):
    if depth > max_depth:
        return
    prefix = "  " * depth
    n_params = num_params(module)
    if n_params > threshold:
        printf("{:60s}\t{:10.3f}M".format(prefix + name, n_params / 1024 / 1024))
    for n, m in module.named_children():
        if depth == 0:
            child_name = n
        else:
            child_name = "{}.{}".format(name, n)
        param_trace(child_name, m, depth + 1, max_depth, threshold, printf)


@torch.no_grad()
def hash_bn(module):
    summary = []
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            w = m.weight.detach().mean().item()
            b = m.bias.detach().mean().item()
            rm = m.running_mean.detach().mean().item()
            rv = m.running_var.detach().mean().item()
            summary.append((w, b, rm, rv))

    if not summary:
        return 0.0, 0.0

    w, b, rm, rv = [np.mean(col) for col in zip(*summary)]
    p = np.mean([w, b])
    s = np.mean([rm, rv])

    return p, s


@torch.no_grad()
def hash_params(module):
    return torch.as_tensor([p.mean() for p in module.parameters()]).mean().item()


@torch.no_grad()
def hashm(module):
    p = hash_params(module)
    _, s = hash_bn(module)

    return p, s
