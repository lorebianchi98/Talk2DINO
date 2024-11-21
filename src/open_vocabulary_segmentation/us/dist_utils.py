# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
import os
import sys
import torch
import torch.distributed as dist
import torch.distributed.nn


def dist_info() -> str:
    """Check distributed training env variables"""
    keys = [
        "NODE_RANK",
        "GROUP_RANK",
        "LOCAL_RANK",
        "RANK",
        "GLOBAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        # for now, torch.distributed.run env variables
        # https://github.com/pytorch/pytorch/blob/d69c22dd61/torch/distributed/run.py#L121
        "ROLE_RANK",
        "LOCAL_WORLD_SIZE",
        "WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
    ]
    rs = []
    for key in keys:
        r = os.getenv(key)
        if r:
            s = f"{key} = {r}"
            rs.append(s)

    return " | ".join(rs)


def dprint(s: str, printf=print):
    rank = get_node_rank()
    local_rank = os.getenv("LOCAL_RANK")

    kwargs = {}
    if printf == print:
        kwargs["flush"] = True

    printf(f"[NODE RANK {rank}, LOCAL RANK {local_rank}] " + s, **kwargs)


def get_local_rank():
    """Pytorch lightning save local rank to environment variable "LOCAL_RANK"."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank


def get_node_rank():
    """ """
    rank = os.getenv("NODE_RANK") or os.getenv("GROUP_RANK")
    if rank:
        rank = int(rank)
    return rank


def is_master():
    """Check whether here is master node"""
    node_rank = get_node_rank()
    return node_rank in [None, 0]


def is_rank_zero():
    return get_local_rank() == 0


def is_global_zero():
    return get_node_rank() in [None, 0] and get_local_rank() == 0


class ContiguousGrad(torch.autograd.Function):
    """some distributed operations (e.g. all_gather) require contiguous input,
    but sometimes following op generates non-contiguous gradient (e.g. einsum).
    At that case, this class makes the gradient contiguous.

    Usage:
        x = ContiguousGrad.apply(x)
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()


def gather_cat(x: torch.Tensor, grad=False, contiguous_grad=False) -> torch.Tensor:
    """Gather tensors & concat
    [!] distributed operations should be executed in all devices.
    i.e. you should not use a distributed op with is_rank_zero().

    Args:
        x (torch.tensor; [D, ...])
        grad (bool): if True, gather tensors with gradient flow
        contiguous_grad (bool): apply ContiguousGrad to the output tensor to ensure
            the contiguous gradient. A distributed op requires contiguous input.

    Return: torch.tensor; [D*n_gpus, ...]
    """
    if not grad:
        gathers = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gathers, x)
    else:
        gathers = torch.distributed.nn.all_gather(x)

    if x.ndim == 0:
        gathers = torch.stack(gathers)
    else:
        gathers = torch.cat(gathers)

    if contiguous_grad:
        gathers = ContiguousGrad.apply(gathers)

    return gathers


def reduce(x: torch.Tensor, reduce_op="mean") -> torch.Tensor:
    """
    Args:
        x
        reduce_op: sum / mean
    """
    assert reduce_op in ["sum", "mean"]
    #  dist.barrier()
    dist.all_reduce(x)

    if reduce_op == "mean":
        x /= dist.get_world_size()

    return x
