# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def build_scheduler(config, optimizer):
    num_steps = config.total_steps
    min_lr = config.min_lr
    warmup_steps = config.warmup_steps

    lr_scheduler = None
    if config.lr_scheduler.name == "cosine":
        warmup_sched = LinearLR(
            optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps
        )
        cos_sched = CosineAnnealingLR(optimizer, T_max=num_steps - warmup_steps, eta_min=min_lr)
        lr_scheduler = SequentialLR(optimizer, [warmup_sched, cos_sched], milestones=[warmup_steps])
    else:
        raise NotImplementedError(f"lr scheduler {config.lr_scheduler.name} not implemented")

    return lr_scheduler
