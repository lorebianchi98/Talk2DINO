# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
from torch import optim as optim


def set_weight_decay(named_parameters, config):
    has_decay = []
    no_decay = []

    for name, param in named_parameters:
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(config, model):
    """Build optimizer, set weight decay of normalization to 0 by default."""
    parameters = set_weight_decay(model.named_parameters(), config)

    opt_name = config.optimizer.name
    if opt_name == "adamw":
        optimizer = optim.AdamW(
            parameters,
            eps=config.optimizer.eps,
            betas=config.optimizer.betas,
            lr=config.base_lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    return optimizer
