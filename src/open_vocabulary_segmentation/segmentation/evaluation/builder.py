# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import torch
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.datasets.pipelines import Compose
from omegaconf import OmegaConf
from datasets import get_template

def build_dataset_class_tokens(text_transform, template_set, classnames):
    tokens = []
    templates = get_template(template_set)
    for classname in classnames:
        tokens.append(
            torch.stack([text_transform(template.format(classname)) for template in templates])
        )
    # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
    tokens = torch.stack(tokens)

    return tokens


def build_seg_dataset(config):
    """Build a dataset from config."""
    cfg = mmcv.Config.fromfile(config)
    dataset = build_dataset(cfg.data.test)
    return dataset


def build_seg_dataloader(dataset):
    # batch size is set to 1 to handle varying image size (due to different aspect ratio)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=True,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False,
    )
    return data_loader
