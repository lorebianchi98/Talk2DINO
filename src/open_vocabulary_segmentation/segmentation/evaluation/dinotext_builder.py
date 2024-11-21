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

from .dinotext_seg import DINOTextSegInference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dinotext_seg_inference(
    model,
    dataset,
    config,
    seg_config,
):
    dset_cfg = mmcv.Config.fromfile(seg_config)  # dataset config
    with_bg = dataset.dataset.CLASSES[0] == "background"
    if with_bg:
        classnames = dataset.dataset.CLASSES[1:]
    else:
        classnames = dataset.dataset.CLASSES
    text_tokens = model.build_dataset_class_tokens(config.evaluate.template, classnames)
    text_embedding = model.build_text_embedding(text_tokens)
    kwargs = dict(with_bg=with_bg)
    if hasattr(dset_cfg, "test_cfg"):
        kwargs["test_cfg"] = dset_cfg.test_cfg

    model_type = config.model.type
    if model_type == "DINOText":
        seg_model = DINOTextSegInference(model, text_embedding, classnames, **kwargs, **config.evaluate)
    else:
        raise ValueError(model_type)

    seg_model.CLASSES = dataset.dataset.CLASSES
    seg_model.PALETTE = dataset.dataset.PALETTE

    return seg_model
