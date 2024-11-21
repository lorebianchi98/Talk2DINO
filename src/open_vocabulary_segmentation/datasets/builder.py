# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------------
import os.path as osp
import random
import warnings
from functools import partial

import numpy as np
import torch.distributed as dist
import webdataset as wds
from braceexpand import braceexpand
from .collate import collate
from timm.data import create_transform
from torchvision import transforms as T
import us

# from sclip.clip import tokenize


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(config):
    dataset_train = build_dataset(config=config)
    us.dprint("successfully build train dataset")

    dc_collate = partial(collate, samples_per_gpu=config.batch_size)
    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=dist.get_rank(), seed=config.seed
    )
    data_loader_train = wds.WebLoader(
        dataset_train.batched(config.batch_size, dc_collate, partial=False),
        batch_size=None,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=init_fn,
    )

    train_len = len(dataset_train)
    train_nbatches = max(1, train_len // (config.batch_size * dist.get_world_size()))
    data_loader_train = data_loader_train.with_epoch(train_nbatches).with_length(train_nbatches)

    return dataset_train, data_loader_train


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning,
    and continue."""
    warnings.warn(repr(exn))
    return True


def build_dataset(config):
    """
    Args:
        config: CONFIG.data (CONFIG = global config)
    """
    img_transform = build_img_transform(config.img_aug)
    text_transform = build_text_transform()
    split = "train"
    dataset_type = None
    tar_file_list = []
    total_length = 0
    for ds in config.dataset[split]:
        ds_meta = config.dataset.meta[ds]
        if dataset_type is None:
            dataset_type = ds_meta.type
        else:
            assert dataset_type == ds_meta.type, "All datasets must be of the same type"

        prefix = ds_meta.prefix
        path = ds_meta.path
        length = ds_meta.length
        cur_tar_file_list = []
        for tar_file in braceexpand(osp.join(path, prefix)):
            if osp.exists(tar_file):
                cur_tar_file_list.append(tar_file)
        print(f"Found {len(cur_tar_file_list)} files for dataset {ds}")
        tar_file_list.extend(cur_tar_file_list)
        total_length += length

    print(f"Found {len(tar_file_list)} files in total for split {split}")
    dataset = (
        wds.WebDataset(tar_file_list, repeat=True, handler=warn_and_continue)
        .shuffle(40000)  # datapoint-level shuffle
        .decode("pil", handler=warn_and_continue)
        .rename(
            image="jpg;png;jpeg",
            text="text;txt",
            org_caption="text;txt",
            keep=False,
            handler=warn_and_continue,
        )
        .map_dict(image=img_transform, text=text_transform, handler=warn_and_continue)
        .with_length(total_length)
    )

    return dataset


def build_img_transform(config):
    if not config.deit_aug:
        transform = T.Compose(
            [
                T.RandomResizedCrop(config.img_size, scale=config.img_scale),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=us.DEFAULT_MEAN, std=us.DEFAULT_STD),
            ]
        )
    else:
        # deit_aug
        transform = create_transform(
            input_size=config.img_size,
            is_training=True,
            color_jitter=config.color_jitter if config.color_jitter > 0 else None,
            auto_augment=config.auto_augment if config.auto_augment != "none" else None,
            re_prob=config.re_prob,
            re_mode=config.re_mode,
            re_count=config.re_count,
        )

    return transform


def build_text_transform():
    transform = Tokenize()

    return transform


class Tokenize:
    """Wrapper class for CLIP tokenize function."""

    def __init__(self, max_seq_len=77, truncate=True):
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True

        result = tokenize(texts, self.max_seq_len, self.truncate)

        if expanded_dim:
            return result[0]

        return result
