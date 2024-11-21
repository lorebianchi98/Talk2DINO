# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import os
import re
from collections import defaultdict

import torch
from mmcv.runner import CheckpointLoader
from omegaconf import read_write
from torch.nn.parallel.distributed import DistributedDataParallel

from .logger import get_logger

missing_keys_whitelist = [r"clip_.+_encoder\..+", r".+_loss\..+"]


def check_whitelist(key, whitelist):
    """Check whether the given key matches any of the patterns in the whitelist.
    """
    for pattern in whitelist:
        if re.match(pattern, key):
            return True
    return False

def clean_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    return new_state_dict

def load_checkpoint(config, model, optimizer, lr_scheduler, scaler):
    logger = get_logger()
    logger.info(f"==============> Resuming form {config.checkpoint.resume}....................")
    checkpoint = CheckpointLoader.load_checkpoint(config.checkpoint.resume, map_location="cpu")
    if "model" in checkpoint:
        msg = model.load_state_dict(checkpoint["model"], strict=False)
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        state_dict = clean_state_dict(state_dict)
        msg = model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError("No model state_dict found in checkpoint.")
    whitelist_cnt = 0
    
    if msg.missing_keys or msg.unexpected_keys:
        logger.info("#" * 80)
        if msg.missing_keys:
            logger.info("!!! Missing keys !!!")
            for key in msg.missing_keys:
                if check_whitelist(key, missing_keys_whitelist):
                    whitelist_cnt += 1
                    continue

                logger.info(f"\t {key}")

            logger.info(f"Whitelist of missing keys: {missing_keys_whitelist}")
            logger.info(f"# of Whitelisted missing keys = {whitelist_cnt}")
            if len(msg.missing_keys) > whitelist_cnt:
                logger.warning("Please check the missing keys.")

        if msg.unexpected_keys:
            if len(msg.unexpected_keys) == 1 and msg.unexpected_keys[0] == "contrastive_loss.logit_scale":
                logger.info("Ignoring unexpected key: contrastive_loss.logit_scale")
            else:   
                logger.info("!!! Unexpected keys !!!")
                for key in msg.unexpected_keys:
                    logger.info(f"\t {key}")

                raise ValueError("Unexpected keys are found in the checkpoint.")
            logger.info("#" * 80)

    metrics = defaultdict(float)
    is_resume = (
        not config.evaluate.eval_only
        and "optimizer" in checkpoint
        and "lr_scheduler" in checkpoint
        and "step" in checkpoint
    )
    if is_resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        with read_write(config):
            config.train.start_step = checkpoint["step"] + 1
        scaler.load_state_dict(checkpoint["scaler"])
        logger.info(
            f"=> loaded successfully '{config.checkpoint.resume}' (step {checkpoint['step']})"
        )
        default_metrics = {"max_accuracy": 0.0, "max_voc_miou": 0.0, "max_context_miou": 0.0}
        metrics = checkpoint.get("metrics", default_metrics)

    del checkpoint
    torch.cuda.empty_cache()
    return metrics


class CheckpointManager:
    def __init__(self, k, output_dir):
        self.k = k
        self.output_dir = output_dir
        self.ckpts = []

    def add(self, miou, ckpt_kwargs, step):
        fn = f"ckpt_{step}_miou{miou:.2f}.pth"
        self.ckpts.append((miou, fn))
        # TODO heapq
        self.ckpts = sorted(self.ckpts, key=lambda x: x[0], reverse=True)

        #  print(f"Step {step} | #ckpts = {len(self.ckpts)} | ckpts = {self.ckpts}")

        save_cur = True
        if len(self.ckpts) > self.k:
            assert len(self.ckpts) == self.k + 1
            remove_fn = self.ckpts.pop(-1)[1]
            if remove_fn == fn:
                save_cur = False
            else:
                self.remove(remove_fn)

        if save_cur:
            self.save(fn, ckpt_kwargs)

    def save(self, fn, ckpt_kwargs):
        save_checkpoint(**ckpt_kwargs, filename=fn)

    def remove(self, fn):
        path = os.path.join(self.output_dir, fn)
        os.remove(path)


def save_checkpoint(
    config, step, model, optimizer, lr_scheduler, scaler, metrics=None, filename="checkpoint.pth"
):
    if isinstance(model, DistributedDataParallel):
        model = model.module

    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": config,
    }
    if metrics is not None:
        save_state["metrics"] = metrics

    torch.save(save_state, os.path.join(config.output, filename))