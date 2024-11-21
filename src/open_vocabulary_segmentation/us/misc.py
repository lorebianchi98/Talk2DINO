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

import os.path as osp
import tempfile
import warnings

import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmseg.apis.test import np2tmp

device = "cuda" if torch.cuda.is_available() else "cpu"

from typing import Optional
def collect_results_cpu(result_part: list,
                        size: int,
                        tmpdir: Optional[str] = None) -> Optional[list]:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    # rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    part_file = osp.join(tmpdir, f'part_{rank}.pkl')  # type: ignore
    mmcv.dump(result_part, part_file)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            part_result = mmcv.load(part_file)
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)  # type: ignore
        return ordered_results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    pred_qualitatives = []
    gt_qualitatives = []

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            if device == 'cpu':
                data['img_metas'] = [e.data[0] for e in data['img_metas']]
            result = model(return_loss=False, rescale=True, **data)

        for pred_qualitative, index in zip(result, batch_indices):
            pred_qualitatives.append(pred_qualitative+1)
            seg_map_gt = dataset.dataset.get_gt_seg_map_by_idx(index + dataset.indices.start)
            # seg_map_gt[seg_map_gt == 255] = 0
            gt_qualitatives.append(seg_map_gt)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.dataset.pre_eval(result, indices=[i+dataset.indices.start for i in batch_indices])

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if world_size > 1:
        if gpu_collect:
            results = collect_results_gpu(results, len(dataset))
        else:
            results = collect_results_cpu(results, len(dataset), tmpdir)
    return results, pred_qualitatives, gt_qualitatives, len(dataset.dataset.CLASSES)
