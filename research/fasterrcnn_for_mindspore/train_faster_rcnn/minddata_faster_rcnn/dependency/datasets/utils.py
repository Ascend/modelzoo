import copy
import sys
from collections import Sequence

import mmcv
# from mmcv.runner import obj_from_dict
# import torch

import matplotlib.pyplot as plt
import numpy as np
from .concat_dataset import ConcatDataset
from .repeat_dataset import RepeatDataset
from .. import datasets

# TODO: this is the origin get_dist_info function, how should it be modified as for "dist" ?
# import torch.distributed as dist
# def get_dist_info():
#     if torch.__version__ < '1.0':
#         initialized = dist._initialized
#     else:
#         if dist.is_available():
#             initialized = dist.is_initialized()
#         else:
#             initialized = False
#     if initialized:
#         rank = dist.get_rank()
#         world_size = dist.get_world_size()
#     else:
#         rank = 0
#         world_size = 1
#     return rank, world_size


def get_dist_info():
    rank = 0
    world_size = 1
    return rank, world_size


# # TODO: this is copy from mmcv.runner, need to modify
def obj_from_dict(dictInfo, originType=None, input_params=None):  # parent datasetType; info -->dictInfo;args --> params
    """Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.

    Args:
        dictInfo (dict): Object types and arguments.
        originType (:class:`module`): Module which may containing expected object
            classes.
        input_params (dict, optional): Default arguments for initializing the
            object.

    Returns:
        any type: Object built from the dict.
    """

    # assert isinstance(dictInfo, dict) and 'type' in dictInfo
    if type(dictInfo) != dict or 'type' not in dictInfo:
        raise TypeError('input must be a dict and contain "type" key ')
    # assert isinstance(default_args, dict) or default_args is None
    if type(input_params) != dict and input_params is not None:
        raise TypeError('input default_args must be a dict')

    params = dictInfo.copy()
    target_type = params.pop('type')     # obj_type --> target_type; default_args--> input_params
    if input_params is not None:
        for key, value in input_params.items():  # name --> key
            params.setdefault(key, value)
    if type(target_type) == str:
        if originType is None:
            target_type = sys.modules[target_type]
        else:
            target_type = getattr(originType, target_type)
    elif type(target_type) != type:
        raise TypeError('target type must be a str or other valid type')
    return target_type(**params)


# def to_tensor(data):
#     """Convert objects of various python types to :obj:`torch.Tensor`.
#
#     Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
#     :class:`Sequence`, :class:`int` and :class:`float`.
#     """
#     if isinstance(data, torch.Tensor):
#         return data
#     elif isinstance(data, np.ndarray):
#         return torch.from_numpy(data)
#     elif isinstance(data, Sequence) and not mmcv.is_str(data):
#         return torch.tensor(data)
#     elif isinstance(data, int):
#         return torch.LongTensor([data])
#     elif isinstance(data, float):
#         return torch.FloatTensor([data])
#     else:
#         raise TypeError('type {} cannot be converted to tensor.'.format(
#             type(data)))


def to_numpy(data):
    return np.asarray(data)


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()


def get_dataset(data_cfg):
    if data_cfg['type'] == 'RepeatDataset':
        return RepeatDataset(
            get_dataset(data_cfg['dataset']), data_cfg['times'])

    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if 'proposal_file' in data_cfg.keys():
        if isinstance(data_cfg['proposal_file'], (list, tuple)):
            proposal_files = data_cfg['proposal_file']
        else:
            proposal_files = [data_cfg['proposal_file']]
    else:
        proposal_files = [None] * num_dset
    assert len(proposal_files) == num_dset

    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']] * num_dset
    assert len(img_prefixes) == num_dset

    dsets = []
    for i in range(num_dset):
        data_info = copy.deepcopy(data_cfg)
        data_info['ann_file'] = ann_files[i]
        data_info['proposal_file'] = proposal_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        dset = obj_from_dict(data_info, datasets)
        dsets.append(dset)
    if len(dsets) > 1:
        dset = ConcatDataset(dsets)
    else:
        dset = dsets[0]
    return dset
