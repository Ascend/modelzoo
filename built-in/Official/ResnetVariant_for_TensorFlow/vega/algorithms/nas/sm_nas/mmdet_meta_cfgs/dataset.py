"""Modules for Dataset."""

import random

from ..utils import dict2str, str_warp
from .module import Module


class Dataset(Module):
    """Class of dataset."""

    attr_space = {
        'img_scale': [
            {'train': (512, 512), 'val': (512, 512), 'test': (512, 512)},
            {'train': (800, 600), 'val': (800, 600), 'test': (800, 600)},
            {'train': (1000, 600), 'val': (1000, 600), 'test': (1000, 600)}]}
    id_attrs = ['dataset', 'img_scale', 'multiscale_mode']

    def __init__(self,
                 dataset='CocoDataset',
                 batch_size=2,
                 num_workers=2,
                 multiscale_mode='range',
                 num_classes=81,
                 **kwargs):
        super(Dataset, self).__init__(**kwargs)
        data_root_default = '/cache/data/COCO2017'
        img_scale_default = dict(
            train=[(512, 512), (512, 300)],
            val=(512, 512),
            test=(512, 512))
        img_norm_cfg_default = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)

        self.dataset_type = dataset
        self.batch_size = batch_size
        self.workers = num_workers
        self.multiscale_mode = multiscale_mode
        self.num_classes = num_classes

        self.data_root = kwargs['data_root'] if 'data_root' in kwargs else data_root_default
        img_scale = kwargs['img_scale'] if 'img_scale' in kwargs else img_scale_default
        self.img_scale = dict()
        for k, v in img_scale.items():
            self.img_scale.update({k: tuple(v)})
        self.img_norm = kwargs['img_norm_cfg'] if 'img_norm_cfg' in kwargs else img_norm_cfg_default
        self.flip_ratio = kwargs['flip_ratio'] if 'flip_ratio' in kwargs else 0.5
        self.size_divisor = kwargs['size_divisor'] if 'size_divisor' in kwargs else 32

        self.data_setting = []
        for task in ['train', 'val', 'test']:
            img_scale_ = self.img_scale[task]
            setting = [img_scale_]
            if task == 'train':
                setting.append(str_warp(self.multiscale_mode))
            self.data_setting.extend(setting)

    def __str__(self):
        """Get image size(test) str."""
        return '{}({}x{})'.format(self.dataset_type.upper(),
                                  self.test_img_size[0], self.test_img_size[1])

    @property
    def test_img_size(self):
        """Get test image size."""
        return self.img_scale['test']

    @property
    def config(self):
        """Return dataset config for mmdet."""
        return dict(dataset_type=str_warp(self.dataset_type),
                    img_norm=self.img_norm_cfg,
                    train_pipeline=self.train_pipeline_cfg,
                    test_pipeline=self.test_pipeline_cfg,
                    data_setting=self.data_setting_cfg)

    @property
    def train_pipeline_cfg(self):
        """Generate train pipeline config."""
        img_scale = self.img_scale['train']
        data = (
            "[\n"
            "    dict(type='LoadImageFromFile'),\n"
            "    dict(type='LoadAnnotations', with_bbox=True),\n"
            f"    dict(type='Resize', img_scale=({img_scale[0]}, {img_scale[1]}), keep_ratio=True),\n"
            f"    dict(type='RandomFlip', flip_ratio={self.flip_ratio}),\n"
            "    dict(type='Normalize', **img_norm_cfg),\n"
            f"    dict(type='Pad', size_divisor={self.size_divisor}),\n"
            "    dict(type='DefaultFormatBundle'),\n"
            "    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),\n"
            "]")
        return data

    @property
    def test_pipeline_cfg(self):
        """Generate test pipeline config."""
        img_scale = self.img_scale['test']
        data = (
            "[\n"
            "    dict(type='LoadImageFromFile'),\n"
            "    dict(\n"
            "        type='MultiScaleFlipAug',\n"
            f"        img_scale=({img_scale[0]}, {img_scale[1]}),\n"
            "        flip=False,\n"
            "        transforms=[\n"
            "            dict(type='Resize', keep_ratio=True),\n"
            "            dict(type='RandomFlip'),\n"
            "            dict(type='Normalize', **img_norm_cfg),\n"
            f"            dict(type='Pad', size_divisor={self.size_divisor}),\n"
            "            dict(type='ImageToTensor', keys=['img']),\n"
            "            dict(type='Collect', keys=['img']),\n"
            "        ])\n"
            "]")
        return data

    @property
    def data_setting_cfg(self):
        """Return data setting."""
        data = (
            "dict(\n"
            f"    imgs_per_gpu={self.batch_size},\n"
            f"    workers_per_gpu={self.workers},\n"
            "    type=dataset_type,\n"
            "    train=dict(\n"
            "        type=dataset_type,\n"
            "        ann_file=data_root + 'annotations/instances_train2017.json',\n"
            "        img_prefix=data_root + 'train2017/',\n"
            "        pipeline=train_pipeline),\n"
            "    val=dict(\n"
            "        type=dataset_type,\n"
            "        ann_file=data_root + 'annotations/instances_val2017.json',\n"
            "        img_prefix=data_root + 'val2017/',\n"
            "        pipeline=test_pipeline),\n"
            "    test=dict(\n"
            "        type=dataset_type,\n"
            "        ann_file=data_root + 'annotations/instances_val2017.json',\n"
            "        img_prefix=data_root + 'val2017/',\n"
            "        pipeline=test_pipeline))")
        return data

    @property
    def img_norm_cfg(self):
        """Return image normalization config."""
        return dict2str(self.img_norm, in_one_line=True)

    @classmethod
    def sample(cls, data_setting, fore_part=None):
        """Return sampled_params."""
        sampled_params = dict()
        for key, value in cls.attr_space.items():
            attr = random.choice(value)
            sampled_params.update({key: attr})
        sampled_params.update(data_setting)
        sampled_params.update(cls.quest_param(fore_part=fore_part))
        return cls(**sampled_params, fore_part=fore_part)
