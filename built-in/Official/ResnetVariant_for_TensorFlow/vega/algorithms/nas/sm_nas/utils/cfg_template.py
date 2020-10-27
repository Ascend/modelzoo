"""Generate config file for mmdetection."""

from .str2dict import str2dict, str_warp


class CfgGenerator(object):
    """Config generator."""

    def __init__(self,
                 detector=None,
                 num_stages=None,
                 pretrained=None,
                 backbone=None,
                 neck=None,
                 rpn_head=None,
                 shared_head=None,
                 roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 img_norm=None,
                 train_pipeline=None,
                 test_pipeline=None,
                 dataset_type=None,
                 data_root=None,
                 data_setting=None,
                 optimizer=None,
                 lr_config=None,
                 epoch=None,
                 load_from=None
                 ):
        self.detector = detector
        self.num_stages = num_stages
        self.pretrained = pretrained
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.shared_head = shared_head
        self.roi_extractor = roi_extractor
        self.bbox_head = bbox_head
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.img_norm = img_norm
        self.train_pipeline = train_pipeline
        self.test_pipeline = test_pipeline
        self.dataset_type = dataset_type
        self.data_root = data_root
        self.data_setting = data_setting
        self.optimizer = optimizer
        self.lr_config = lr_config
        self.epoch = epoch
        self.load_from = load_from

    @staticmethod
    def cut_none_part(cfg):
        """Cut part of none."""
        parts = ['num_stages', 'backbone', 'neck', 'rpn_head', 'bbox_roi_extractor', 'shared_head',
                 'bbox_head']
        for part in parts:
            str_ = '{}=None,\n'.format(
                part) if part != 'bbox_head' else '{}=None'.format(part)
            if str_ in cfg:
                index = cfg.find(str_)
                cfg = cfg[:index - 4] + cfg[index + len(str_):]
        return cfg

    @staticmethod
    def get_attr_from_dictstr(str, attr):
        """Get attribute from dict str."""
        try:
            dict_ = str2dict(str)
            return dict_[attr]
        except BaseException:
            return None

    @property
    def config(self):
        """Get config string."""
        cfg = (
            "# model settings\n"
            "model = dict(\n"
            f"    type={self.detector},\n"
            f"    num_stages={self.num_stages},\n"
            f"    pretrained='{self.pretrained}',\n"
            f"    backbone={self.backbone},\n"
            f"    neck={self.neck},\n"
            f"    rpn_head={self.rpn_head},\n"
            f"    bbox_roi_extractor={self.roi_extractor},\n"
            f"    shared_head={self.shared_head},\n"
            f"    bbox_head={self.bbox_head})\n"
            "# model training and testing settings\n"
            f"train_cfg = {self.train_cfg}\n"
            f"test_cfg = {self.test_cfg}\n"
            "    # soft-nms is also supported for rcnn testing\n"
            "    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)\n"
            "# dataset settings\n"
            f"dataset_type = {self.dataset_type}\n"
            f"data_root = '{self.data_root}'\n"
            f"img_norm_cfg = {self.img_norm}\n"
            f"train_pipeline = {self.train_pipeline}\n"
            f"test_pipeline = {self.test_pipeline}\n"
            f"data = {self.data_setting}\n"
            "# optimizer\n"
            f"optimizer = {self.optimizer}\n"
            "optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))\n"
            "# learning policy\n"
            f"lr_config = {self.lr_config}\n"
            "checkpoint_config = dict(interval=1)\n"
            "# yapf:disable\n"
            "log_config = dict(\n"
            "    interval=50,\n"
            "    hooks=[\n"
            "        dict(type='TextLoggerHook'),\n"
            "        # dict(type='TensorboardLoggerHook')\n"
            "    ])\n"
            "# yapf:enable\n"
            "# runtime settings\n"
            f"total_epochs = {self.epoch}\n"
            "# val_interval=total_epochs\n"
            "dist_params = dict(backend='nccl')\n"
            "log_level = 'INFO'\n"
            "work_dir = None\n"
            f"load_from = {self.load_from}\n"
            "resume_from = None\n"
            "workflow = [('train', 1)]\n")
        return self.cut_none_part(cfg)
