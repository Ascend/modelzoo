# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""util for fasterrcnn"""
import os
import numpy as np
from .dataset import data_to_mindrecord_byte_image

_init_value = np.array(0.0)
summary_init = {
    'Precision/mAP': _init_value,
    'Precision/mAP@.50IOU': _init_value,
    'Precision/mAP@.75IOU': _init_value,
    'Precision/mAP (small)': _init_value,
    'Precision/mAP (medium)': _init_value,
    'Precision/mAP (large)': _init_value,
    'Recall/AR@1': _init_value,
    'Recall/AR@10': _init_value,
    'Recall/AR@100': _init_value,
    'Recall/AR@100 (small)': _init_value,
    'Recall/AR@100 (medium)': _init_value,
    'Recall/AR@100 (large)': _init_value,
}


def bbox2result_1image(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        result = [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)]
    else:
        result = [bboxes[labels == i, :] for i in range(num_classes - 1)]
    return result


def create_mindrecord(mindrecord_dir, prefix, args_opt, config, phase="train"):
    if not os.path.isdir(mindrecord_dir):
        os.makedirs(mindrecord_dir)
    if args_opt.dataset == "coco":
        if os.path.isdir(config.dataset_root):
            print("Create Mindrecord.")
            data_to_mindrecord_byte_image("coco", True, prefix) if phase == "train" else \
                data_to_mindrecord_byte_image("coco", False, prefix, file_num=1)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            print("coco_root not exits.")
    else:
        pass


def eval_trial(outputs, ann_file, eval_types, args_opt):
    if args_opt.dataset == "coco":
        from .datasets.coco.coco_util import coco_eval
        coco_eval(outputs, ann_file, eval_types, single_result=True)
    elif args_opt.dataset == "other":
        pass
