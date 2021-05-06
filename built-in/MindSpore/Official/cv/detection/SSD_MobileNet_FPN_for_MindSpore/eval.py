# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Evaluation for SSD"""

import os
import argparse
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.ssd import SSD300, SsdInferWithDecoder, ssd_mobilenet_v2, ssd_mobilenet_v1_fpn
from src.dataset import create_ssd_dataset, create_mindrecord
from src.config import config
from src.eval_utils import apply_eval
from src.box_utils import default_boxes

def ssd_eval(dataset_path, ckpt_path, anno_json):
    """SSD evaluation."""
    batch_size = 1
    ds = create_ssd_dataset(dataset_path, batch_size=batch_size, repeat_num=1,
                            is_training=False, use_multiprocessing=False)
    if config.model == "ssd300":
        net = SSD300(ssd_mobilenet_v2(), config, is_training=False)
    else:
        net = ssd_mobilenet_v1_fpn(config=config)
    net = SsdInferWithDecoder(net, Tensor(default_boxes), config)

    print("Load Checkpoint!")
    param_dict = load_checkpoint(ckpt_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)

    net.set_train(False)
    total = ds.get_dataset_size() * batch_size
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    eval_param_dict = {"net": net, "dataset": ds, "anno_json": anno_json}
    mAP = apply_eval(eval_param_dict)

    print("\n========================================\n")
    print(f"mAP: {mAP}")

def get_eval_args():
    parser = argparse.ArgumentParser(description='SSD evaluation')
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint file path.")
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"),
                        help="run platform, support Ascend ,GPU and CPU.")
    return parser.parse_args()

if __name__ == '__main__':
    args_opt = get_eval_args()
    if args_opt.dataset == "coco":
        json_path = os.path.join(config.coco_root, config.instances_set.format(config.val_data_type))
    elif args_opt.dataset == "voc":
        json_path = os.path.join(config.voc_root, config.voc_json)
    else:
        raise ValueError('SSD eval only supprt dataset mode is coco and voc!')

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.run_platform, device_id=args_opt.device_id)

    mindrecord_file = create_mindrecord(args_opt.dataset, "ssd_eval.mindrecord", False)

    print("Start Eval!")
    ssd_eval(mindrecord_file, args_opt.checkpoint_path, json_path)
