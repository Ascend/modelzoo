# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import tqdm
import pathlib
import argparse
import tensorflow as tf

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent.parent))

from eval.DetectionIoUEvaluator import DetectionIoUEvaluator
from eval.inference import DB
from config.db_config import cfg
from postprocess.utils import load_each_image_lable



def evaluate_all(gt_file_dir, gt_img_dir, ckpt_path, gpuid='2'):
    db = DB(ckpt_path)
    img_list = os.listdir(gt_img_dir)
    results = []
    evaluator = DetectionIoUEvaluator()
    for img_name in tqdm.tqdm(img_list):
        pred_box_list, pred_score_list, _ = db.detect_img(os.path.join(gt_img_dir, img_name),
                                                          ispoly=True,
                                                          show_res=True)
        pre = []
        for p in pred_box_list:
            pre.append({
                'points': [tuple(e) for e in p],
                'text': 123,
                'ignore': False,
            })
        gt_file_name = os.path.splitext(img_name)[0] + '.jpg.txt'
        label_info = load_each_image_lable(os.path.join(gt_file_dir, gt_file_name))
        results.append(evaluator.evaluate_image(label_info, pre))

    metrics = evaluator.combine_results(results)
    print(metrics)
    return metrics


def evalution_loop(ckpt_path):
    
    output = {}
    ckpt = tf.train.get_checkpoint_state(ckpt_path)

    for cp in ckpt.all_model_checkpoint_paths:
        gt_img_dir = cfg.EVAL.IMG_DIR
        gt_file_dir = cfg.EVAL.LABEL_DIR
        metric = evaluate_all(gt_file_dir, gt_img_dir, cp)
        output[cp.split('/')[-1]]=metric

    return output


def evalution_single_case(ckpt_path):
    gt_img_dir = cfg.EVAL.IMG_DIR
    gt_file_dir = cfg.EVAL.LABEL_DIR
    metric = evaluate_all(gt_file_dir, gt_img_dir, ckpt_path)
    return metric

def evaluate(gt_file_dir, gt_img_dir, ckpt_path, gpuid='2'):
    db = DB(ckpt_path)
    img_list = os.listdir(gt_img_dir)
    results = []
    evaluator = DetectionIoUEvaluator()
    for img_name in img_list:
        if(img_name=='img1.jpg'):
            db.detect_img2(os.path.join(gt_img_dir, img_name),ispoly=True,show_res=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DB-tf')
    parser.add_argument('--ckptpath', default='./logs3/ckpt/DB_resnet_v1_50_adam_model.ckpt-38381',
                        type=str,
                        help='load model')
    args = parser.parse_args()

    gt_img_dir = cfg.EVAL.IMG_DIR
    gt_file_dir = cfg.EVAL.LABEL_DIR

    evalution_single_case(args.ckptpath)