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
import tqdm
from DetectionIoUEvaluator import DetectionIoUEvaluator
from inference import DB
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


if __name__ == '__main__':
    ckpt_path = "./logs/ckpt/"
    gt_img_dir = cfg.EVAL.IMG_DIR
    gt_file_dir = cfg.EVAL.LABEL_DIR

    metrics = evaluate_all(gt_file_dir, gt_img_dir, ckpt_path, gpuid="1")
    # print(precision, recall, f1_score)
