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
import argparse
import time

import tensorflow as tf
import numpy as np
from tqdm import trange

from utils.config import Config
from utils.image_reader import ImageReader
from model import ICNet, ICNet_BN

# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced ICNet")

    parser.add_argument("--model", type=str, default='',
                        help="Model to use.",
                        choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others'],
                        required=True)
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes'],
                        required=True)
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])

    return parser.parse_args()

def main():
    args = get_arguments()  
    cfg = Config(dataset=args.dataset, is_training=False, filter_scale=args.filter_scale)
    
    model = model_config[args.model]

    reader = ImageReader(cfg=cfg, mode='eval')
    net = model(image_reader=reader, cfg=cfg, mode='eval')
    
    # mIoU
    pred_flatten = tf.reshape(net.output, [-1,])
    label_flatten = tf.reshape(net.labels, [-1,])

    mask = tf.not_equal(label_flatten, cfg.param['ignore_label'])
    indices = tf.squeeze(tf.where(mask), 1)
    gt = tf.cast(tf.gather(label_flatten, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    if cfg.dataset == 'ade20k':
        pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes']+1)
    elif cfg.dataset == 'cityscapes':
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])
    
    net.create_session()
    net.restore(cfg.model_paths[args.model])
    
    for i in trange(cfg.param['eval_steps'], desc='evaluation', leave=True):
        _ = net.sess.run(update_op)
             
    print('mIoU: {}'.format(net.sess.run(mIoU)))
    
if __name__ == '__main__':
    main()
