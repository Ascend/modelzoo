# Copyright 2021 Huawei Technologies Co., Ltd
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
# =============================================================================
"""
Preprocess of test dataset
- IIIT5K
- ICDAR03
- SVT
"""
import argparse
import os.path as ops
import os
import math
import time
import sys
import tensorflow as tf
import numpy as np
import glog as log
import tqdm
from PIL import Image

cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)

def init_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str,default='./data/test/svt1/processed',
                        help='Directory of the test dataset')
    parser.add_argument('-o', '--output_bin_dir', type=str,default='./img_bin',
                        help='Directory where preprocessed images were stored')
    parser.add_argument('-l', '--output_label_dir', type=str, default='./labels',
                        help='Directory where labels were stored')
    parser.add_argument('-a', '--annotation_file', type=str, default='./data/test/svt1/annotation.txt',
                        help='Path of annotation file')
    parser.add_argument('-b', '--batchsize', type=int, default=64,
                        help='batchsize of input dataset')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_batch_data(data_dir,annotation):
    ''' 
    :params  data_dir: directory of images
    :params  annotation: a list of pairs (image_name, labels)

    :return imgs: resized image data, shape:(batchsize, 32, 100, 3)
    :return labels: labels for each images
    '''
    
    imgs = []
    labels = [] 
    
    for index, ann in enumerate(annotation):
        img_name,label = ann.split(",")[0],ann.split(",")[1]
        label = label.strip()
        labels.append(label.lower())
        img_path = os.path.join(data_dir, img_name)
        img = Image.open(img_path)
        if ".png" in img_name:
            img = img.convert('RGB')

        img = img.resize((100,32),Image.BILINEAR)
        img = np.array(img).astype(np.float32)
        img = (img-127.5)/255
        img_shape = img.shape

        if len(img_shape)==2:
           img = img.reshape([32,100,1])
           img = np.concatenate([img,img,img],axis=2)
        
        imgs.append(img)
    return imgs, labels


def get_annotation(annotation_file):
    ann_file = open(annotation_file,'r')
    annotation_list = [line.strip("\n") for line in ann_file.readlines()]
    ann_file.close()
    return annotation_list


def preprocess_shadownet(dataset_dir, img_dir, label_dir, annotation_file, batchsize=64):
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    sess = tf.Session()

    with sess.as_default():
        log.info('Start preprocessing...')

        annotation_list = get_annotation(annotation_file)
        num_iterations = len(annotation_list)//batchsize
        epoch_tqdm = tqdm.tqdm(range(num_iterations))
        for i in epoch_tqdm:
            anns = annotation_list[i*batchsize:(i+1)*batchsize]
            batch_data, batch_label = get_batch_data(dataset_dir, anns)
            (np.array(batch_data)).tofile(os.path.join(img_dir,'batch_data_'+str(i).rjust(3,'0')+'.bin'))
            with open(os.path.join(label_dir,'batch_label_'+str(i).rjust(3,'0')+'.txt'), 'w') as f:
                f.write(','.join(batch_label))

        epoch_tqdm.close()

if __name__ == '__main__':
    args = init_args()

    preprocess_shadownet(
        dataset_dir=args.dataset_dir,
        img_dir=args.output_bin_dir,
        label_dir=args.output_label_dir,
        annotation_file=args.annotation_file,
        batchsize=args.batchsize
    )
