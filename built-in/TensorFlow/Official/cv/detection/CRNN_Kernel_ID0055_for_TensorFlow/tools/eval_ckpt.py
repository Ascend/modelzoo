#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-8 下午10:24
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : recongnize_chinese_pdf.py
# @IDE: PyCharm
import os
import argparse


def init_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str,default='data/',
                        help='Directory containing test_features.tfrecords')
    parser.add_argument('-a', '--annotation_file', type=str,default='data/',
                        help='Directory containing test_features.tfrecords')
    parser.add_argument('-c', '--char_dict_path', type=str,default='data/char_dict/char_dict.json',
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,default='data/char_dict/ord_map.json',
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-w', '--weights_path', type=str, required=True,
                        help='Path to pre-trained weights')
    parser.add_argument('-i', '--device_id', type=str, default='0',
                        help='which npu device to use')
    parser.add_argument('-s', '--scripts', type=str, default='tools/evaluate_shadownet.py',
                        help='which script to run')
    parser.add_argument('-p', '--process_all', type=int, default=0,
                        help='Whether to process all test dataset')
    parser.add_argument('-r', '--root_dir', type=str,default='./',
                        help='root directory of the project')

    return parser.parse_args()



def main():
    args = init_args()
    ckpt_names = [ f for f in os.listdir(args.weights_path) if '.meta' in f ]
    ckpt_files = [ os.path.join(args.weights_path, ckpt.strip(".meta")) for ckpt in ckpt_names]

    device_id = 'DEVICE_ID=' + str(args.device_id)
    scripts = ' python3 '+ os.path.join(args.root_dir,args.scripts)
    data_dir = ' --dataset_dir='+ os.path.join(args.root_dir,args.dataset_dir)
    annotation_file = ' --annotation_file=' +args.annotation_file
    char_dict = ' --char_dict_path='+os.path.join(args.root_dir,args.char_dict_path)
    ord_map = ' --ord_map_dict_path='+os.path.join(args.root_dir,args.ord_map_dict_path)
    cmd_base = device_id + scripts + \
            annotation_file + \
            data_dir + char_dict + \
            ord_map + ' -p 1'

    for ckpt in ckpt_files:
        weight_path = ' --weights_path='+ckpt
        cmd = cmd_base + weight_path
        os.system(cmd)


if __name__=='__main__':
    main()