#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-25 下午3:56
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : evaluate_shadownet.py
# @IDE: PyCharm Community Edition
"""
Evaluate the crnn model on the synth90k test dataset
"""
import argparse
import os.path as ops
import os
import math
import time
import sys
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import glog as log
import tqdm
from sklearn.metrics import confusion_matrix
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from PIL import Image 

cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)


from crnn_model import crnn_net
from config import global_config
from data_provider import shadownet_data_feed_pipline
from data_provider import tf_io_pipline_fast_tools
from local_utils import evaluation_tools


CFG = global_config.cfg


def init_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str,default='data/',
                        help='Directory containing test_features.tfrecords')
    parser.add_argument('-c', '--char_dict_path', type=str,default='data/char_dict_bak/char_dict.json',
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,default='data/char_dict_bak/ord_map.json',
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-w', '--weights_path', type=str, required=True,
                        help='Path to pre-trained weights')
    parser.add_argument('-a', '--annotation_file', type=str, required=True,
                        help='Path to annotation file')
    parser.add_argument('-v', '--visualize', type=args_str2bool, nargs='?', const=False,
                        help='Whether to display images')
    parser.add_argument('-p', '--process_all', type=args_str2bool, nargs='?', const=False,
                        help='Whether to process all test dataset')

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


def evaluate_shadownet(dataset_dir, weights_path, char_dict_path,
                       ord_map_dict_path,annotation_file,
                       is_visualize=False,
                       is_process_all_data=False):
    """

    :param dataset_dir:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param is_visualize:
    :param is_process_all_data:
    :return:
    """
    
    batchsize = 64
    test_images = tf.placeholder(tf.float32, shape=[batchsize, 32, 100, 3],name="test_images")
     
    # declare crnn net
    shadownet = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )
    # set up decoder
    decoder = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )

    # compute inference result
    test_inference_ret = shadownet.inference(
        inputdata=test_images,
        name='shadow_net',
        reuse=False
    )
    test_decoded, test_log_prob = tf.nn.ctc_greedy_decoder(
        test_inference_ret,
        CFG.ARCH.SEQ_LENGTH * np.ones(batchsize),
        merge_repeated=True
    )

    # Set saver configuration
    saver = tf.train.Saver()
    
    # NPU CONFIG
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('allow_mix_precision')
    custom_op.parameter_map["mix_compile_mode"].b = False  # 混合计算
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    sess = tf.Session(config=config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        log.info('Start predicting...')

        per_char_accuracy = 0
        full_sequence_accuracy = 0.0

        total_labels_char_list = []
        total_predictions_char_list = []
        

        annotation_list = get_annotation(annotation_file)
        num_iterations = len(annotation_list)//batchsize
        epoch_tqdm = tqdm.tqdm(range(num_iterations))
        for i in epoch_tqdm:
        #for i in range(num_iterations):
            anns = annotation_list[i*batchsize:(i+1)*batchsize]
            batch_data, batch_label = get_batch_data(dataset_dir, anns)
            test_predictions_value = sess.run(test_decoded,feed_dict={test_images: batch_data}) 
            test_predictions_value = decoder.sparse_tensor_to_str(test_predictions_value[0])
            

            per_char_accuracy += evaluation_tools.compute_accuracy(
                        batch_label, test_predictions_value, display=False, mode='per_char'
                    )

            full_sequence_accuracy += evaluation_tools.compute_accuracy(
                        batch_label, test_predictions_value, display=False, mode='full_sequence'
                    )
            for index, ann in enumerate(anns):
                log.info(ann)
                log.info("predicted values :{}".format(test_predictions_value[index]))
        


        epoch_tqdm.close()
        avg_per_char_accuracy = per_char_accuracy / num_iterations
        avg_full_sequence_accuracy = full_sequence_accuracy / num_iterations
        log.info('Mean test per char accuracy is {:5f}'.format(avg_per_char_accuracy))
        log.info('Mean test full sequence accuracy is {:5f}'.format(avg_full_sequence_accuracy))
        print('Mean test per char accuracy is {:5f}'.format(avg_per_char_accuracy))
        print('Mean test full sequence accuracy is {:5f}'.format(avg_full_sequence_accuracy))



if __name__ == '__main__':
    """
    test code
    """
    args = init_args()
    
    print('checkpoint {}'.format(args.weights_path))
    evaluate_shadownet(
        dataset_dir=args.dataset_dir,
        weights_path=args.weights_path,
        char_dict_path=args.char_dict_path,
        ord_map_dict_path=args.ord_map_dict_path,
        annotation_file=args.annotation_file,
        is_visualize=args.visualize,
        is_process_all_data=args.process_all
    )
