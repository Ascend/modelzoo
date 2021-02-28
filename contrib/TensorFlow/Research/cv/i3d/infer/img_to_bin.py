from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import logging

import numpy as np
import tensorflow as tf

import i3d_new_bn as i3d
from lib.dataset import ActionDataset
from lib.load_data import load_info
from lib.feed_queue import FeedQueue
from lib.label_trans import *

_FRAME_SIZE = 224
_QUEUE_SIZE = 20
_QUEUE_PROCESS_NUM = 5
_MIX_WEIGHT_OF_RGB = 0.5
_MIX_WEIGHT_OF_FLOW = 0.5
_LOG_ROOT = 'output'

# NOTE: Before running, change the path of data
_DATA_ROOT = {
    'ucf101': {
        'rgb': '../data/jpegs_256',
        'flow': '../data/tvl1_flow//{:s}'
    },
    'hmdb51': {
        'rgb': '../data/hmdb51/jpegs_256',
        'flow': '../data/tvl1_flow/{:s}'
    }
}

# NOTE: Before running, change the path of checkpoints
_CHECKPOINT_PATHS = {
    'rgb': '../model/ucf101_rgb_0.946_model-44520',
    'flow': '../model/ucf101_flow_0.963_model-28620',
}

_CHANNEL = {
    'rgb': 3,
    'flow': 2,
}

_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}

_CLASS_NUM = {
    'ucf101': 101,
    'hmdb51': 51
}


def main(dataset, mode, split):
    assert mode in ['rgb', 'flow', 'mixed']
    log_dir = os.path.join(_LOG_ROOT, 'test-%s-%s-%d' % (dataset, mode, split))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(level=logging.INFO, filename=os.path.join(
        log_dir, 'log-%s-%d' % (mode, split)+'.txt'), filemode='w', format='%(message)s')

    label_map = get_label_map(os.path.join(
        '../data', dataset, 'label_map.txt'))
    print('_DATA_ROOT[dataset]:', _DATA_ROOT[dataset])

    _, test_info_rgb, class_num, _ = load_info(
        dataset, root=_DATA_ROOT[dataset], mode='rgb', split=split)
    print('test_info_rgb:', test_info_rgb)
    _, test_info_flow, _, _ = load_info(
        dataset, root=_DATA_ROOT[dataset], mode='flow', split=split)

    label_holder = tf.placeholder(tf.int32, [None])
    if mode in ['rgb', 'mixed']:
        rgb_data = ActionDataset(
            dataset, class_num, test_info_rgb, 'frame{:06d}{:s}.jpg', mode='rgb')
        print('rgb_data:', rgb_data)
        rgb_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['rgb']])
        info_rgb, _ = rgb_data.gen_test_list()  #[([0, 251, 1, 1, False, False],), ([1, 251, 1, 1, False, False],)]
#         print('info_rgb:', info_rgb)
    if mode in ['flow', 'mixed']:
        flow_data = ActionDataset(
            dataset, class_num, test_info_flow, 'frame{:06d}{:s}.jpg', mode='flow')
        flow_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['flow']])
        info_flow, _ = flow_data.gen_test_list()


    if mode in ['rgb', 'mixed']:
        # Start Queue
        rgb_queue = FeedQueue(queue_size=_QUEUE_SIZE)
        rgb_queue.start_queue(rgb_data.get_video, args=info_rgb,
                              process_num=_QUEUE_PROCESS_NUM)
    if mode in ['flow', 'mixed']:
        flow_queue = FeedQueue(queue_size=_QUEUE_SIZE)
        flow_queue.start_queue(flow_data.get_video,
                               args=info_flow, process_num=_QUEUE_PROCESS_NUM)

    # Here we start the test procedure
    print('----Here we start!----')
    print('Output wirtes to '+ log_dir)
    true_count = 0
    video_size = len(test_info_rgb)
    print('video_size:', video_size)
#     error_record = open(os.path.join(
#         log_dir, 'error_record_'+mode+'.txt'), 'w')
#     rgb_fc_data = np.zeros((video_size, _CLASS_NUM[dataset]))
#     flow_fc_data = np.zeros((video_size, _CLASS_NUM[dataset]))
#     label_data = np.zeros((video_size, 1))

    # just load 1 video for test,this place needs to be improved
    for i in range(video_size):
        if mode in ['rgb', 'mixed']:
            rgb_clip, label = rgb_queue.feed_me()
            # print('rgb_clip, label:', rgb_clip, label)
            rgb_clip = rgb_clip/255
            #input_rgb = rgb_clip[np.newaxis, :, :, :, :]
            input_rgb = rgb_clip[np.newaxis, :, :, :, :]
            print('input_rgb:', input_rgb.shape)
            video_name = rgb_data.videos[i].name
            print('video_name:', video_name)
            input_rgb.tofile('../data/rgb/' + video_name + '.bin')
        if mode in ['flow', 'mixed']:
            flow_clip, label = flow_queue.feed_me()
            flow_clip = 2*(flow_clip/255)-1
            input_flow = flow_clip[np.newaxis, :, :, :, :]
            video_name = flow_data.videos[i].name
            input_flow.tofile('../data/flow/' + video_name + '.bin')

        input_label = np.array([label]).reshape(-1)
        print('input_label.shape:', input_label.shape)
#        print('input_rgb.shape:', input_rgb.shape)
#        print('input_flow.shape:', input_flow.shape)
        if mode in ['flow', 'mixed']:
            video_name = flow_data.videos[i].name
            input_label.tofile('../data/label/' + video_name + '.bin')
        if mode in ['rgb', 'mixed']:
            video_name = rgb_data.videos[i].name
            input_label.tofile('../data/label/' + video_name + '.bin')


if __name__ == '__main__':
    description = 'Test Finetuned I3D Model'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset', type=str, help="name of dataset, e.g., ucf101")
    p.add_argument('mode', type=str, help="type of data, e.g., rgb")
    p.add_argument('split', type=int, help="split of data, e.g., 1")
    main(**vars(p.parse_args()))
