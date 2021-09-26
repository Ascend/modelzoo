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

from npu_bridge.npu_init import *
import cv2
from timers import *
import os
import codecs
import numpy as np

import tensorflow as tf
import model

from post_processing import decode

tf.app.flags.DEFINE_string('test_data_path', './test_data/', '')
tf.app.flags.DEFINE_string('gpu_list', '2', '')
tf.app.flags.DEFINE_string('checkpoint_path', './ckpts/', '')
tf.app.flags.DEFINE_string('output_dir', './result/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

FLAGS = tf.app.flags.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


def get_images():
    """
    find image files in test data path
    :return: list of files found
    """
    files = []
    exts = ['jpg', 'png', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=1024):  # 2400
    """
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape

    # limit the max side
    if max(h, w) > max_side_len:
        ratio = float(max_side_len) / h if h > w else float(max_side_len) / w
    else:
        ratio = 1.
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def restore_geo_quad_map(geo_map, xy_scale=4):
    # restore geo_map in parallel.
    d0, d1, d2, d3, theta = np.split(geo_map, 5, axis=-1)
    dir_x = np.concatenate((np.cos(theta), -np.sin(theta)), axis=-1)
    dir_y = np.concatenate((np.sin(theta), np.cos(theta)), axis=-1)
    p0_offset = - dir_x * d3 - dir_y * d0
    p1_offset = dir_x * d1 - dir_y * d0
    p2_offset = dir_x * d1 + dir_y * d2
    p3_offset = - dir_x * d3 + dir_y * d2
    geo_offset = np.concatenate((p0_offset, p1_offset, p2_offset, p3_offset), axis=-1)

    _, h, w, _ = geo_map.shape
    X, Y = np.meshgrid(range(w), range(h))
    originXY = np.stack((X, Y, X, Y, X, Y, X, Y), axis=2) * xy_scale
    geo_quad_map = originXY + geo_offset

    return geo_quad_map


def detectQuads(score_map, geo_rbox_map, score_map_thresh=0.8):
    """
    restore text boxes.
    """
    geo_quad_map = restore_geo_quad_map(geo_rbox_map)

    score_map = score_map[0, :, :, 0]
    geo_rbox_map = geo_rbox_map[0, :, :, :]
    geo_quad_map = geo_quad_map[0, :, :, :]
    quads = decode(score_map, geo_rbox_map, geo_quad_map)

    return quads


def saveDetResult(im_fn, boxes, no_write_images=FLAGS.no_write_images):
    # save boxes into file.
    dir_name = os.path.basename(os.path.dirname(im_fn))
    im_name = os.path.basename(im_fn)
    out_dir = os.path.join(FLAGS.output_dir, dir_name)
    if not os.path.exists(out_dir):
        os.mkdir(outdir)
    txt_name = os.path.join(out_dir, im_name[:-4] + '.txt')
    if boxes is not None:
        with codecs.open(txt_name, 'w', 'utf-8') as f:
            for box in boxes:
                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                    box[0, 0], box[0, 1], box[1, 0], box[1, 1],
                    box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
                cv2.polylines(im, [box.astype(np.int32).reshape((-1, 1, 2))],
                              True, color=(255, 0, 0), thickness=1)
    if not no_write_images:
        img_path = os.path.join(FLAGS.output_dir, dir_name, im_name)
        cv2.imwrite(img_path, im[:, :, ::-1])


def main(argv=None):
    # make output directory.
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            # get image list.
            im_fn_list = get_images()

            for im_fn in im_fn_list:
                try:
                    im = cv2.imread(im_fn)
                    if im is None: continue
                    im = im[:, :, ::-1]  # RGB
                    im_resized, (ratio_h, ratio_w) = resize_image(im)
                except:
                    continue
                TM_BEGIN('duration')
                TM_BEGIN('net')
                score, geo_rbox = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                TM_END('net')
                TM_BEGIN('nms')
                quads = detectQuads(score, geo_rbox)
                TM_END('nms')
                print('{}: net {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, TM_PICK('net') * 1000, TM_PICK('nms') * 1000))

                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                # print( boxes)
                TM_END('duration')
                print('[timing] {}'.format(TM_PICK('duration')))
                # save detection result.
                saveDetResult(im_fn, boxes)
            TM_DISPLAY()


if __name__ == '__main__':
    tf.app.run()
