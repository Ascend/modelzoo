import os
import cv2
import time
import tqdm
import argparse
import numpy as np
import tensorflow as tf

from config.db_config import cfg
from shapely.geometry import Polygon
from postprocess.post_process import SegDetectorRepresenter
import networks.model as model
import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

def get_args():
    parser = argparse.ArgumentParser(description='DB-tf')
    parser.add_argument('--ckptpath', default='./logs3/ckpt/DB_resnet_v1_50_adam_model.ckpt-38381',
                        type=str,
                        help='load model')
    parser.add_argument('--imgpath',
                        default='./datasets/total_text/test_images/img1.jpg',
                        type=str)
    parser.add_argument('--gpuid', default='0',
                        type=str)
    parser.add_argument('--ispoly', default=True,
                        type=bool)
    parser.add_argument('--show_res', default=True,
                        type=bool)

    args = parser.parse_args()

    return args


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

from tensorflow.python.compat import compat

class DB():

    def __init__(self, ckpt_path):

        tf.reset_default_graph()
        self._input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)


        with compat.forward_compatibility_horizon(2019, 5, 1):
            self._binarize_map, self._threshold_map, self._thresh_binary = model.dbnet(self._input_images,is_training=False)

        # variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        # saver = tf.train.Saver(variable_averages.variables_to_restore())
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
        # self.sess = tf.Session(config=gpu_config)

        variable_averages = tf.train.ExponentialMovingAverage(0.997)
        saver = tf.train.Saver(tf.global_variables())
        
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        
        self.sess = tf.Session(config=config)
        saver.restore(self.sess, ckpt_path)
        self.decoder = SegDetectorRepresenter()
        print('restore model from:', ckpt_path)

    def __del__(self):
        self.sess.close()

    def detect_img(self, img_path, ispoly=True, show_res=True):
        img = cv2.imread(img_path)
        img_copy = img.copy()
        mean = np.array([122.67891434, 116.66876762, 104.00698793])
        img = img.astype(np.float32)
        img -= mean
        img = img / 255.0
        h, w, _ = img.shape
        resized_img, ratio, size = self._resize_img(img, max_size=800)
        print(ratio)

        s = time.time()
        binarize_map, threshold_map, thresh_binary = self.sess.run(
            [self._binarize_map, self._threshold_map, self._thresh_binary],
            feed_dict={self._input_images: [resized_img]})
        net_time = time.time() - s

        s = time.time()
        boxes, scores = self.decoder([resized_img], binarize_map, ispoly)
        boxes = boxes[0]
        area = h * w
        res_boxes = []
        res_scores = []
        for i, box in enumerate(boxes):
            box[:, 0] *= ratio[1]
            box[:, 1] *= ratio[0]
            if Polygon(box).convex_hull.area > cfg.FILTER_MIN_AREA * area:
                res_boxes.append(box)
                res_scores.append(scores[0][i])
        post_time = time.time() - s

        if show_res:
            img_name = os.path.splitext(os.path.split(img_path)[-1])[0]
            make_dir('./show')
            cv2.imwrite('show/' + img_name + '_binarize_map.jpg', binarize_map[0][0:size[0], 0:size[1], :] * 255)
            cv2.imwrite('show/' + img_name + '_threshold_map.jpg', threshold_map[0][0:size[0], 0:size[1], :] * 255)
            cv2.imwrite('show/' + img_name + '_thresh_binary.jpg', thresh_binary[0][0:size[0], 0:size[1], :] * 255)
            for box in res_boxes:
                cv2.polylines(img_copy, [box.astype(np.int).reshape([-1, 1, 2])], True, (0, 255, 0))
                # print(Polygon(box).convex_hull.area, Polygon(box).convex_hull.area/area)
            cv2.imwrite('show/' + img_name + '_show.jpg', img_copy)

        return res_boxes, res_scores, (net_time, post_time)

    def detect_batch(self, batch):
        pass

    def _resize_img(self, img, max_size=640):
        h, w, _ = img.shape

        ratio = float(max(h, w)) / max_size

        new_h = int((h / ratio // 32) * 32)
        new_w = int((w / ratio // 32) * 32)

        resized_img = cv2.resize(img, dsize=(new_w, new_h))

        input_img = np.zeros([max_size, max_size, 3])
        input_img[0:new_h, 0:new_w, :] = resized_img

        ratio_w = w / new_w
        ratio_h = h / new_h

        return input_img, (ratio_h, ratio_w), (new_h, new_w)


if __name__ == "__main__":
    args = get_args()

    db = DB(args.ckptpath, args.gpuid)

    db.detect_img(args.imgpath, args.ispoly, args.show_res)

    img_list = os.listdir('./datasets/total_text/train_images/')

    net_all = 0
    post_all = 0
    pipe_all = 0

    for i in tqdm.tqdm(img_list[10:]):
        _, _, (net_time, post_time) = db.detect_img(
            os.path.join('./datasets/total_text/train_images/', i), args.ispoly,
            show_res=True)
        net_all += net_time
        post_all += post_time
        pipe_all += (net_time + post_time)

    print('net:', net_all / len(img_list))
    print('post:', post_all / len(img_list))
    print('pipe:', pipe_all / len(img_list))
