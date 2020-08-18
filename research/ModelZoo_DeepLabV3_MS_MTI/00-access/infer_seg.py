import sys
import os
import gflags
import logging
import numpy as np
import cv2
from PIL import Image
import pickle

from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="Davinci",   
    save_graphs=False)
context.set_context(device_id=int(os.getenv('DEVICE_ID')))
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.transforms.py_transforms as py_transforms
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import TrainOneStepCell
import mindspore.nn as nn
from mindspore.ops.operations import TensorAdd
from mindspore.ops import operations as P
from mindspore.common.initializer import XavierUniform, HeUniform
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import _InternalCallbackParam, RunContext
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from nets import net_factory


FLAGS = gflags.FLAGS

gflags.DEFINE_string('image_lst', '', 
    'input images list with absolute image path')

gflags.DEFINE_string('dst_dir', '', 'where results mask saved')

gflags.DEFINE_integer('crop_size', 513, 'crop size')

gflags.DEFINE_integer('ignore_label', 255, 'ignore label')

gflags.DEFINE_integer('num_classes', 21, 'number of classes')

gflags.DEFINE_multi_float('image_mean', [103.53, 116.28, 123.675], 'image means')

gflags.DEFINE_multi_float('image_std', [57.375, 57.120, 58.395], 'image stds')

gflags.DEFINE_string('model', 'deeplab_v3_s16', 'select model')

gflags.DEFINE_bool('freeze_bn', False, 'freeze bn')

gflags.DEFINE_string('ckpt_path', '', 'select trained model')

gflags.DEFINE_string('palette_path', '',
    'pickle that contains pallete')


def get_pallete(palette_path):
    with open(palette_path, 'rb') as f:
        p = pickle.load(f)
    return p


def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    bins_ = np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n**2)
    return bins_.reshape(n, n)


def resize_long(img, long_size=512):

    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


class BuildEvalNetwork(nn.Cell):
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
 
    def construct(self, input_data):
        output = self.network(input_data)
        return output


def infer():

    FLAGS(sys.argv)

    palette = get_pallete(FLAGS.palette_path)
    if not os.path.exists(FLAGS.dst_dir):
        os.makedirs(FLAGS.dst_dir)
    
    with open(FLAGS.image_lst) as f:
        img_lst = f.readlines()
    
    # load model
    # nets
    if FLAGS.model == 'deeplab_v3_s16':
        network = net_factory.nets_map[FLAGS.model]('eval', FLAGS.num_classes,
                                                    16, FLAGS.freeze_bn)
    elif FLAGS.model == 'deeplab_v3_s8':
        network = net_factory.nets_map[FLAGS.model]('eval', FLAGS.num_classes,
                                                    8, FLAGS.freeze_bn)
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(
            FLAGS.model))
    eval_net = BuildEvalNetwork(network)
    param_dict = load_checkpoint(FLAGS.ckpt_path)
    load_param_into_net(eval_net, param_dict)

    for l in img_lst:
        img_path = l.strip()
        print ('processing: ', img_path)
        img_ = cv2.imread(img_path)
        ori_h, ori_w, _ = img_.shape
        img_ = resize_long(img_, FLAGS.crop_size)
        resize_h, resize_w, _ = img_.shape
        # mean, std
        image_mean = np.array(FLAGS.image_mean)
        image_std = np.array(FLAGS.image_std)
        img_ = (img_ - image_mean) / image_std
        # pad to crop_size
        pad_h = FLAGS.crop_size - img_.shape[0]
        pad_w = FLAGS.crop_size - img_.shape[1]
        if pad_h > 0 or pad_w > 0 :
            img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, 
                cv2.BORDER_CONSTANT, value=0)
        img_ = img_.transpose((2,0,1))
        img_ = img_[np.newaxis,:,:,:].copy()
        net_out = eval_net(Tensor(img_, mstype.float32))
        map_res = net_out.asnumpy().argmax(axis=1).reshape(FLAGS.crop_size,
            FLAGS.crop_size)
        map_res = map_res[:resize_h, :resize_w]
        map_res = cv2.resize(map_res, (ori_w, ori_h), 
            interpolation=cv2.INTER_NEAREST)
        if not(FLAGS.dst_dir == ''):
            map_save = Image.fromarray(map_res.astype('uint8'))
            map_save.putpalette(palette)
            msk_name = img_path.split('/')[-1].split('.')[0] + '.png'
            map_save.save(os.path.join(FLAGS.dst_dir, msk_name))
        
if __name__ == '__main__':
    infer()