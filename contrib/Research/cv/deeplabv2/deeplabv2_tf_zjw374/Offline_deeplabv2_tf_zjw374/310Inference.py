from datetime import datetime
import os
import sys
import time
import numpy as np
import tensorflow as tf
#from PIL import Image
"""
from network import *
from utils import ImageReader, decode_labels, inv_preprocess, prepare_label, write_log, read_labeled_image_list
"""


# def test_setup(self):
#     # Create queue coordinator.
#     self.coord = tf.train.Coordinator()
#
#     # Input size
#     input_size = (self.conf.input_height, self.conf.input_width)
#
#     # Load reader
#     with tf.name_scope("create_inputs"):
#         reader = ImageReader(
#             self.conf.data_dir,
#             self.conf.valid_data_list,
#             None,  # the images have different sizes
#             False,  # no data-aug
#             False,  # no data-aug
#             self.conf.ignore_label,
#             IMG_MEAN,
#             self.coord)
#         image, label = reader.image, reader.label  # [h, w, 3 or 1]
#     # Add one batch dimension [1, h, w, 3 or 1]
#     self.image_batch, self.label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)
#
#     # Create network
#     if self.conf.encoder_name not in ['res101', 'res50', 'deeplab']:
#         print('encoder_name ERROR!')
#         print("Please input: res101, res50, or deeplab")
#         sys.exit(-1)
#     elif self.conf.encoder_name == 'deeplab':
#         net = Deeplab_v2(self.image_batch, self.conf.num_classes, False)
#     else:
#         net = ResNet_segmentation(self.image_batch, self.conf.num_classes, False, self.conf.encoder_name)
#
#     # predictions
#     raw_output = net.outputs
#     raw_output = tf.image.resize_bilinear(raw_output, tf.shape(self.image_batch)[1:3, ])
#     raw_output = tf.argmax(raw_output, axis=3)
#     pred = tf.expand_dims(raw_output, dim=3)
#     self.pred = tf.reshape(pred, [-1, ])
#     # labels
#     gt = tf.reshape(self.label_batch, [-1, ])
#     # Ignoring all labels greater than or equal to n_classes.
#     temp = tf.less_equal(gt, self.conf.num_classes - 1)
#     weights = tf.cast(temp, tf.int32)
#
#     # fix for tf 1.3.0
#     gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))
#
#     # Pixel accuracy
#     self.accu, self.accu_update_op = tf.contrib.metrics.streaming_accuracy(
#         self.pred, gt, weights=weights)
#
#     # mIoU
#     self.mIoU, self.mIou_update_op = tf.contrib.metrics.streaming_mean_iou(
#         self.pred, gt, num_classes=self.conf.num_classes, weights=weights)
#
#     # confusion matrix
#     self.confusion_matrix = tf.contrib.metrics.confusion_matrix(
#         self.pred, gt, num_classes=self.conf.num_classes, weights=weights)
#
#     # Loader for loading the checkpoint
#     self.loader = tf.train.Saver(var_list=tf.global_variables())

num_classes = 21
valid_num_steps = 1447

def compute_IoU_per_class(confusion_matrix):
    mIoU = 0
    for i in range(num_classes):
        # IoU = true_positive / (true_positive + false_positive + false_negative)
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i]) - TP
        IoU = TP / (TP + FP + FN)
        print('class %d: %.3f' % (i, IoU))
        mIoU += IoU / num_classes
    print('mIoU: %.3f' % mIoU)

def graph():
    #tf.reset_default_graph()

    pred_ = tf.placeholder(tf.int64, shape = None)
    gt_ = tf.placeholder(tf.int64, shape = None)
    print(pred_.name)
    print(gt_.name)
    pred = tf.reshape(pred_,[-1])
    gt = tf.reshape(gt_,[-1])
    #pred = tf.cast(pred, dtype=tf.int64)#
    gt = tf.cast(gt, dtype=tf.uint8)#
    # Ignoring all labels greater than or equal to n_classes.
    temp = tf.less_equal(gt, num_classes - 1)
    weights = tf.cast(temp, tf.int32)

    # fix for tf 1.3.0
    gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))#true select gt ;False select temp

    # Pixel accuracy
    accu, accu_update_op = tf.contrib.metrics.streaming_accuracy(
        pred, gt, weights=weights)

    # mIoU
    mIoU, mIou_update_op = tf.contrib.metrics.streaming_mean_iou(
        pred, gt, num_classes=num_classes, weights=weights)

    # confusion matrix
    confusion_matrix = tf.contrib.metrics.confusion_matrix(
        pred, gt, num_classes=num_classes, weights=weights)

    # # Start queue threads.
    # threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    # Test!
    
    
    return accu, accu_update_op, mIoU, mIou_update_op, confusion_matrix

def test(sess):
    confusion_matrix_ = np.zeros((num_classes, num_classes), dtype=np.int)
    accu, accu_update_op, mIou, mIou_update_op, confusion_matrix = graph()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    f = open('./data/val.txt','r')

    #for step in range(valid_num_steps):
    for step, line in enumerate(f.readlines()):
        
        pred_path = './output/' + line.split(' ')[0].split('/')[2] + '_output_0.bin'
        label_path = './bin_dataset/' + line.split(' ')[1].split('\n')[0]+'.bin'
        preds = np.fromfile(pred_path,dtype = np.int64)
        label = np.fromfile(label_path, dtype = np.float32)
        print("*******************",label,label.dtype)
        print(len(preds),len(label))
        #t_preds = tf.convert_to_tensor(preds)
        #t_label = tf.convert_to_tensor(label)
        #t_preds = tf.cast(t_preds, dtype=tf.int64)
        #t_label = tf.cast(t_label, dtype=tf.uint8)
        
        _, _, c_matrix = sess.run([accu_update_op, mIou_update_op, confusion_matrix], feed_dict = {'Placeholder:0': preds, 'Placeholder_1:0': label.astype(np.int64)})
        
        confusion_matrix_ += c_matrix
        # if step % 1 == 0:
        print('step {:d}'.format(step))
    print('Pixel Accuracy: {:.3f}'.format(accu.eval(session=sess)))
    print('Mean IoU: {:.3f}'.format(mIou.eval(session=sess)))
    compute_IoU_per_class(confusion_matrix_)

if __name__ == "__main__":
    tf.reset_default_graph()
    with tf.Session() as sess:
        
        test(sess)
    
