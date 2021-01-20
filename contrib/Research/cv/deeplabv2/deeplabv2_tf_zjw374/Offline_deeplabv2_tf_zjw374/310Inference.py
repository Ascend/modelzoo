from datetime import datetime
import os
import sys
import time
import numpy as np
import tensorflow as tf

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
   
    pred_ = tf.placeholder(tf.int64, shape = None)
    gt_ = tf.placeholder(tf.int64, shape = None)
    pred = tf.reshape(pred_,[-1])
    gt = tf.reshape(gt_,[-1])
    gt = tf.cast(gt, dtype=tf.uint8)
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
        print(len(preds),len(label))
         
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
    
