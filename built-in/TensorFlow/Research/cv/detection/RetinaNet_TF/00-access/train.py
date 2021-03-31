
from npu_bridge.npu_init import *
from networks import backbone
import tensorflow as tf
import numpy as np
from utils import generate_anchors, read_batch_data
from ops import smooth_l1, focal_loss
from config import BATCH_SIZE, IMG_H, IMG_W, K, WEIGHT_DECAY, LEARNING_RATE

def npu_tf_optimizer(opt):
    npu_opt = NPUDistributedOptimizer(opt)
    return npu_opt
anchors_p3 = generate_anchors(area=32, stride=8)
anchors_p4 = generate_anchors(area=64, stride=16)
anchors_p5 = generate_anchors(area=128, stride=32)
anchors_p6 = generate_anchors(area=256, stride=64)
anchors_p7 = generate_anchors(area=512, stride=128)
anchors = np.concatenate((anchors_p3, anchors_p4, anchors_p5, anchors_p6, anchors_p7), axis=0)

def train():
    inputs = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_H, IMG_W, 3])
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, None, K])
    target_bbox = tf.placeholder(tf.float32, [BATCH_SIZE, None, 4])
    foreground_mask = tf.placeholder(tf.float32, [BATCH_SIZE, None])
    valid_mask = tf.placeholder(tf.float32, [BATCH_SIZE, None])
    is_training = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32)
    (class_logits, box_logits, _, _) = backbone(inputs, is_training)
    class_loss = (tf.reduce_sum((focal_loss(class_logits, labels) * valid_mask)) / tf.reduce_sum(foreground_mask))
    box_loss = (tf.reduce_sum((smooth_l1((box_logits - target_bbox)) * foreground_mask)) / tf.reduce_sum(foreground_mask))
    l2_reg = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    total_loss = ((class_loss + box_loss) + (l2_reg * WEIGHT_DECAY))
    with tf.variable_scope('opt'):
        Opt = npu_tf_optimizer(tf.train.MomentumOptimizer(learning_rate, momentum=0.9)).minimize(total_loss)
    sess = tf.Session(config=npu_session_config_init())
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v2_50'))
    saver.restore(sess, './resnet_ckpt/resnet_v2_50.ckpt')
    saver = tf.train.Saver()
    LR = LEARNING_RATE
    for i in range(1000):
        if (i == 60000):
            LR /= 10
        if (i == 80000):
            LR /= 10
        (IMGS, FOREGROUND_MASKS, VALID_MASKS, LABELS, TARGET_BBOXES) = read_batch_data(anchors)
        [_, TOTAL_LOSS, CLASS_LOSS, BOX_LOSS] = sess.run([Opt, total_loss, class_loss, box_loss], feed_dict={inputs: ((IMGS / 127.5) - 1.0), labels: LABELS, target_bbox: TARGET_BBOXES, foreground_mask: FOREGROUND_MASKS, valid_mask: VALID_MASKS, is_training: True, learning_rate: LR})
        if ((i % 100) == 0):
            print(('Iteration: %d, Total Loss: %f, Class Loss: %f, Box Loss: %f' % (i, TOTAL_LOSS, CLASS_LOSS, BOX_LOSS)))
        if ((i % 5000) == 0):
            saver.save(sess, './model/model.ckpt')
        pass
if (__name__ == '__main__'):
    train()
