import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np
sys.path.append(os.getcwd())
from tensorflow.contrib import slim
from nets import model_train as model
from utils.dataset import data_provider as data_provider
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from tensorflow.python.client import timeline

tf.app.flags.DEFINE_float('learning_rate', 1e-5, '')
tf.app.flags.DEFINE_integer('max_steps', 50000, '')
tf.app.flags.DEFINE_integer('decay_steps', 30000, '')
tf.app.flags.DEFINE_float('decay_rate', 0.1, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('num_readers', 4, '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path',"checkpoints_mlt/" , '')
tf.app.flags.DEFINE_string('logs_path', 'logs_mlt/', '')
tf.app.flags.DEFINE_string('pretrained_model_path', 'data/vgg_16.ckpt', '')
tf.app.flags.DEFINE_boolean('restore', False, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 2000, '')
FLAGS = tf.app.flags.FLAGS


def pad_input(inputs,target_shape=[1216,1216,3]):

    h,w = inputs.shape[:2]
    out = np.zeros(target_shape).astype(np.uint8)
    out[0:h,0:w,:] = inputs

    return out


def pad_bbox(inputs, count=576):
    if len(inputs)>count:
        return inputs[:count]
   
    else:    
        out = inputs.copy()
        num_inputs = len(out)
        num_pad = count - num_inputs
        
        for i in range(num_pad):
            #out.append(inputs[i%num_inputs].copy())
            out.append([0,0,0,0,1])
        return out

def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(FLAGS.logs_path + StyleTime)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    input_image = tf.placeholder(tf.float32, shape=[1, 1216, 1216, 3], name='input_image')
    input_bbox = tf.placeholder(tf.float32, shape=[576, 5], name='input_bbox')
    num_input_bbox = tf.placeholder(tf.int32, shape=[1], name='num_input_bbox')
    #input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)

    with tf.name_scope('model' ) as scope:
        bbox_pred, cls_pred, cls_prob = model.model(input_image)
        total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = model.loss_v2(bbox_pred, cls_pred, input_bbox,num_input_bbox)
                                                                             
        batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_path + StyleTime, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    config = tf.ConfigProto()
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    with tf.Session(config=config) as sess:
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            restore_step = int(ckpt.split('.')[0].split('_')[-1])
            print("continue training from previous checkpoint {}".format(restore_step))
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            restore_step = 0
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)
        data_generator = data_provider.get_batch(num_workers=FLAGS.num_readers)
        start = time.time()
    
        for step in range(restore_step, FLAGS.max_steps):
            data = next(data_generator)
            inputs_padded = [pad_input(data[0][0])]
            bbox_padded = pad_bbox(data[1])
            ml, tl,ce_loss, box_loss, _, summary_str = sess.run([model_loss, total_loss,
                                               rpn_cross_entropy,
                                               rpn_loss_box,
                                               train_op, summary_op],
                                              feed_dict={input_image: inputs_padded,
                                                         input_bbox: bbox_padded,
                                                         num_input_bbox: [len(data[1])]})
            summary_writer.add_summary(summary_str, global_step=step)
            print('model loss :', ml, 'ce_loss: ', ce_loss, 'box_loss:',box_loss)
            if step != 0 and step % FLAGS.decay_steps == 0:
                sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
                    step, ml, tl, avg_time_per_step, learning_rate.eval()))

            if (step + 1) % FLAGS.save_checkpoint_steps == 0:
                filename = ('ctpn_{:d}'.format(step + 1) + '.ckpt')
                filename = os.path.join(FLAGS.checkpoint_path, filename)
                saver.save(sess, filename)
                print('Write model to: {:s}'.format(filename))

if __name__ == '__main__':
    tf.app.run()
