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
# Copyright 2020 Huawei Technologies Co., Ltd
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

import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np

from model import ICNet_BN
from utils.config import Config
from utils.visualize import decode_labels
from utils.image_reader import ImageReader, prepare_label

from npu_bridge.estimator import npu_ops 
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig 
i_list = [2025, 7921, 32041]
i_count = 0
from tensorflow import float32  


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced ICNet")
    
    parser.add_argument("--random-mirror", default=False,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--update-mean-var", default=False,
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", default=False,
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument("--dataset", required=True,
                        help="Which dataset to trained with",
                        choices=['cityscapes', 'ade20k', 'others'])
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    #parser.add_argument("--dataurl", required=True,
    #                    help="Which dataset to trained with Ascend 910",
    #                    choices=['$cityscapes_dataset', 'others'])
    
    return parser.parse_args()

def get_mask(gt, num_classes, ignore_label):
    '''
    less_equal_class = tf.less_equal(gt, num_classes-1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)
    #indices = tf.ones([32041,], dtype=tf.int32)

    '''

    less_equal_class = tf.less_equal(gt, 18)
    not_equal_ignore = tf.not_equal(gt, 255)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    sess=tf.Session()
    mask = sess.run(mask)
    for i in range(len(mask)):
        for j in range(len(mask[i].shape)):
            if mask[i][j] == True:
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    indices = mask.astype(np.int32)     
    

    return indices
    
    

    
def create_loss(output, label, num_classes, ignore_label):
    # raw_pred = tf.reshape(output, [-1, num_classes]) # 原始代码
    
    tmp_tianyili = tf.size(output) // 19
    raw_pred = tf.reshape(output, [tmp_tianyili, num_classes]) 

    

    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    
    global i_list
    global i_count
    # print("i_list[i_count]", i_list[i_count])
    label = tf.reshape(label, [i_list[i_count],])
    i_count = i_count + 1
    if i_count > 2:
        i_count = 0
    

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)

    loss = tf.nn.softmax(logits=pred)
    reduced_loss = tf.reduce_mean(loss) 
    #reduced_loss = loss

    return reduced_loss

def create_losses(net, label, cfg):
    # Get output from different branches
    sub4_out = net.layers['sub4_out']
    sub24_out = net.layers['sub24_out']
    sub124_out = net.layers['conv6_cls']

    loss_sub4 = create_loss(sub4_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
    loss_sub24 = create_loss(sub24_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
    loss_sub124 = create_loss(sub124_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])

    l2_losses = [cfg.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    
    # Calculate weighted loss of three branches, you can tune LAMBDA values to get better results.
    reduced_loss = cfg.LAMBDA1 * loss_sub4 +  cfg.LAMBDA2 * loss_sub24 + cfg.LAMBDA3 * loss_sub124 + tf.add_n(l2_losses)

    return loss_sub4, loss_sub24, loss_sub124, reduced_loss

class TrainConfig(Config):
    def __init__(self, dataset, is_training,  filter_scale=1, random_scale=None, random_mirror=None):
        Config.__init__(self, dataset, is_training, filter_scale, random_scale, random_mirror)

    # Set pre-trained weights here (You can download weight using `python script/download_weights.py`) 
    # Note that you need to use "bnnomerge" version.
    ######################################################################################################################3
    model_weight = './model/cityscapes/icnet_cityscapes_train_30k_bnnomerge.npy'
    #model_weight = ''
    
    # Set hyperparameters here, you can get much more setting in Config Class, see 'utils/config.py' for details.
    LAMBDA1 = 0.16
    LAMBDA2 = 0.4
    LAMBDA3 = 1.0
    BATCH_SIZE = 1
    LEARNING_RATE = 5e-4

def main():
    """Create the model and start the training."""
    args = get_arguments()
    #print("args --dataurl:", args.dataurl)

    """
    Get configurations here. We pass some arguments from command line to init configurations, for training hyperparameters, 
    you can set them in TrainConfig Class.

    Note: we set filter scale to 1 for pruned model, 2 for non-pruned model. The filters numbers of non-pruned
          model is two times larger than prunde model, e.g., [h, w, 64] <-> [h, w, 32].
    """
    cfg = TrainConfig(dataset=args.dataset, 
                is_training=True,
                random_scale=False,
                random_mirror=False,
                filter_scale=args.filter_scale)
    cfg.display()
    

    # Setup training network and training samples
    train_reader = ImageReader(cfg=cfg, mode='train')
    train_net = ICNet_BN(image_reader=train_reader, 
                            cfg=cfg, mode='train')

    loss_sub4, loss_sub24, loss_sub124, reduced_loss = create_losses(train_net, train_net.labels, cfg)

    # Setup validation network and validation samples
    with tf.variable_scope('', reuse=True):
        val_reader = ImageReader(cfg, mode='eval')
        val_net = ICNet_BN(image_reader=val_reader, 
                            cfg=cfg, mode='train')

        val_loss_sub4, val_loss_sub24, val_loss_sub124, val_reduced_loss = create_losses(val_net, val_net.labels, cfg)

    # Using Poly learning rate policy 
    base_lr = tf.constant(cfg.LEARNING_RATE)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / cfg.TRAINING_STEPS), cfg.POWER))
    
    # Set restore variable 
    restore_var = tf.global_variables()
    all_trainable = [v for v in tf.trainable_variables() if ('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma]

    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if args.update_mean_var == False:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        opt_conv = tf.train.MomentumOptimizer(learning_rate, cfg.MOMENTUM)
       
        grads = tf.gradients(reduced_loss, all_trainable)
        #print("测试测试测试：grads：", grads)
        #grads = tf.ones([3, 3, 3, 32], dtype=float32)
        #print("测试测试grads_constant", grads_constant)
        #grads = tf.constant(grads_constant, dtype=tf.float32,shape=(3, 3, 3, 32))
        #print("测试测试测试grads_constant", grads_constant)
        train_op = opt_conv.apply_gradients(zip(grads, all_trainable))
        # print("测试测试测试：train_op：", train_op)
    
    # Create session & restore weights (Here we only need to use train_net to create session since we reuse it)
    train_net.create_session()

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

    # Iterate over training steps.
    for step in range(cfg.TRAINING_STEPS):
        start_time = time.time()
            
        feed_dict = {step_ph: step}
        if step % cfg.SAVE_PRED_EVERY == 0:

            config = tf.ConfigProto()
            custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name =  "NpuOptimizer"
            custom_op.parameter_map["use_off_line"].b = True # 必须显示开启，在昇腾AI处理器执行训练
            config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显示关闭remap
            # config.graph_options.rewrite_options.optimizers.extend(["GradFusionOptimizer"]) #分布式添加

            loss_value, loss1, loss2, loss3, val_loss_value = train_net.sess.run([reduced_loss, loss_sub4, loss_sub24, loss_sub124, val_reduced_loss], feed_dict=feed_dict)
            
            train_net.save(saver, cfg.SNAPSHOT_DIR, step)
        else:
            
            config = tf.ConfigProto()
            custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name =  "NpuOptimizer"
            custom_op.parameter_map["use_off_line"].b = True # 必须显示开启，在昇腾AI处理器执行训练
            config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显示关闭remap
            
            loss_value, loss1, loss2, loss3, val_loss_value = train_net.sess.run([reduced_loss, loss_sub4, loss_sub24, loss_sub124, val_reduced_loss], feed_dict=feed_dict)  

        duration = time.time() - start_time
        print('step {:d} \t total loss = {:.3f}, sub4 = {:.3f}, sub24 = {:.3f}, sub124 = {:.3f}, val_loss: {:.3f} ({:.3f} sec/step)'.\
                    format(step, loss_value, loss1, loss2, loss3, val_loss_value, duration))
    
    
if __name__ == '__main__':
    main()