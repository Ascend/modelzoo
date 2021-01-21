from __future__ import print_function

import datetime
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 warning 和 Error 
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error


import sys
import time
import math
import threading
from multiprocessing import Process
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops


from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu import npu_plugin
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_config import NpuExecutePlacement
from npu_bridge.estimator.npu import npu_scope
from npu_bridge.estimator.npu import util
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_optimizer import allreduce
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.hccl import hccl_ops
from hccl.manage.api import create_group
from hccl.manage.api import get_local_rank_id
from hccl.manage.api import get_rank_size
from hccl.manage.api import get_local_rank_size
from hccl.manage.api import get_rank_id
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.npu_cpu.npu_cpu_ops import embeddingrankid
from npu_bridge import tf_adapter

from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager




from sklearn.metrics import roc_auc_score




class WideDeep:
    """support performing mean pooling Operation on multi-hot feature.
    dataset_argv: e.g. [10000, 17, [False, ...,True,True], 10]
    """

    def __init__(self, dataset_argv, 
                       architect_argv, 
                       ptmzr_argv,
                       reg_argv, 
                       input_data, 
                       host_params_shape,
                       loss_mode='batch'):
        self.id_vocab_size, self.field_num = dataset_argv
        self.embed_size, self.layer_dims, self.act_func = architect_argv
        self.keep_prob, self._lambda = reg_argv
        self.input_data = input_data
        self.host_params_shape = host_params_shape
        
        self.embed_dim = self.field_num * self.embed_size
        self.all_layer_dims = [self.embed_dim] + self.layer_dims + [1]

        self.ptmzr_argv = ptmzr_argv # adam args
        self.loss_mode = loss_mode
        
        self.init_graph()
    # 
    def init_graph(self):
        self.init_inputs()
        self.init_variables()
        # 
        self.unique_emb, self.unique_emb_moment, self.unique_emb_volecity, \
            self.unique_weight_table, self.unique_w_accum, self.unique_w_linear, \
            self.global_unique_pad = self.get_global_unique_id_params(self.global_unique_id)
        # 

        self.wts_mask = tf.expand_dims(self.local_wts, axis=2)

        self.id_emb = tf.gather(self.unique_emb, self.local_unique_id_inver) # [bs, 39, 80]
        self.id_emb_mask = tf.multiply(self.id_emb, self.wts_mask)
        self.deep_out = self.deep_forward(self.id_emb_mask, self.keep_prob, training=True)

        self.id_weight = tf.gather(self.unique_weight_table, self.local_unique_id_inver)
        self.id_weight_mask = tf.multiply(self.id_weight, self.wts_mask, name="id_weight_mask")
        self.wide_out = tf.reshape(tf.reduce_sum(self.id_weight_mask, axis=1), shape=[-1, ], name="wide_out")


        self.logits = self.deep_out + self.wide_out

        self.train_preds = tf.sigmoid(self.logits, name='predicitons')
        self.log_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.local_labels)
        self.base_loss = tf.reduce_mean(self.log_loss)
        self.wide_loss = self.base_loss
        if self.loss_mode == "batch":
            self.l2_loss = tf.nn.l2_loss(self.id_emb)
        else:
            self.l2_loss = tf.constant([0.0])
        self.deep_loss = self.base_loss + self._lambda * self.l2_loss

        self.init_optimizer(self.deep_loss, self.wide_loss)
        self.init_evaluate()
    # 

    def init_inputs(self):
        self.global_unique_id = self.input_data["global_unique_id"]
        self.global_unique_id_inver = self.input_data["global_unique_id_inver"]
        self.global_unique_shape = self.input_data["global_unique_shape"]

        self.global_wts = self.input_data["global_wts"]
        self.global_labels = self.input_data["global_labels"]
        
        self.rank_size = get_rank_size()
        self.rank_id = get_rank_id()
        self.batch_size = self.global_labels.shape[0] // self.rank_size
        
        self.local_labels = tf.split(self.global_labels, self.rank_size, axis=0)[self.rank_id]
        self.local_wts = tf.split(self.global_wts, self.rank_size, axis=0)[self.rank_id]
        self.local_unique_id_inver = tf.split(self.global_unique_id_inver, self.rank_size, axis=0)[self.rank_id]
        

        self.bar = tf.constant([[1,1,0]], dtype=tf.float32)
        self.chief_num = get_local_rank_size() // get_rank_size()
        self.host_params_address = tf.placeholder(tf.uint64, shape=[self.chief_num, 3], name='host_params_address')
    # 
    def init_variables(self):
        with tf.variable_scope("embedding", reuse=False):
            self.global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                           initializer=tf.constant_initializer(1), trainable=False)
        # 
        with tf.variable_scope("deep_layer", reuse=False):
            self.h_w, self.h_b = [], []
            for i in range(len(self.all_layer_dims) - 1):
                self.h_w.append(tf.get_variable('h%d_w' % (i + 1), shape=self.all_layer_dims[i: i + 2],
                                                initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                                dtype=tf.float32,
                                                trainable=True,
                                                collections=[tf.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_wts"]))
                self.h_b.append(
                    tf.get_variable('h%d_b' % (i + 1), shape=[self.all_layer_dims[i + 1]], 
                                    initializer=tf.zeros_initializer,
                                    dtype=tf.float32,
                                    trainable=True,
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_bias"]))
        # 
    # 
    def get_global_unique_id_params(self, global_unique_id):
        global_unique_id = tf.cast(global_unique_id, dtype=tf.int32)
        self.global_unique_host_params, \
            self.unique_id_localized_mask, self.global_unique_id_localized, \
            self.global_unique_id_localized_address, self.global_unique_host_params_localized = self.get_host_params(global_unique_id, mode="train")
        # 
        self.global_unique_host_params = hccl_ops.allreduce(self.global_unique_host_params, reduction='sum')
        # 
        global_unique_emb, global_unique_emb_moment, global_unique_emb_volecity, \
            global_unique_ftrl_w, global_unique_ftrl_z, global_unique_ftrl_n, global_unique_pad \
                = self.host_params_split(self.global_unique_host_params)
        # 
        return global_unique_emb, global_unique_emb_moment, global_unique_emb_volecity, \
            global_unique_ftrl_w, global_unique_ftrl_z, global_unique_ftrl_n, global_unique_pad
    # 

    def host_params_split(self, unique_host_params):
        # self.host_params_split_index = [80, 80, 80, 1, 1, 1, 57] # emb, emb_moment, emb_volecity, weight, z_w, n_w, pad
        pad_dim = self.host_params_shape[1] - (self.embed_size + 1) * 3
        self.host_params_split_index = [self.embed_size, self.embed_size, self.embed_size, 
                                        1, 1, 1, 
                                        pad_dim ] # emb, emb_moment, emb_volecity, weight, z_w, n_w, pad
        emb, emb_moment, emb_volecity, ftrl_w, ftrl_z, ftrl_n, pad = tf.split(unique_host_params, self.host_params_split_index, axis=1)
        return emb, emb_moment, emb_volecity, ftrl_w, ftrl_z, ftrl_n, pad
    # 
    def get_host_params(self, global_unique_id, mode="train"):
        """  paddding -1  对 global_unique_id 涉及到的 计算的影响  """
        unique_id_localized_mask = tf.where(tf.equal(tf.cast(get_rank_id(), dtype=tf.int32), global_unique_id % get_rank_size()))
        g_unique_id_localized = tf.gather_nd(global_unique_id, unique_id_localized_mask)
        # 
        global_unique_id_localized_address = embeddingrankid(self.host_params_address, 
                                                                  #tf.cast(g_unique_id_localized,tf.int32), 
                                                                  g_unique_id_localized,
                                                                  row_memory=self.host_params_shape[1] * 4)
        global_unique_host_params_localized = hccl_ops.remote_read(global_unique_id_localized_address, data_type=tf.float32) # [5, 300]
        self.bar2 = hccl_ops.allreduce(self.bar, reduction='sum')
        # 
        # self.fix_shape = [16000*40, 300]
        fix_shape = tf.shape(global_unique_id)[0]
        #fix_shape = 48000
        # self.fix_shape = self.global_unique_shape[0]
        global_unique_host_params = tf.scatter_nd(tf.cast(unique_id_localized_mask, tf.int32), 
                                                  global_unique_host_params_localized,
                                                  shape=[fix_shape, self.host_params_shape[1]], 
                                                  name='host_params_scatter')
        return global_unique_host_params, \
               unique_id_localized_mask, g_unique_id_localized, \
               global_unique_id_localized_address, global_unique_host_params_localized
        # 
    # 


    def wide_forward(self, id_weight_mask):
        id_weight_mask_sum = tf.reduce_sum(id_weight_mask, axis=1)
        wide_output = tf.reshape((id_weight_mask_sum + self.wide_b), shape=[-1, ], name="wide_out")
        return wide_output
    # 
    
    def deep_forward(self, id_emb_mask, keep_prob, training):
        deep_out = tf.reshape(id_emb_mask, [-1, self.embed_dim])
        for i in range(len(self.h_w)):
            deep_out = tf.nn.relu(deep_out)
            if training:
                deep_out = npu_ops.dropout(deep_out, keep_prob=keep_prob)
            deep_out = tf.matmul(deep_out, self.h_w[i])

            deep_out = deep_out + self.h_b[i]
        deep_out = tf.reshape(deep_out, [-1, ], name="deep_out")
        return deep_out
    # 
    
    def init_optimizer(self, deep_loss, wide_loss):
        self.opt_deep, self.lr_deep, self.eps_deep = self.ptmzr_argv[0]
        opt_wide, self.wide_lr, self.wide_dc, self.wide_l1, self.wide_l2 = self.ptmzr_argv[1]

        self.deep_optimzer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.lr_deep, epsilon=self.eps_deep)
        var_list = [var for var in tf.trainable_variables() if 'deep' in var.name]
        
        loss_scale_coef = 1000.0
        grads_vars_list = self.deep_optimzer.compute_gradients(deep_loss * loss_scale_coef, var_list=var_list)
        grads_vars_list = [(grad / loss_scale_coef, var) for grad,var in grads_vars_list]
        grads_vars_list = [(hccl_ops.allreduce(grad, reduction='sum', fusion=0)/self.rank_size, var) for grad,var in grads_vars_list]
        
        self.deep_update = self.deep_optimzer.apply_gradients(grads_vars_list)
        # 

        self.emb_params = self.adam_update_embedding(deep_loss)
        self.weight_params = self.ftrl_update_weight(self.wide_loss)
        self.remote_write_op = self.remote_write(self.emb_params, self.weight_params)
        self.train_op = [self.deep_update, self.remote_write_op]
    # 
    
    def _adam_iter_(self, global_steps, dtype=tf.float32):
        self._use_locking = False
        self.lr_t = ops.convert_to_tensor(self.lr_deep, name="learning_rate")
        self.beta1 = ops.convert_to_tensor(0.9, name="beta1")
        self.beta2 = ops.convert_to_tensor(0.999, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self.eps_deep, name="epsilon")
        
        global_steps = math_ops.cast(global_steps, dtype)

        self.lr_t = math_ops.cast(self.lr_t, dtype)
        self.beta1 = math_ops.cast(self.beta1, dtype)
        self.beta2 = math_ops.cast(self.beta2, dtype)
        self.epsilon_t = math_ops.cast(self._epsilon_t, dtype)

        self.beta1_power = math_ops.pow(self.beta1, global_steps, name="beta1_power")
        self.beta2_power = math_ops.pow(self.beta2, global_steps, name="beta2_power")
        self.beta1_power = math_ops.cast(self.beta1_power, dtype)
        self.beta2_power = math_ops.cast(self.beta2_power, dtype)

    def _adam_update(self, var, grad, moment, volecity, global_steps, dtype=tf.float32):
        self._adam_iter_(global_steps, dtype=dtype)

        self.beta1_moment = self.beta1 * moment
        self.beta1_grad = (1.0 - self.beta1) * grad 
        self.beta2_volecity = self.beta2 * volecity
        self.beta2_grad_square = (1.0 - self.beta2) * (grad * grad)
        self.moment_t = self.beta1_moment + self.beta1_grad
        self.volecity_t = self.beta2_volecity + self.beta2_grad_square 
        
        # power 衰减
        print("-----  power 衰减  ------")
        moment_hat = self.moment_t / (1.0 - self.beta1_power)
        volecity_hat = self.volecity_t / (1.0 - self.beta2_power)
        var_update = var - self.lr_t * moment_hat / (math_ops.sqrt(volecity_hat) + self.epsilon_t)
        # 

        return var_update, self.moment_t, self.volecity_t

    def adam_update_embedding(self, loss):
        
        loss_scale_coef = 1000.0
        self.id_emb_grad = tf.gradients(loss * loss_scale_coef, [self.id_emb])
        


        self.unique_id_emb_grad = tf.math.unsorted_segment_sum(tf.reshape(self.id_emb_grad, [-1, self.embed_size]), 
                                                          segment_ids=tf.reshape(self.local_unique_id_inver, [-1]), 
                                                          num_segments=tf.shape(self.global_unique_id)[0], name="adam_unsorted")                                                   
        # 
        self.unique_id_emb_grad = self.unique_id_emb_grad / loss_scale_coef
        self.unique_id_emb_grad = hccl_ops.allreduce(self.unique_id_emb_grad, reduction='sum', fusion=0) / self.rank_size
        
        self.var_t, self.moment_t, self.volecity_t = self._adam_update(self.unique_emb, 
                                                                 self.unique_id_emb_grad, 
                                                                 self.unique_emb_moment, 
                                                                 self.unique_emb_volecity, 
                                                                 self.global_step,
                                                                 dtype=tf.float32)
        
        self.host_emb_params = tf.concat([self.var_t, self.moment_t, self.volecity_t], axis=1)
        # 
        
        return self.host_emb_params
        
    def _ftrl_iter_(self, global_steps, dtype=tf.float32):
        # lr, lr_power=-0.5, l1=0.0, l2=0.0, l2_shrinkage=0.0
        self.wide_lr = ops.convert_to_tensor(self.wide_lr, name="learning_rate")
        self.wide_lr_power = ops.convert_to_tensor(-0.5, name="learning_rate")
        self.wide_dc = ops.convert_to_tensor(self.wide_dc, name="learning_rate")
        self.wide_l1 = ops.convert_to_tensor(self.wide_l1, name="learning_rate")
        self.wide_l2 = ops.convert_to_tensor(self.wide_l2, name="learning_rate")
        self.wide_l2_shrinkage = ops.convert_to_tensor(0.0, name="learning_rate")
    #   
    def _ftrl_update(self, grad, var, accum, linear, global_steps, dtype=tf.float32):
        """
        https://tensorflow.google.cn/api_docs/cc/class/tensorflow/ops/sparse-apply-ftrl-v2
        """
        self._ftrl_iter_(global_steps)
        # grad_with_shrinkage = grad + 2 * l2_shrinkage * var
        grad_shrinkage = grad + 2.0 * tf.multiply(self.wide_l2_shrinkage, var)

        # accum_new = accum + grad_with_shrinkage * grad_with_shrinkage
        accum_new = accum + grad_shrinkage * grad_shrinkage

        # linear += grad_with_shrinkage + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
        sigma = (math_ops.pow(accum_new, -self.wide_lr_power) - math_ops.pow(accum, -self.wide_lr_power)) / self.wide_lr 
        # linear_new = linear + grad_shrinkag + sigma * var
        linear_new = linear + grad_shrinkage - sigma * var
        
        # quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
        quadratic = 1.0 / (math_ops.pow(accum_new, self.wide_lr_power) * self.wide_lr ) + 2.0 * self.wide_l2
        
        # var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
        new_var = (tf.math.sign(linear_new) * self.wide_l1 - linear_new) / quadratic
        new_mask = tf.cast(tf.math.greater(tf.abs(linear_new), self.wide_l1), dtype=tf.float32)
        var_new = tf.multiply(new_var, new_mask)

        return var_new, accum_new, linear_new
    # 
    def ftrl_update_weight(self, loss):
        self.id_weight_grad = tf.gradients(loss, [self.id_weight])
        
        self.unique_id_weight_grad = tf.math.unsorted_segment_sum(tf.reshape(self.id_weight_grad, [-1, 1]), 
                                                               segment_ids=tf.reshape(self.local_unique_id_inver, [-1]), 
                                                               num_segments=tf.shape(self.global_unique_id)[0], name="ftrl_unsorted")                                                   
        self.unique_id_weight_grad = hccl_ops.allreduce(self.unique_id_weight_grad, reduction='sum') / self.rank_size
        
        # 

        self.var_new, self.accum_new, self.linear_new = self._ftrl_update(self.unique_id_weight_grad, 
                                                                       self.unique_weight_table, 
                                                                       self.unique_w_accum, 
                                                                       self.unique_w_linear, 
                                                                       self.global_step,
                                                                       dtype=tf.float32)
        # 
        weight_params = tf.concat([self.var_new, self.accum_new, self.linear_new], axis=1)
        return weight_params
    # 
    def remote_write(self, emb_params, weight_params):
        self.host_params = tf.concat([emb_params, weight_params, self.global_unique_pad], axis=1)
        # 
        self.host_params_localized = tf.gather_nd(self.host_params, self.unique_id_localized_mask)
        new_update_op = hccl_ops.remote_write(self.global_unique_id_localized_address, 
                                              self.host_params_localized, data_type=tf.float32) 
        # 
        return new_update_op
    # 

    def need_init_varlist(self):
        var_list = [var for var in tf.global_variables() if var is not self.id_weight
                    and var is not self.id_emb]
        return var_list
    # 
    def init_eval_inputs(self):
        self.EVAL_labels = tf.placeholder(tf.float32, shape=[None, ], name='EVAL_labels')
        self.EVAL_wts = tf.placeholder(tf.float32, shape=[None, 39], name='EVAL_wts')
        self.EVAL_unique_id = tf.placeholder(tf.int64, shape=[None], name='EVAL_unique_id')
        self.EVAL_unique_id_inver = tf.placeholder(tf.int32, shape=[None, 39], name='EVAL_unique_id_inver')
        self.EVAL_unique_shape = tf.placeholder(tf.int32, shape=[1], name='EVAL_unique_shape')
    # 
    def init_evaluate(self):
        self.init_eval_inputs()
        self.EVAL_unique_id = tf.cast(self.EVAL_unique_id, dtype=tf.int32)

        self.global_unique_id_emb, self.global_unique_emb_moment, self.global_unique_emb_volecity, \
            self.global_unique_ftrl_w, self.global_unique_ftrl_z, self.global_unique_ftrl_n, \
            self.global_unique_pad = self.get_global_unique_id_params(self.EVAL_unique_id)
        # 
        EVAL_wt_mask = tf.expand_dims(self.EVAL_wts, axis=2)
        # 
        EVAL_id_emb = tf.gather(self.global_unique_id_emb, self.EVAL_unique_id_inver) # [bs, 39, 80]
        EVAL_id_emb_mask = tf.multiply(EVAL_id_emb, EVAL_wt_mask)
        
        EVAL_id_weight = tf.gather(self.global_unique_ftrl_w, self.EVAL_unique_id_inver)
        EVAL_id_weight_mask = tf.multiply(EVAL_id_weight, EVAL_wt_mask, name="eval_id_weight_mask")

        EVAL_wide_out = tf.reshape(tf.reduce_sum(EVAL_id_weight_mask, axis=1), shape=[-1, ], name="wide_out")
        EVAL_deep_out = self.deep_forward(EVAL_id_emb_mask, 1.0, training=False)
        EVAL_logits = EVAL_deep_out + EVAL_wide_out

        self.EVAL_preds = tf.sigmoid(EVAL_logits, name='predictionNode')
        self.EVAL_log_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=EVAL_logits, labels=self.EVAL_labels)
        self.EVAL_base_loss = tf.reduce_mean(self.EVAL_log_loss)
    # 
#

def input_fn_tfrecord(data_path,file_pattern, batch_size=16000,
                      num_epochs=1, num_parallel=16, perform_shuffle=False, line_per_sample=1000):
    batch_size = int(batch_size/line_per_sample)
    def extract_fn(data_record):
        features = {
            # Extract features using the keys set during creation
            'label': tf.FixedLenFeature(shape=(line_per_sample, ), dtype=tf.float32),
            'feat_ids': tf.FixedLenFeature(shape=(39 * line_per_sample,), dtype=tf.int64),
            'feat_vals': tf.FixedLenFeature(shape=(39 * line_per_sample,), dtype=tf.float32),
        }
        sample = tf.parse_single_example(data_record, features)
        sample['feat_ids'] = tf.cast(sample['feat_ids'], dtype=tf.int32)
        return sample

    def unique_fn(sample):
        # with ops.get_default_graph()._kernel_label_map({"Unique" : "parallel"}):
        uniq_matrix, reverse_matrix = tf.unique(tf.reshape(sample['feat_ids'], [-1]) ) # [bs, 39] --> [ None ]
        sample['global_unique_id'] = tf.reshape(uniq_matrix, [-1])
        sample['global_unique_id_inver'] = tf.reshape(reverse_matrix, [-1, 39 * line_per_sample])
        sample['global_unique_shape'] = tf.shape(uniq_matrix)
        return sample

    def reshape_fn(sample):    
        sample['label'] = tf.reshape(sample['label'], [-1,])
        sample['feat_ids'] = tf.reshape(sample['feat_ids'], [-1, 39])
        sample['feat_vals'] = tf.reshape(sample['feat_vals'], [-1, 39])
        sample['global_unique_id'] = tf.reshape(sample['global_unique_id'], [-1])
        sample['global_unique_id_inver'] = tf.reshape(sample['global_unique_id_inver'], [-1, 39])
        sample['global_unique_shape'] = tf.reshape(sample['global_unique_shape'], [1])
        return sample 

    line_num = int(batch_size/line_per_sample)
    unique_shape = line_num * 3000
    local_rank_size = get_local_rank_size()
    if (local_rank_size == 8):
        unique_shape = 120000
    path = data_path
    all_files = os.listdir(path)
    files = [os.path.join(path,f) for f in all_files if f.startswith(file_pattern)]
    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.repeat(num_epochs)
    # if perform_shuffle:
    #    dataset = dataset.shuffle(16000,seed=1234)
    dataset = dataset.map(extract_fn).batch(batch_size, drop_remainder=True)


    padded_shapes = {
        'label' : tf.TensorShape([batch_size, line_per_sample, ]),
        'feat_ids' : tf.TensorShape([batch_size, 39*line_per_sample,]),
        'feat_vals' : tf.TensorShape([batch_size, 39*line_per_sample,]),
        'global_unique_id' : tf.TensorShape([unique_shape]),
        'global_unique_id_inver' : tf.TensorShape([batch_size, 39 * line_per_sample]),
        'global_unique_shape' : tf.TensorShape([None])
    }
    padding_values = {
        'label': tf.constant(0, dtype=tf.float32),
        'feat_ids': tf.constant(0, dtype=tf.int32),
        'feat_vals': tf.constant(0, dtype=tf.float32),
        'global_unique_id': tf.constant(-1, dtype=tf.int32),
        'global_unique_id_inver': tf.constant(0, dtype=tf.int32),
        'global_unique_shape':tf.constant(0, dtype=tf.int32)
    } 
    dataset = dataset.map(unique_fn, num_parallel_calls=16).padded_batch(1, padded_shapes=padded_shapes, padding_values=padding_values, drop_remainder=True).map(reshape_fn)
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)   
    return dataset 
#

	

def init_host_params_table(sess, config, chief_num, rank_id, host_params_shape=[200000, 256]): 
    host_params_address_table = tf.constant([[1,1,0]],dtype=tf.uint64)
    bar = tf.constant([[1,1,0]], dtype=tf.float32)
    # Graph 1
    # host_params_shape = [200000, 300]
    if 0 == rank_id:
        with npu_scope.npu_variable_scope(placement=NpuExecutePlacement.HOST):
            emb_table = np.random.uniform(-0.001, 0.001, size=host_params_shape[0] * 80).reshape([host_params_shape[0], 80])
            emb_opt_params = np.zeros([host_params_shape[0], 80 *2])
            emb_params = np.concatenate([emb_table, emb_opt_params], axis=1)

            weight_table = np.random.uniform(-0.001, 0.001, size=host_params_shape[0] * 1).reshape([host_params_shape[0], 1])
            weight_opt_params = np.zeros([host_params_shape[0], 2])
            weight_params = np.concatenate([weight_table, weight_opt_params], axis=1)

            pad_params = np.zeros([host_params_shape[0], host_params_shape[1] - 80*3 - 3])

            host_params = np.concatenate([emb_params, weight_params, pad_params], axis=1)
            # 
            host_params_table = tf.get_variable(name='host_params',
                                                shape=host_params_shape,  # [80, 80, 80, 1, 1, 1, 77]
                                                initializer=tf.constant_initializer(host_params),
                                                #initializer=tf.random_uniform_initializer(-0.001, 0.001),
                                                dtype=tf.float32)
        # 
        sess.run(util.variable_initializer_in_host([host_params_table ]))
        host_params_address_table = npu_plugin.get_var_addr_and_size("host_params")
    else:
        host_params_address_table = npu_plugin.malloc_shared_memory("host_params", host_params_shape, tf_adapter.DT_FLOAT)
        
    bar1 = hccl_ops.allreduce(bar, reduction='sum')
    sess.run(bar1)
    host_params_address_table = np.reshape(host_params_address_table, (1,2))
    
    # Graph 3
    # Create plane group, allgather host_params_table addr 
    group_name = "group" + str(rank_id)
    if chief_num == 2:
        create_group(group_name, chief_num, [rank_id, rank_id+8])
    elif chief_num == 4:
        create_group(group_name, chief_num, [rank_id, rank_id+8, rank_id+16, rank_id+24])
    rank = tf.reshape(tf.cast(get_rank_id(), dtype=tf.uint64), shape=[1,1])
    host_params_address_table = tf.concat([rank, host_params_address_table], axis=1)
    
    if chief_num != 1:
        host_params_address_table = hccl_ops.allgather(host_params_address_table, chief_num, group_name)
    # 
    host_params_address = sess.run(host_params_address_table)
    # print(' ------------- after broadcast:', rank_id, host_params_address)
    rank_size = get_rank_size()
    size = config["batch_size"] * rank_size * config["field_num"] * 300 * 4
    npu_plugin.rdma_remote_init(remote_var_list=[host_params_address], mem_size=size) # 
    return host_params_address
#  


def evaluate(sess, model, host_params_address, config, test_iterator, next_element):
    total_start_time = time.time()
    sess.run([test_iterator.initializer])
    # 
    # model.init_evaluate(test_input_data)

    log_loss_list = []
    pred_list = []
    label_list = []
    # 
    current_steps = 0
    finished = False
    while not finished:
        try:
            current_steps += 1
            test_batch_features = sess.run(next_element)
            feed_dict = {
                model.EVAL_labels : test_batch_features["label"].reshape(-1), 
                model.EVAL_wts : test_batch_features["feat_vals"].reshape(-1,39), 
                model.EVAL_unique_id : test_batch_features["global_unique_id"], 
                model.EVAL_unique_id_inver : test_batch_features["global_unique_id_inver"].reshape(-1,39), 
                model.EVAL_unique_shape : test_batch_features["global_unique_shape"], 
                model.host_params_address : host_params_address,
            }
            run_dict = {
                "EVAL_preds": model.EVAL_preds,
                "EVAL_log_loss": model.EVAL_log_loss,
            }
            results = sess.run(fetches=run_dict, feed_dict=feed_dict)
            log_loss_list.extend(results["EVAL_log_loss"])
            pred_list.extend(results["EVAL_preds"])
            label_list.extend(test_batch_features["label"])
        except tf.errors.OutOfRangeError as e:
            finished = True

    auc = roc_auc_score(label_list, pred_list)
    mean_log_loss = np.mean(log_loss_list)
    return auc, mean_log_loss
# 

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # dataset
    parser.add_argument('--data_dir', default='/autotest/data',
                        help="""directory of dataset.""")

    parser.add_argument('--max_epochs', default=10, type=int,
                        help="""total epochs for training""")

    parser.add_argument('--display_every', default=100, type=int,
                        help="""the frequency to display info""")

    parser.add_argument('--batch_size', default=16000, type=int,
                        help="""batch size for one NPU""")

    parser.add_argument('--max_steps', default=15000, type=int,
                        help="""max train steps""")

    # model file
    parser.add_argument('--model_dir', default='./model',
                        help="""log directory""")
 
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args

if __name__ == '__main__':

    args = parse_args()

    npu_int = npu_ops.initialize_system()
    npu_shutdown = npu_ops.shutdown_system()

    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["master_slave_mode"].b = True
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('allow_mix_precision')
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    #custom_op.parameter_map["enable_data_pre_proc"].b = True
    sess_config.gpu_options.allow_growth = True
    custom_op.parameter_map["hcom_parallel"].b = True

    

    init_sess = tf.Session(config=sess_config)

    # NPU Init Sess
    init_sess.run(npu_int)
    
    global_start_time = time.time()
    
    rank_size = get_rank_size()
    local_rank_size = get_local_rank_size()
    chief_num =  rank_size // local_rank_size
    rank_id = get_local_rank_id()
    print('rank_size:', rank_size)
    # 

    config = {
        "output_path": args.model_dir,
        "data_path": args.data_dir,
        
        "train_file_pattern": "train",
        "test_file_pattern": "test",
        "batch_size": args.batch_size,
        
        "field_num": 39,
        "id_emb_dim": 80,
        "host_params_shape": [200000, 256],

        "deep_layer_args": [[1024, 512, 256, 256], "relu"], 
        "reg_args": [0.7, 5e-6],
        "opt_args": ['lazyadam', 5e-5 * rank_size, 5e-7],
        "reg_loss_mode": "batch",

        "train_epoch": 20,

        "train_size": 41257636,
        "test_size": 4582981,
    }
    #

    if not os.path.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    
    optimizer_array = [['lazyadam', 5e-5 * rank_size, 5e-7], ['ftrl', 1e-2, 1, 1e-8, 1e-8]]
    with tf.device('/cpu:0'):
        np.random.seed(1234)
        tf.set_random_seed(1234)
        train_dataset = input_fn_tfrecord(data_path=config["data_path"],
                                          file_pattern="train", # config["train_file_pattern"], 
                                          batch_size=int(rank_size * config["batch_size"]),
                                          perform_shuffle=True, 
                                          num_epochs=config["train_epoch"])
        train_iterator = train_dataset.make_initializable_iterator()
        train_next_iter = train_iterator.get_next()
        train_input_data = {"global_labels" : tf.reshape(train_next_iter["label"], [-1]),
                            "global_wts" : tf.reshape(train_next_iter["feat_vals"], [-1, 39]),
                            "global_unique_id" : train_next_iter["global_unique_id"], 
                            "global_unique_id_inver" : train_next_iter["global_unique_id_inver"],
                            "global_unique_shape" : train_next_iter["global_unique_shape"] }
        test_dataset = input_fn_tfrecord(data_path=config["data_path"],
                                        file_pattern=config["test_file_pattern"], 
                                        batch_size=int(config["batch_size"]),
                                        perform_shuffle=False, 
                                        num_epochs=1)
        test_iterator = test_dataset.make_initializable_iterator()
        test_next_iter = test_iterator.get_next()
    # 
    model = WideDeep([200000, 39],
                     [80, [1024, 512, 256, 128], 'relu'],
                     [['adam', 3e-4, 9e-8], ['ftrl', 3.5e-2, 1, 3e-8, 1e-6]],
                     [1.0, 9e-6],
                     train_input_data,
                     config["host_params_shape"],
                     loss_mode="none"
                     )
           
    model.init_evaluate()
    # 

    print_steps = args.display_every
    evaluate_steps = int(config["train_size"] // config["batch_size"]) // 5
    stop_steps = args.max_steps
    
    with tf.Session(config=sess_config) as sess:
        sess.run([train_iterator.initializer])
        # sess.run(tf.variables_initializer(model.need_init_varlist()) )
        sess.run(tf.global_variables_initializer())


        print("=========hhhh============")
        host_params_address = init_host_params_table(sess, config, chief_num, rank_id, host_params_shape=config["host_params_shape"])

        total_start_time = time.time()
        
        current_steps = 0
        train_finished = False
        while not train_finished:
            try:
                current_steps += 1
                run_dict = {
                    "train_op": model.train_op,
                    "deep_loss": model.deep_loss,
                    "wide_loss": model.wide_loss,
                    "train_preds": model.train_preds,
                }
                
                feed_dict = { model.host_params_address: host_params_address, }
                start_time = time.time()
                results = sess.run(fetches=run_dict, feed_dict=feed_dict)
                end_time = time.time()
                step_time = end_time-start_time
                fps = rank_size * config["batch_size"] / step_time

                if current_steps % print_steps == 0:
                    print( "----------" * 10 )
                    print( "current_steps: {}, deep_loss: {}, step_time: {}, fps: {}".format(current_steps, results["deep_loss"], step_time, fps) )
                    print( "----------" * 10 )
                
                if current_steps % evaluate_steps == 0:
                    test_auc, test_mean_log_loss = evaluate(sess, model, host_params_address, config, test_iterator, test_next_iter)
                    print( "current_steps: {}; log_loss: {}; Test_auc: {} ".format(current_steps, test_mean_log_loss, test_auc) )
                if current_steps >= stop_steps:
                    train_finished = True
                
            except tf.errors.OutOfRangeError as e:
                train_finished = True
        print( "training {} steps, consume time: {} ".format( current_steps, time.time() - total_start_time ) )   
        tf.train.write_graph(sess.graph, config["output_path"], 'widedeep_graph.pbtxt', as_text=True)
    # 
# 







