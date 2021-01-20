"""DeepFM related model"""
from __future__ import print_function

import pickle
import tensorflow as tf
from tf_util import build_optimizer, init_var_map, \
    get_field_index, get_field_num, split_mask, split_param, sum_multi_hot, \
    activate
import config as config
#import horovod.tensorflow as hvd
from hccl.manage.api import get_local_rank_id
from hccl.manage.api import get_rank_size
from hccl.manage.api import get_rank_id
from npu_bridge.hccl import hccl_ops
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager

#modify dropout
from npu_bridge.estimator import npu_ops

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

class FMNN_v2:
    """support performing mean pooling Operation on multi-hot feature.
    dataset_argv: e.g. [10000, 17, [False, ...,True,True], 10]
    """
    def __init__(self,  dataset_argv, architect_argv, init_argv, ptmzr_argv,
                 reg_argv, _input_data, loss_mode='batch', merge_multi_hot=False,
                 cross_layer=False, batch_norm=False):
        # self.graph = tf.Graph()
        #with tf.variable_scope('FMNNMG') as scope:
        (num_comm_feat, self.dim_comm,
         self.multi_hot_flags, self.multi_hot_len, self.num_unique_feat, self.dim_unique) = dataset_argv
        self.one_hot_flags = [not flag for flag in self.multi_hot_flags]
        embed_size, layer_dims, act_func = architect_argv
        keep_prob, _lambda = reg_argv
        #self.ptmzr_argv=ptmzr_argv
        self.num_onehot = sum(self.one_hot_flags)
        self.num_multihot = sum(self.multi_hot_flags) / self.multi_hot_len
        self.num_fields = self.dim_comm + self.dim_unique
        print(self.num_fields)
        input_dim4lookup = self.num_fields
        if merge_multi_hot:
            self.embed_dim = (self.num_multihot +
                              self.num_onehot) * embed_size
            self.num_fields = self.num_multihot + self.num_onehot
        else:
            self.embed_dim = input_dim4lookup * embed_size
        self.all_layer_dims = [self.embed_dim + 1] + layer_dims + [1]
        self.log = ('input dim: %d\nnum inputs: %d\nembed size(each): %d\n'
                    'embedding layer: %d\nlayers: %s\nactivate: %s\n'
                    'keep_prob: %g\nl2(lambda): %g\nmerge_multi_hot: %s\n' %
                    (num_comm_feat, num_comm_feat +self.num_unique_feat, embed_size,
                     self.embed_dim, self.all_layer_dims, act_func,
                     keep_prob, _lambda, merge_multi_hot))
        initializer_range=0.02
        self.local_v = tf.get_variable('local_V', shape=[self.num_unique_feat, embed_size],
                                    #initializer=tf.random_uniform_initializer(init_argv[1], init_argv[2]),
                                    #initializer=tf.zeros_initializer,
                                    initializer=self.create_initializer(initializer_range),
                                    dtype=tf.float32)
        self.local_w = tf.get_variable('local_W', shape=[self.num_unique_feat, 1],
                                    #initializer=tf.random_uniform_initializer(init_argv[1], init_argv[2]),
                                    #initializer=tf.zeros_initializer,
                                    initializer=self.create_initializer(initializer_range),
                                    dtype=tf.float32)

        self.fm_v = tf.get_variable('V', shape=[num_comm_feat, embed_size],
                                    #initializer=tf.random_uniform_initializer(init_argv[1], init_argv[2]),
                                    #initializer=tf.zeros_initializer,\
                                    initializer=self.create_initializer(initializer_range),
                                    dtype=tf.float32)
        self.fm_w = tf.get_variable('W', shape=[num_comm_feat, 1],
                                    #initializer=tf.random_uniform_initializer(init_argv[1], init_argv[2]),
                                    #initializer=tf.zeros_initializer,
                                    initializer=self.create_initializer(initializer_range),
                                    dtype=tf.float32)
        #self.fm_b = tf.get_variable('b', shape=[1], initializer=tf.zeros_initializer, dtype=tf.float32)
        self.fm_b = tf.get_variable('b', shape=[1], initializer=self.create_initializer(initializer_range), dtype=tf.float32)
        self.h_w, self.h_b = [], []
        for i in range(len(self.all_layer_dims) - 1):
            self.h_w.append(tf.get_variable('h%d_w' % (i + 1), shape=self.all_layer_dims[i: i + 2],
                                            #initializer=tf.random_uniform_initializer(init_argv[1], init_argv[2]),
                                            #initializer=tf.zeros_initializer,
                                            initializer=self.create_initializer(initializer_range),
                                            dtype=tf.float32))
            self.h_b.append(
                tf.get_variable('h%d_b' % (i + 1), shape=[self.all_layer_dims[i + 1]],
                                #initializer=tf.zeros_initializer,
                                initializer=tf.random_uniform_initializer(-0.05, 0.05),
                                dtype=tf.float32))

        self.lbl_hldr = tf.placeholder(tf.float32, shape=[None])
                
        # run unique embedding lookup feed unique id
        self.id_hldr_uniq_mpi = tf.placeholder(tf.int32, shape=[None], name='id_unique')
        self.t_embedding_uniq = tf.gather(self.local_v, self.id_hldr_uniq_mpi)
        self.t_w_uniq = tf.gather(self.local_w, self.id_hldr_uniq_mpi)
        '''
        self.wt_hldr_uniq = tf.placeholder(tf.float32, shape=[None, self.dim_unique], name='wt_unique')
        self.id_hldr_uniq = tf.placeholder(tf.int64, shape=[None, self.dim_unique], name='id_unique')
        mask_uniq = self.wt_hldr_uniq
        id_hldr_uniq = self.id_hldr_uniq
        
        self.id_hldr_uniq_mpi = tf.placeholder(tf.int64, shape=[None], name='id_unique')
        self._embedding_uniq = tf.gather(self.local_v, self.id_hldr_uniq_mpi)
        
        self._w_uniq = tf.gather(self.local_w, self.id_hldr_uniq_mpi)
        # run forward & backward feed embedding directly
        # reverse embedding with mask by gather
        self._embed_data = tf.gather(self.common_embedding, self.id_uniq_inver)
        self._W_data = tf.gather(self.common_w, self.id_uniq_inver)
        # run forward & backward feed embedding directly
        self._embed_data = tf.placeholder(tf.float32, shape=[None, self.dim_unique, embed_size], name='V_feed')
        self._W_data = tf.placeholder(tf.float32, shape=[None,  self.dim_unique], name='w_feed')

        self.embedding_uniq_ = tf.Variable(initial_value=self._embed_data, trainable=True, validate_shape=False, name='V_unique')
        self.W_uniq_  = tf.Variable(initial_value=self._W_data, trainable=True, validate_shape=False, name='w_unique')
        '''
        # in graph
        # sample  40 {10 sparse ; 30 dense}
        # None: batch_size
        # batch_indices ... >????
        # [B,S]
        self.wt_hldr_uniq = tf.placeholder(tf.float32, shape=[None, self.dim_unique], name='wt_unique')# 当前节点的数据value  dim_unique 需要被分割的数量，sparse
        # [N*B,S]
        self.id_hldr_uniq_all = tf.placeholder(tf.int32, shape=[None, self.dim_unique], name='id_unique_all')# 所有节点需要被分割的id
        # [B]
        self.batch_indices = tf.placeholder(tf.int32, shape=[None], name='batch_indices')# np.range�?list 当前节点的数据在all里面的行**   �?weight 行一�?        # 当前节点要用到的样本是在 ，哪些行是自己的�?
        # 因为前面是dense 后面是sparse ，所以要讲sparse的id 从零开�?# 目前前面是dense 后面是sparse ，拉�?        # [B,S]->[B*S]  ;  [?] [B*S]
        with ops.get_default_graph()._kernel_label_map({"Unique": "parallel"}):
            self.id_uniq_u_, self.id_uniq_inver_all_ = tf.unique(tf.reshape((self.id_hldr_uniq_all-config.common_feat_num), [-1]), out_idx=tf.dtypes.int32)

        # [?] ;  [?]
        #self.id_uniq_u_, self.id_uniq_inver_all_ = tf.unique(tf.reshape(self.id_hldr_uniq_all, [-1]), out_idx=tf.dtypes.int32)
        self.id_uniq_u = self.id_uniq_u_ # global unique

        # general_dim ... >   [N*B*S] ; [N*B，S]
        self.id_uniq_inver_all = tf.reshape(self.id_uniq_inver_all_, [-1, config.general_dim]) # *** global unique
        # batch_indices ... >   [N*B，S] ; [B，S]
        self.id_uniq_inver_ori = tf.gather(self.id_uniq_inver_all, indices=self.batch_indices)
        # batch_indices ...   [B，S] ; [B，S] ***改过
        self.id_uniq_inver = tf.reshape(self.id_uniq_inver_ori, [-1, self.dim_unique])
        self.id_uniq_local_mask = tf.where(tf.equal(tf.cast(get_rank_id(), dtype=tf.int32), self.id_uniq_u%get_rank_size())) #动态的****看是否是本地
        #sen
        #self.indices_global = tf.reshape(self.id_uniq_local_mask, [-1])
        self.id_uniq_local = tf.gather_nd(self.id_uniq_u, self.id_uniq_local_mask) #动态的 ，可以用gather V2 LOCAL mask 表示 id_)uniq 的哪个位�?        
        self.id_uniq_local_ = tf.reshape(self.id_uniq_local, [-1, 1])# 动态的 转换local的id�?        
        self.id_uniq_owner = self.id_uniq_local // get_rank_size() # 动态的 转换local的id�?    
        #sen
        #self.indices_local = tf.reshape(self.id_uniq_owner, [-1])    
        self.index_all = tf.reshape(tf.range(0,tf.shape(self.id_uniq_u)[0]), [-1,1])
        index_all = self.index_all
        self.uniq_index = self.id_uniq_local_mask #tf.gather_nd(index_all, self.id_uniq_local_mask)
        #self.uniq_index = tf.gather_nd(index_all, self.id_uniq_local_mask)

        #self.uniq_index = self.id_uniq_local_mask
        mask_uniq = self.wt_hldr_uniq
        # uniquq lookup
        #[local?] [voc,E]�?[local?,E]
        self._embedding_uniq = tf.gather(self.local_v, self.id_uniq_owner)
        #[local?] [voc,E]�?[local?,E]
        self._w_uniq = tf.gather(self.local_w, self.id_uniq_owner)
        # scatter add at global unique
        self.fix_shape = 16000*40 #tf.shape(self.id_hldr_uniq_all)[0] * self.dim_unique
        #self.fix_shape = tf.shape(self.id_hldr_uniq_all)[0] * self.dim_unique
        self.common_embedding = tf.scatter_nd(self.uniq_index, self._embedding_uniq, shape=[self.fix_shape, embed_size])
        self.common_w = tf.scatter_nd(self.uniq_index, self._w_uniq, shape=[self.fix_shape, 1])

        #[local?]  [local?,E] �?[?,E]
        #self.common_embedding = tf.scatter_nd(self.uniq_index, self._embedding_uniq, shape=[tf.shape(self.id_uniq_u)[0], embed_size])
        #[local?]  [local?,E] �?[?,E]
        #self.common_w = tf.scatter_nd(self.uniq_index, self._w_uniq, shape=[tf.shape(self.id_uniq_u)[0], 1])
        #self.common_embedding = tf.scatter_nd(self.id_uniq_local_2, self._embedding_uniq, shape=[tf.shape(self.id_uniq_u)[0], embed_size])
        #self.common_w = tf.scatter_nd(self.id_uniq_local_, self._w_uniq, shape=[tf.shape(self.id_uniq_u)[0], 1])
        # [?,E] ; [?*E]
        self.common_w = tf.reshape(self.common_w, [-1])
        # communicate with other node -- ring-allreduce
        if get_rank_size() > 1:
             self.common_embedding = hccl_ops.allreduce(self.common_embedding, reduction='sum', fusion=0)# average=False)
             self.common_w = hccl_ops.allreduce(self.common_w,reduction='sum', fusion=0)#average=False)

        #self.common_embedding = self.common_embedding
        #self.common_w = self.common_w

        # run forward & backward feed embedding directly
        # reverse embedding with mask by gather
        # [?,E] [B,S] ;[B,S,E]
        self._embed_data = tf.gather(self.common_embedding, self.id_uniq_inver)
        #self.id_uniq_inver =[10]*10000
        self._W_data = tf.gather(self.common_w, self.id_uniq_inver)
        # transfer variable for back gradient

        #  [B,S,E] ;   [B,S,E]
        #self.embedding_uniq_ = tf.Variable(initial_value=self._embed_data, trainable=True, validate_shape=False, name='V_unique')
        #self.W_uniq_  = tf.Variable(initial_value=self._W_data, trainable=True, validate_shape=False, name='w_unique')
        self.embedding_uniq_ = self._embed_data
        self.W_uniq_ = self._W_data
        #---------------------------------

        #print('embedding_uniq', (tf.gather(self.local_w, id_hldr_uniq)))

        # embedding multiply mask
        self.embedding_uniq = tf.multiply(self.embedding_uniq_, tf.expand_dims(mask_uniq,2))
        self.W_uniq = tf.multiply(self.W_uniq_, mask_uniq)
        
        self.wt_hldr_dense = tf.placeholder(tf.float32, shape=[None, self.dim_comm], name='wt_dense')
        self.id_hldr_dense = tf.placeholder(tf.int32, shape=[None, self.dim_comm], name='id_dense')
        mask_dense = self.wt_hldr_dense
        id_hldr_dense = self.id_hldr_dense
        mask = tf.expand_dims(mask_dense, 2)
        self.embedding_dense = tf.multiply(tf.gather(self.fm_v, self.id_hldr_dense), mask)
        self.W_dense = tf.multiply(tf.gather(self.fm_w, self.id_hldr_dense), mask)
        
        self.embedding = tf.concat([self.embedding_dense,self.embedding_uniq], axis=1)
        self.W = tf.concat([self.W_dense, tf.expand_dims(self.W_uniq, 2)], axis=1)

        #self.embedding_uniq = tf.multiply(tf.gather(self.local_v, self.id_hldr_uniq), tf.expand_dims(self.wt_hldr_uniq,2))
        #self.W_uniq = tf.multiply(tf.gather(self.local_w, self.id_hldr_uniq), tf.expand_dims(self.wt_hldr_uniq,2))
        
        #self.embedding = tf.concat([self.embedding_dense,self.embedding_uniq], axis=1)
        #self.W = tf.concat([self.W_dense, self.W_uniq], axis=1)
        
        logits = self.forward_cut( self.embedding, self.W,
            act_func, keep_prob, training=True,
            merge_multi_hot=merge_multi_hot, cross_layer=cross_layer, 
            batch_norm=batch_norm)
       
        log_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=self.lbl_hldr)
        self.log_loss = tf.reduce_mean(log_loss)
        self.train_preds = tf.sigmoid(logits, name='predictions')
        
        if loss_mode == 'full':
            self.l2_loss = _lambda * (tf.nn.l2_loss(self.fm_w) +
                                      tf.nn.l2_loss(self.fm_v) +
                                      tf.nn.l2_loss(self.local_w) + 
                                      tf.nn.l2_loss(self.local_v))
        else:  # 'batch'
            self.l2_loss = _lambda * (tf.nn.l2_loss(self.W) +
                                      tf.nn.l2_loss(self.embedding))
        self.loss = self.log_loss + self.l2_loss
        
        #self.loss = self.log_loss
        #self.init_optimizer(self.loss, self.log_loss)
        #for unique grad
        self.grad_embed_unique = tf.placeholder(tf.float32, shape = [None, embed_size], name='grad_embed_unique')
        self.grad_w_unique = tf.placeholder(tf.float32, shape = [None,1], name='grad_w_unique')
        
        # for eval
        # run unique embedding lookup feed unique id
        # for simulate
        
        # for segmentsum
        self.grad_embed_values_seg = tf.placeholder(tf.float32, shape=[None, 80], name='embedding_grad')
        self.grad_w_values_seg = tf.placeholder(tf.float32, shape=[None,1], name='w_grad')

        self.indices = tf.placeholder(tf.int32, shape = [None], name='indices_grad')
        self.unique_size = tf.placeholder(tf.int32,  name = 'size_unique')
        self.grad_embed_values_sum = tf.unsorted_segment_sum(self.grad_embed_values_seg, self.indices, self.unique_size)
        self.grad_w_values_sum = tf.unsorted_segment_sum(self.grad_w_values_seg, self.indices, self.unique_size)
        
        self.wt_eval_hldr_dense = tf.placeholder(tf.float32, shape=[None, self.dim_comm], name='eval_wt_dense')
        self.id_eval_hldr_dense = tf.placeholder(tf.int32, shape=[None, self.dim_comm], name='eval_id_dense')
        eval_mask_dense = self.wt_eval_hldr_dense
        eval_id_hldr_dense = self.id_eval_hldr_dense
        eval_mask = tf.expand_dims(eval_mask_dense, 2)
        self.eval_embedding_dense = tf.multiply(tf.gather(self.fm_v, self.id_eval_hldr_dense), eval_mask)
        self.eval_W_dense = tf.multiply(tf.gather(self.fm_w, self.id_eval_hldr_dense), eval_mask)
        
        self.wt_eval_hldr_uniq = tf.placeholder(tf.float32, shape=[None, self.dim_unique], name='eval_wt_unique')
        # hvd version
        self.eval_id_hldr_uniq_all = tf.placeholder(tf.int32, shape=[None, self.dim_unique], name='eval_id_unique_all')
        with ops.get_default_graph()._kernel_label_map({"Unique": "parallel"}):
            self.eval_id_uniq_u_, self.eval_id_uniq_inver_all_ = tf.unique(tf.reshape((self.eval_id_hldr_uniq_all-config.common_feat_num), [-1]), out_idx=tf.dtypes.int32)
        #self.id_uniq_u_, self.id_uniq_inver_all_ = tf.unique(tf.reshape(self.id_hldr_uniq_all, [-1]), out_idx=tf.dtypes.int32)
        self.eval_id_uniq_u = self.eval_id_uniq_u_
        self.eval_id_uniq_inver_all = tf.reshape(self.eval_id_uniq_inver_all_, [-1, config.general_dim])
        self.eval_id_uniq_inver_ori = self.eval_id_uniq_inver_all #tf.gather(self.id_uniq_inver_all, indices=self.batch_indices)
        self.eval_id_uniq_inver = tf.reshape(self.eval_id_uniq_inver_ori, [-1, self.dim_unique])
        self.eval_id_uniq_local_mask = tf.where(tf.equal(tf.cast(get_rank_id(), dtype=tf.int32), self.eval_id_uniq_u%get_rank_size()))
        #print(self.id_uniq_local_mask)
        #for i in range(10):
        #    print("hahahahahahahah")
        self.eval_id_uniq_local = tf.gather_nd(self.eval_id_uniq_u, self.eval_id_uniq_local_mask)
        self.eval_id_uniq_local_ = tf.reshape(self.eval_id_uniq_local, [-1, 1])
        self.eval_id_uniq_owner = self.eval_id_uniq_local // get_rank_size()
        self.eval_index_all = tf.reshape(tf.range(0,tf.shape(self.eval_id_uniq_u)[0]), [-1,1])
        eval_index_all = self.eval_index_all
        self.eval_uniq_index = tf.gather_nd(eval_index_all, self.eval_id_uniq_local_mask)
        eval_mask_uniq = self.wt_eval_hldr_uniq
        # uniquq lookup
        self._eval_embedding_uniq = tf.gather(self.local_v, self.eval_id_uniq_owner)
        self._eval_w_uniq = tf.gather(self.local_w, self.eval_id_uniq_owner)
        # scatter add at global unique
        #self.eval_common_embedding = tf.scatter_nd(self.eval_uniq_index, self._eval_embedding_uniq, shape=[tf.shape(self.eval_id_uniq_u)[0], embed_size])
        #self.eval_common_w = tf.scatter_nd(self.eval_uniq_index, self._eval_w_uniq, shape=[tf.shape(self.eval_id_uniq_u)[0], 1])
        
        self.eval_common_embedding = tf.scatter_nd(self.eval_uniq_index, self._eval_embedding_uniq, shape=[self.fix_shape, embed_size])
        self.eval_common_w = tf.scatter_nd(self.eval_uniq_index, self._eval_w_uniq, shape=[self.fix_shape, 1])
        
        #self.common_embedding = tf.scatter_nd(self.id_uniq_local_2, self._embedding_uniq, shape=[tf.shape(self.id_uniq_u)[0], embed_size])
        #self.common_w = tf.scatter_nd(self.id_uniq_local_, self._w_uniq, shape=[tf.shape(self.id_uniq_u)[0], 1])
        self.eval_common_w = tf.reshape(self.eval_common_w, [-1])
        # communicate with other node -- ring-allreduce
        if get_rank_size() > 1:
             self.eval_common_embedding = hccl_ops.allreduce(self.eval_common_embedding,  reduction='sum', fusion=0)#average=False)
             self.eval_common_w = hccl_ops.allreduce(self.eval_common_w,  reduction='sum', fusion=0)#average=False)

        ###singel p
        #self.eval_common_embedding = self.eval_common_embedding
        #self.eval_common_w = self.eval_common_w

        # run forward & backward feed embedding directly
        # reverse embedding with mask by gather
        self._eval_embed_data = tf.gather(self.eval_common_embedding, self.eval_id_uniq_inver)
        self._eval_W_data = tf.gather(self.eval_common_w, self.eval_id_uniq_inver)

        # mpi version
        '''
        self.id_eval_hldr_uniq = tf.placeholder(tf.int64, shape=[None, self.dim_unique], name='eval_id_unique')
        eval_mask_uniq = self.wt_eval_hldr_uniq
        eval_id_hldr_uniq = self.id_eval_hldr_uniq
        # for unique 
        #self.wt_eval_hldr_uniq = tf.placeholder(tf.float32, shape=[None, ], name='eval_wt_unique')
        #self.id_eval_hldr_uniq = tf.placeholder(tf.int64, shape=[None, ], name='eval_id_unique')
        #eval_mask_uniq = tf.reshape(self.wt_eval_hldr_uniq, [1, -1])
        #eval_id_hldr_uniq = tf.reshape(self.id_eval_hldr_uniq,[1, -1])
        # for mpi
        self.eval_embedding_uniq_ = tf.gather(self.local_v, eval_id_hldr_uniq)
        self.eval_W_uniq_ = tf.gather(self.local_w, eval_id_hldr_uniq)
        

        self.eval_hldr_embed_uniq = tf.placeholder(tf.float32, shape=[None, self.dim_unique, embed_size], name='eval_hldr_embed_unique')
        self.eval_hldr_w_uniq = tf.placeholder(tf.float32, shape=[None, self.dim_unique], name='eval_hldr_w__unique')
        '''
        eval_mask_uniq = self.wt_eval_hldr_uniq
        self.eval_embedding_uniq = tf.multiply(self._eval_embed_data, tf.expand_dims(eval_mask_uniq, 2))
        self.eval_W_uniq = tf.multiply(self._eval_W_data, self.wt_eval_hldr_uniq)
        # run forward & backward feed embedding directly

        
        self.eval_embedding = tf.concat([self.eval_embedding_dense, self.eval_embedding_uniq], axis=1)
        self.eval_W = tf.concat([self.eval_W_dense, tf.reshape(self.eval_W_uniq, [-1,  self.dim_unique, 1])], axis=1)
        
        # for serial
        #self.eval_embedding_uniq = tf.multiply(tf.gather(self.local_v, eval_id_hldr_uniq), tf.expand_dims(eval_mask_uniq,2))
        #self.eval_W_uniq = tf.multiply(tf.gather(self.local_w, eval_id_hldr_uniq), tf.expand_dims(eval_mask_uniq,2))
        
        #print(self.eval_W_uniq,'aaaa')

        #self.eval_embedding = tf.concat([self.eval_embedding_dense, self.eval_embedding_uniq],axis=1)
        #self.eval_W = tf.concat([self.eval_W_dense, self.eval_W_uniq], axis=1)

        eval_logits = self.forward_cut( self.eval_embedding, 
            self.eval_W, act_func, keep_prob,
            training=False, merge_multi_hot=merge_multi_hot,
            cross_layer=cross_layer, batch_norm=batch_norm)
        self.eval_preds = tf.sigmoid(eval_logits, name='predictionNode')
        self.eval_preds = tf.clip_by_value(self.eval_preds, 1e-6, 1.0-1e-6, name = 'predictionNode')
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.ptmzr, log = build_optimizer(ptmzr_argv, self.loss)
        # self.log += log
    def create_initializer(self,initializer_range=0.02):
        """Creates a `truncated_normal_initializer` with the given range."""
        return tf.truncated_normal_initializer(stddev=initializer_range)
    def init_optimizer(self, deep_loss, wide_loss):
        self.opt_deep, self.lr_deep, self.eps_deep,_1 = self.ptmzr_argv[0]
        #opt_wide, wide_lr, wide_dc, wide_l1, wide_l2 = self.ptmzr_argv[1]
        # #
        # self.wide_ptmzr = tf.train.FtrlOptimizer(learning_rate=wide_lr, initial_accumulator_value=wide_dc,
        #                                     l1_regularization_strength=wide_l1,
        #                                     l2_regularization_strength=wide_l2)
        # self.wide_update = self.wide_ptmzr.minimize(wide_loss, var_list=tf.get_collection("wide"))
        #
        if self.opt_deep == "adam":
            self.deep_optimzer = tf.train.AdamOptimizer(learning_rate=self.lr_deep, epsilon=self.eps_deep)
        elif self.opt_deep == "lazyadam":
            self.deep_optimzer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.lr_deep, epsilon=self.eps_deep)
        #
        loss_scale_manager = FixedLossScaleManager(loss_scale=1000.0)
        self.deep_optimzer = NPULossScaleOptimizer(self.deep_optimzer, loss_scale_manager)
        self.deep_update = self.deep_optimzer.minimize(deep_loss)
        self.train_op = self.deep_update    

    def forward_cut(self, embedding, W, act_func, keep_prob,
                training, merge_multi_hot=False,
                cross_layer=False, batch_norm=False):

        linear_out = tf.reduce_sum(W, 1)
        b = tf.reshape(tf.tile(tf.identity(self.fm_b),tf.shape(embedding)[0:1]),[-1, 1])
        v2x2 = tf.square(embedding)
        vx2 = tf.square(tf.reduce_sum(embedding, 1))
        vxs = tf.reduce_sum(v2x2, 1)
        fm_out = 0.5 * tf.reduce_sum(vx2 - vxs, 1)

        hidden_output = tf.concat(
            [tf.reshape(embedding, [-1, self.embed_dim]), b], axis=1)
        cross_layer_output = None
        for i in range(len(self.h_w)):
            if training:
                hidden_output = tf.matmul(
                    npu_ops.dropout(
                        activate(act_func, hidden_output), keep_prob=keep_prob),
                    self.h_w[i]) + self.h_b[i]
                #hidden_output = tf.matmul(
                #    activate(act_func, hidden_output),
                #    self.h_w[i]) + self.h_b[i]
            else:
                hidden_output = tf.matmul(
                    activate(act_func, hidden_output),
                    self.h_w[i]) + self.h_b[i]
            # if batch_norm:
            #     hidden_output = tf.layers.batch_normalization(
            #         hidden_output, training=training)
            if cross_layer_output is not None:
                cross_layer_output = tf.concat(
                    [cross_layer_output, hidden_output], 1)
            else:
                cross_layer_output = hidden_output

            if cross_layer and i == len(self.h_w) - 2:
                hidden_output = cross_layer_output

        return tf.reshape(hidden_output, [-1, ]) + \
            tf.reshape(fm_out, [-1, ]) + \
            tf.reshape(linear_out, [-1, ])

    def forward(self, wt_hldr, id_hldr, act_func, keep_prob,
                training, merge_multi_hot=False,
                cross_layer=False, batch_norm=False):
        mask = tf.expand_dims(wt_hldr, 2)
        if merge_multi_hot and self.num_multihot > 0:
            one_hot_mask, multi_hot_mask = split_mask(
                mask, self.multi_hot_flags, self.num_multihot)
            one_hot_w, multi_hot_w = split_param(
                self.fm_w, id_hldr, self.multi_hot_flags)
            one_hot_v, multi_hot_v = split_param(
                self.fm_v, id_hldr, self.multi_hot_flags)

            # linear part
            multi_hot_wx = sum_multi_hot(
                multi_hot_w, multi_hot_mask, self.num_multihot)
            one_hot_wx = tf.multiply(one_hot_w, one_hot_mask)
            wx = tf.concat([one_hot_wx, multi_hot_wx], axis=1)

            # fm part (reduce multi-hot vector's length to k*1)
            multi_hot_vx = sum_multi_hot(
                multi_hot_v, multi_hot_mask, self.num_multihot)
            one_hot_vx = tf.multiply(one_hot_v, one_hot_mask)
            vx_embed = tf.concat([one_hot_vx, multi_hot_vx], axis=1)
        else:
            # [batch, input_dim4lookup, 1]
            # wx = tf.multiply(tf.gather(self.fm_w, id_hldr), mask)
            uid = tf.mod(tf.reshape(id_hldr[:,-1], [-1,1]), self.num_unique_feat)
            #uid = tf.reshape(id_hldr[:,0], [-1,1])
            wx = tf.multiply(tf.concat(
                [tf.gather(self.local_w, uid),
                    tf.gather(self.fm_w, id_hldr[:,1:self.num_fields])], axis=1), mask)
            # [batch, input_dim4lookup, embed_size]
            #vx_embed = tf.multiply(tf.gather(self.fm_v, id_hldr), mask)
            print(id_hldr[:,0], tf.mod(tf.reshape(id_hldr[:,0], [-1,1]), self.num_unique_feat))
            vx_embed = tf.multiply(tf.concat(
                [tf.gather(self.local_v, uid),
                    tf.gather(self.fm_v, id_hldr[:,:self.num_fields-1])], axis=1), mask)

        linear_out = tf.reduce_sum(wx, 1)
        b = tf.reshape(tf.tile(tf.identity(self.fm_b),
                               tf.shape(wt_hldr)[0:1]),
                       [-1, 1])
        v2x2 = tf.square(vx_embed)
        vx2 = tf.square(tf.reduce_sum(vx_embed, 1))
        vxs = tf.reduce_sum(v2x2, 1)
        fm_out = 0.5 * tf.reduce_sum(vx2 - vxs, 1)

        hidden_output = tf.concat(
            [tf.reshape(vx_embed, [-1, self.embed_dim]), b], axis=1)
        cross_layer_output = None
        for i in range(len(self.h_w)):
            if training:
                hidden_output = tf.matmul(
                    npu_ops.dropout(#tf.nn.dropout(
                        activate(act_func, hidden_output), keep_prob=keep_prob),
                    self.h_w[i]) + self.h_b[i]
            else:
                hidden_output = tf.matmul(
                    activate(act_func, hidden_output),
                    self.h_w[i]) + self.h_b[i]
            # if batch_norm:
            #     hidden_output = tf.layers.batch_normalization(
            #         hidden_output, training=training)
            if cross_layer_output is not None:
                cross_layer_output = tf.concat(
                    [cross_layer_output, hidden_output], 1)
            else:
                cross_layer_output = hidden_output

            if cross_layer and i == len(self.h_w) - 2:
                hidden_output = cross_layer_output

        return tf.reshape(hidden_output, [-1, ]) + \
            tf.reshape(fm_out, [-1, ]) + \
            tf.reshape(linear_out, [-1, ]), wx, vx_embed

    def dump(self, model_path):
        var_map = {'W': self.fm_w.eval(),
                   'V': self.fm_v.eval(),
                   'b': self.fm_b.eval()}

        for i, (h_w_i, h_b_i) in enumerate(zip(self.h_w, self.h_b)):
            var_map['h%d_w' % (i+1)] = h_w_i.eval()
            var_map['h%d_b' % (i+1)] = h_b_i.eval()

        pickle.dump(var_map, open(model_path, 'wb'))
        print('model dumped at %s' % model_path)


class FMNN:
    def __init__(self,  _rch_argv, _init_argv, _ptmzr_argv, _reg_argv, mode, src, loss_mode='full'):
        self.src = src
        self.graph = tf.Graph()
        with self.graph.as_default():
            X_dim, M, rank, self.l_dims, act_func = _rch_argv
            K = rank
            mbd_dim = M * K + 1
            self.log = 'input dim: %d, features: %d, rank: %d, embedding: %d, self.l_dims: %s, activate: %s, ' % \
                       (X_dim, M, rank, mbd_dim, str(self.l_dims), act_func)

            init_acts = [('W', [X_dim, 1], 'random'),
                         ('V', [X_dim, rank], 'random'),
                         ('b', [1], 'zero')]
            if len(self.l_dims) == 1:
                h1_dim = self.l_dims[0]
                init_acts.extend([('h1_w', [mbd_dim, h1_dim], 'random'),
                                  ('h1_b', [h1_dim], 'zero'),
                                  ('h2_w', [h1_dim, 1], 'random'),
                                  ('h2_b', [1], 'zero'), ])
            elif len(self.l_dims) == 3:
                h1_dim, h2_dim, h3_dim = self.l_dims
                init_acts.extend([('h1_w', [mbd_dim, h1_dim], 'random'),
                                  ('h1_b', [h1_dim], 'zero'),
                                  ('h2_w', [h1_dim, h2_dim], 'random'),
                                  ('h2_b', [h2_dim], 'zero'),
                                  ('h3_w', [h2_dim, h3_dim], 'random'),
                                  ('h3_b', [h3_dim], 'zero'),
                                  ('h4_w', [h3_dim, 1], 'random'),
                                  ('h4_b', [1], 'zero')])
            elif len(self.l_dims) == 5:
                h1_dim, h2_dim, h3_dim, h4_dim, h5_dim = self.l_dims
                init_acts.extend([('h1_w', [mbd_dim, h1_dim], 'random'),
                                  ('h1_b', [h1_dim], 'zero'),
                                  ('h2_w', [h1_dim, h2_dim], 'random'),
                                  ('h2_b', [h2_dim], 'zero'),
                                  ('h3_w', [h2_dim, h3_dim], 'random'),
                                  ('h3_b', [h3_dim], 'zero'),
                                  ('h4_w', [h3_dim, h4_dim], 'random'),
                                  ('h4_b', [h4_dim], 'zero'),
                                  ('h5_w', [h4_dim, h5_dim], 'random'),
                                  ('h5_b', [h5_dim], 'zero'),
                                  ('h6_w', [h5_dim, 1], 'random'),
                                  ('h6_b', [1], 'zero')])
            elif len(self.l_dims) == 7:
                h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, h6_dim, h7_dim = self.l_dims
                init_acts.extend([('h1_w', [mbd_dim, h1_dim], 'random'),
                                  ('h1_b', [h1_dim], 'zero'),
                                  ('h2_w', [h1_dim, h2_dim], 'random'),
                                  ('h2_b', [h2_dim], 'zero'),
                                  ('h3_w', [h2_dim, h3_dim], 'random'),
                                  ('h3_b', [h3_dim], 'zero'),
                                  ('h4_w', [h3_dim, h4_dim], 'random'),
                                  ('h4_b', [h4_dim], 'zero'),
                                  ('h5_w', [h4_dim, h5_dim], 'random'),
                                  ('h5_b', [h5_dim], 'zero'),
                                  ('h6_w', [h5_dim, h6_dim], 'random'),
                                  ('h6_b', [h6_dim], 'zero'),
                                  ('h7_w', [h6_dim, h7_dim], 'random'),
                                  ('h7_b', [h7_dim], 'zero'),
                                  ('h8_w', [h7_dim, 1], 'random'),
                                  ('h8_b', [1], 'zero')])

            var_map, log = init_var_map(_init_argv, init_acts)

            self.log += log
            self.fm_w = tf.get_variable(var_map['W'])
            self.fm_v = tf.get_variable(var_map['V'])
            self.fm_b = tf.get_variable(var_map['b'])
            if len(self.l_dims) == 1:
                self.h1_w = tf.get_variable(var_map['h1_w'])
                self.h1_b = tf.get_variable(var_map['h1_b'])
                self.h2_w = tf.get_variable(var_map['h2_w'])
                self.h2_b = tf.get_variable(var_map['h2_b'])
            elif len(self.l_dims) == 3:
                self.h1_w = tf.get_variable(var_map['h1_w'])
                self.h1_b = tf.get_variable(var_map['h1_b'])
                self.h2_w = tf.get_variable(var_map['h2_w'])
                self.h2_b = tf.get_variable(var_map['h2_b'])
                self.h3_w = tf.get_variable(var_map['h3_w'])
                self.h3_b = tf.get_variable(var_map['h3_b'])
                self.h4_w = tf.get_variable(var_map['h4_w'])
                self.h4_b = tf.get_variable(var_map['h4_b'])
            elif len(self.l_dims) == 5:
                self.h1_w = tf.get_variable(var_map['h1_w'])
                self.h1_b = tf.get_variable(var_map['h1_b'])
                self.h2_w = tf.get_variable(var_map['h2_w'])
                self.h2_b = tf.get_variable(var_map['h2_b'])
                self.h3_w = tf.get_variable(var_map['h3_w'])
                self.h3_b = tf.get_variable(var_map['h3_b'])
                self.h4_w = tf.get_variable(var_map['h4_w'])
                self.h4_b = tf.get_variable(var_map['h4_b'])
                self.h5_w = tf.get_variable(var_map['h5_w'])
                self.h5_b = tf.get_variable(var_map['h5_b'])
                self.h6_w = tf.get_variable(var_map['h6_w'])
                self.h6_b = tf.get_variable(var_map['h6_b'])
            elif len(self.l_dims) == 7:
                self.h1_w = tf.get_variable(var_map['h1_w'])
                self.h1_b = tf.get_variable(var_map['h1_b'])
                self.h2_w = tf.get_variable(var_map['h2_w'])
                self.h2_b = tf.get_variable(var_map['h2_b'])
                self.h3_w = tf.get_variable(var_map['h3_w'])
                self.h3_b = tf.get_variable(var_map['h3_b'])
                self.h4_w = tf.get_variable(var_map['h4_w'])
                self.h4_b = tf.get_variable(var_map['h4_b'])
                self.h5_w = tf.get_variable(var_map['h5_w'])
                self.h5_b = tf.get_variable(var_map['h5_b'])
                self.h6_w = tf.get_variable(var_map['h6_w'])
                self.h6_b = tf.get_variable(var_map['h6_b'])
                self.h7_w = tf.get_variable(var_map['h7_w'])
                self.h7_b = tf.get_variable(var_map['h7_b'])
                self.h8_w = tf.get_variable(var_map['h8_w'])
                self.h8_b = tf.get_variable(var_map['h8_b'])

            self.wt_hldr = tf.placeholder(tf.float32, shape=[None, M])
            self.id_hldr = tf.placeholder(tf.int32, shape=[None, M])
            self.lbl_hldr = tf.placeholder(tf.float32)

            self.fm_wv = tf.concat([self.fm_w, self.fm_v], 1)

            keep_prob = _reg_argv[0]
            _lambda = _reg_argv[1]
            if mode == 'train':
                logits = self.forward(K, M, self.wt_hldr, self.id_hldr, act_func, keep_prob, True)
                log_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.lbl_hldr)
                if _ptmzr_argv[-1] == 'sum':
                    self.loss = tf.reduce_sum(log_loss)
                else:
                    self.loss = tf.reduce_mean(log_loss)
                self.train_preds = tf.sigmoid(logits, name='predictions')
                if loss_mode == 'full':
                    self.loss += _lambda * (tf.nn.l2_loss(self.fm_w) + tf.nn.l2_loss(self.fm_v))
                else: # 'batch'
                    self.loss += _lambda * (tf.nn.l2_loss(self.wx) + tf.nn.l2_loss(self.ori_z))
                self.ptmzr, log = build_optimizer(_ptmzr_argv, self.loss)
                self.log += '%s, reduce by: %s\tkeep_prob: %g' % (log, _ptmzr_argv[-1], keep_prob)

                self.eval_wt_hldr = tf.placeholder(tf.float32, shape=[None, M], name='wt')
                self.eval_id_hldr = tf.placeholder(tf.int32, shape=[None, M], name='id')

                eval_logits = self.forward(K, M, self.eval_wt_hldr, self.eval_id_hldr, act_func, keep_prob,
                                           False)
                self.eval_preds = tf.sigmoid(eval_logits, name='predictionNode')
            else:
                logits = self.forward(K, M, self.wt_hldr, self.id_hldr, act_func, keep_prob, False)
                self.test_preds = tf.sigmoid(logits)

    def forward(self, K, M, v_wts, c_ids, act_func, keep_prob, drop_out=False):
        mask = tf.reshape(v_wts, [-1, M, 1])
        self.wx = tf.multiply(tf.gather(self.fm_w, c_ids), mask)
        print("fm_w {}\nc_ids {}\nafter gather {}".format(self.fm_w.get_shape(), c_ids.get_shape(), tf.gather(self.fm_w, c_ids).get_shape()))

        if self.src == 'criteo':
            v_mbd = tf.reshape(v_wts[:, :13], [-1, 13, 1]) * tf.reshape(self.fm_v[:13, :], [1, 13, K])
            c_mbd = tf.gather(self.fm_v, c_ids[:, 13:])
            #b = tf.reshape(tf.concat([tf.identity(self.fm_b) for _i in range(N)], 0), [-1, 1])
            b = tf.reshape(tf.tile(tf.identity(self.fm_b), tf.shape(v_wts)[0:1]), [-1, 1])
            self.ori_z = tf.concat([v_mbd, c_mbd], 1)
        elif self.src == 'ipinyou':
            mask = tf.concat([tf.reshape(v_wts, [-1, M, 1]) for _i in range(K)], 2)
            c_mbd = tf.multiply(tf.gather(self.fm_v, c_ids), mask)
            #b = tf.reshape(tf.concat([tf.identity(self.fm_b) for _i in range(N)], 0), [N, 1])
            b = tf.reshape(tf.tile(tf.identity(self.fm_b), tf.shape(v_wts)[0:1]), [-1, 1])
            self.ori_z = c_mbd
        z = tf.transpose(self.ori_z, [0, 2, 1])

        v2x2 = tf.square(self.ori_z)
        vx2 = tf.square(tf.reduce_sum(self.ori_z,1))
        vxs = tf.reduce_sum(v2x2, 1)
        fmloss = 0.5 * tf.reduce_sum(vx2 - vxs, 1)
        if drop_out:
            l2 = tf.matmul(
                #tf.nn.dropout
                npu_ops.dropout(activate(act_func, tf.concat([tf.reshape(z, [-1, K*M]), b], 1)), keep_prob=keep_prob),
                self.h1_w) + self.h1_b
                #tf.nn.dropout(activate(act_func, tf.reshape(z, [-1, K*M])), keep_prob=keep_prob),
                #self.h1_w) + self.h1_b
            if len(self.l_dims) == 1:
                yhat = tf.matmul(npu_ops.dropout(activate(act_func, l2), keep_prob=keep_prob), self.h2_w) + self.h2_b
            elif len(self.l_dims) == 3:
                l3 = tf.matmul(npu_ops.dropout(activate(act_func, l2), keep_prob=keep_prob), self.h2_w) + self.h2_b
                l4 = tf.matmul(npu_ops.dropout(activate(act_func, l3), keep_prob=keep_prob), self.h3_w) + self.h3_b
                yhat = tf.matmul(npu_ops.dropout(activate(act_func, l4), keep_prob=keep_prob),
                                 self.h4_w) + self.h4_b
            elif len(self.l_dims) == 5:
                l3 = tf.matmul(npu_ops.dropout(activate(act_func, l2), keep_prob=keep_prob), self.h2_w) + self.h2_b
                l4 = tf.matmul(npu_ops.dropout(activate(act_func, l3), keep_prob=keep_prob), self.h3_w) + self.h3_b
                l5 = tf.matmul(npu_ops.dropout(activate(act_func, l4), keep_prob=keep_prob), self.h4_w) + self.h4_b
                l6 = tf.matmul(npu_ops.dropout(activate(act_func, l5), keep_prob=keep_prob), self.h5_w) + self.h5_b
                yhat = tf.matmul(npu_ops.dropout(activate(act_func, l6), keep_prob=keep_prob),
                                 self.h6_w) + self.h6_b
            elif len(self.l_dims) == 7:
                l3 = tf.matmul(npu_ops.dropout(activate(act_func, l2), keep_prob=keep_prob), self.h2_w) + self.h2_b
                l4 = tf.matmul(npu_ops.dropout(activate(act_func, l3), keep_prob=keep_prob), self.h3_w) + self.h3_b
                l5 = tf.matmul(npu_ops.dropout(activate(act_func, l4), keep_prob=keep_prob), self.h4_w) + self.h4_b
                l6 = tf.matmul(npu_ops.dropout(activate(act_func, l5), keep_prob=keep_prob), self.h5_w) + self.h5_b
                l7 = tf.matmul(npu_ops.dropout(activate(act_func, l6), keep_prob=keep_prob), self.h6_w) + self.h6_b
                l8 = tf.matmul(npu_ops.dropout(activate(act_func, l7), keep_prob=keep_prob), self.h7_w) + self.h7_b
                yhat = tf.matmul(npu_ops.dropout(activate(act_func, l8), keep_prob=keep_prob),
                                 self.h8_w) + self.h8_b
        else:
            #l2 = tf.matmul(tf.nn.dropout(activate(act_func, tf.reshape(z, [-1, M*K])), 1), self.h1_w) + self.h1_b
            l2 = tf.matmul(activate(act_func, tf.concat([tf.reshape(z, [-1, K*M]), b], 1)), self.h1_w) + self.h1_b

            if len(self.l_dims) == 1:
                yhat = tf.matmul(activate(act_func, l2), self.h2_w) + self.h2_b
            elif len(self.l_dims) == 3:
                l3 = tf.matmul(activate(act_func, l2), self.h2_w) + self.h2_b
                l4 = tf.matmul(activate(act_func, l3), self.h3_w) + self.h3_b
                yhat = tf.matmul(activate(act_func, l4), self.h4_w) + self.h4_b
            elif len(self.l_dims) == 5:
                l3 = tf.matmul(activate(act_func, l2), self.h2_w) + self.h2_b
                l4 = tf.matmul(activate(act_func, l3), self.h3_w) + self.h3_b
                l5 = tf.matmul(activate(act_func, l4), self.h4_w) + self.h4_b
                l6 = tf.matmul(activate(act_func, l5), self.h5_w) + self.h5_b
                yhat = tf.matmul(activate(act_func, l6), self.h6_w) + self.h6_b
            elif len(self.l_dims) == 7:
                l3 = tf.matmul(activate(act_func, l2), self.h2_w) + self.h2_b
                l4 = tf.matmul(activate(act_func, l3), self.h3_w) + self.h3_b
                l5 = tf.matmul(activate(act_func, l4), self.h4_w) + self.h4_b
                l6 = tf.matmul(activate(act_func, l5), self.h5_w) + self.h5_b
                l7 = tf.matmul(activate(act_func, l6), self.h6_w) + self.h6_b
                l8 = tf.matmul(activate(act_func, l7), self.h7_w) + self.h7_b
                yhat = tf.matmul(activate(act_func, l8), self.h8_w) + self.h8_b
        #return tf.reshape(tf.add(yhat, fmloss), [-1, ])
        return tf.reshape(yhat,[-1, ]) + tf.reshape(fmloss, [-1, ]) + tf.reshape(tf.reduce_sum(self.wx, 1), [-1, ])

    def dump(self, model_path):
        if len(self.l_dims) == 1:
            var_map = {'W': self.fm_w.eval(), 'V': self.fm_v.eval(), 'b': self.fm_b.eval(), 'h1_w': self.h1_w.eval(),
                       'h1_b': self.h1_b.eval(), 'h2_w': self.h2_w.eval(), 'h2_b': self.h2_b.eval()}
        elif len(self.l_dims) == 3:
            var_map = {'W': self.fm_w.eval(), 'V': self.fm_v.eval(), 'b': self.fm_b.eval(), 'h1_w': self.h1_w.eval(),
                       'h1_b': self.h1_b.eval(), 'h2_w': self.h2_w.eval(), 'h2_b': self.h2_b.eval(),
                       'h3_w': self.h3_w.eval(), 'h3_b': self.h3_b.eval(), 'h4_w': self.h4_w.eval(),
                       'h4_b': self.h4_b.eval()}
        elif len(self.l_dims) == 5:
            var_map = {'W': self.fm_w.eval(), 'V': self.fm_v.eval(), 'b': self.fm_b.eval(), 'h1_w': self.h1_w.eval(),
                       'h1_b': self.h1_b.eval(), 'h2_w': self.h2_w.eval(), 'h2_b': self.h2_b.eval(),
                       'h3_w': self.h3_w.eval(), 'h3_b': self.h3_b.eval(), 'h4_w': self.h4_w.eval(),
                       'h4_b': self.h4_b.eval(), 'h5_w': self.h5_w.eval(), 'h5_b': self.h5_b.eval(),
                       'h6_w': self.h6_w.eval(), 'h6_b': self.h6_b.eval()}
        elif len(self.l_dims) == 7:
            var_map = {'W': self.fm_w.eval(), 'V': self.fm_v.eval(), 'b': self.fm_b.eval(), 'h1_w': self.h1_w.eval(),
                       'h1_b': self.h1_b.eval(), 'h2_w': self.h2_w.eval(), 'h2_b': self.h2_b.eval(),
                       'h3_w': self.h3_w.eval(), 'h3_b': self.h3_b.eval(), 'h4_w': self.h4_w.eval(),
                       'h4_b': self.h4_b.eval(), 'h5_w': self.h5_w.eval(), 'h5_b': self.h5_b.eval(),
                       'h6_w': self.h6_w.eval(), 'h6_b': self.h6_b.eval(), 'h7_w': self.h7_w.eval(),
                       'h7_b': self.h7_b.eval(), 'h8_w': self.h8_w.eval(), 'h8_b': self.h8_b.eval()}

        pickle.dump(var_map, open(model_path, 'wb'))
        print('model dumped at %s' % model_path)
