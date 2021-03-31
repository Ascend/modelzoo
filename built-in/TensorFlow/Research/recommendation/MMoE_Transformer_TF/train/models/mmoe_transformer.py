from __future__ import print_function

import pickle
import tensorflow as tf
from train.models.tf_util import init_var_map, activate
import numpy as np
from .transformer import encode, decode
from .modules import dropout

# v3: batch_norm before act

class TransformerMMOE:

    def __init__(self, train_dataset_iterator, test_dataset_iterator, transformer_params, iter_per_epoch, dataset_argv, embedding_size, expert_argv, tower_argv, init_argv, ptmzr_argv, reg_argv,
                 max_seq_len=300, batch_norm=True, npu_mode=False):
        self.npu_mode = npu_mode
        self.iter_per_epoch = iter_per_epoch
        self.transformer_params = transformer_params
        # 1038446, 15
        (features_size, fields_num) = dataset_argv

        # num_tower_units: dcn deep-layers []
        # tower_deep_layers [100, 100, 100]
        # num_cross_layer 3
        # tower_act_func relu
        tower_deep_layers, num_cross_layer, tower_act_func = tower_argv

        # num_expert_units 16
        # num_experts 8
        # expert_act_func relu
        num_expert_units, num_experts, expert_act_func = expert_argv
        self.num_expert_units = num_expert_units
        self.num_experts = num_experts
        gate_act_func = 'softmax'
        keep_prob, _lambda, l1_lambda = reg_argv

        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.embedding_dim = (fields_num + max_seq_len * 2 * 4) * embedding_size

        # currently no output layer which is different from DNN
        # all_deep_layer = [self.embedding_dim] + tower_deep_layers
        # all_tower_deep_layers [16, 100, 100, 100]
        self.all_tower_deep_layers = [num_expert_units] + tower_deep_layers

        self.log = ('input dim: %d\n'
                    'num inputs: %d\n'
                    'embed size(each): %d\n'
                    'embedding layer: %d\n'
                    'size of expert unit: %d\n'
                    'num of experts: %d\n'
                    'num cross layer: %d\n'
                    'tower deep layers: %s\n'
                    'expert activate: %s\n'
                    'gate activate: %s\n'
                    'tower activate: %s\n'
                    'keep_prob: %g\n'
                    'l2(lambda): %g\n'
                    'l1(lambda): %g\n'
                    'max_seq_len: %s\n' %
                    (features_size, fields_num, embedding_size, self.embedding_dim, num_expert_units, num_experts,
                        num_cross_layer, self.all_tower_deep_layers, expert_act_func, gate_act_func,
                        tower_act_func, keep_prob, _lambda, l1_lambda, max_seq_len))

        # init input layer
        init_acts = [('embed', [features_size, embedding_size], 'random'),
                     ('experts', [self.embedding_dim, num_expert_units, num_experts], 'random'),
                     ('gate1', [self.embedding_dim, num_experts], 'random'),
                     ('gate2', [self.embedding_dim, num_experts], 'random'),
                     ('cross_w', [num_cross_layer, num_expert_units], 'random'),
                     ('cross_b', [num_cross_layer, num_expert_units], 'random')]

        # add tower layers
        for i in range(len(self.all_tower_deep_layers) - 1):
            init_acts.extend([('h%d_w' % (i + 1), self.all_tower_deep_layers[i: i + 2], 'random'),
                              ('h%d_b' % (i + 1), [self.all_tower_deep_layers[i + 1]], 'random')])
        # init_acts
        # ('embed', [1038446, 64], 'random'),
        # ('experts', [896, 16, 8], 'random'),
        # ('gate1', [896, 8], 'random'),
        # ('gate2', [896, 8], 'random'),
        # ('cross_w', [3, 16], 'random'), ('cross_b', [3, 16], 'random'),
        # ('h1_w', [16, 100], 'random'), ('h1_b', [100], 'random'),
        # ('h2_w', [100, 100], 'random'), ('h2_b', [100], 'random'),
        # ('h3_w', [100, 100], 'random'), ('h3_b', [100], 'random')
        var_map, log = init_var_map(init_argv, init_acts)

        self.log += log

        self.embed_v = tf.Variable(var_map['embed'])

        next_train_element = train_dataset_iterator.get_next()

        self.ctr_label = next_train_element["ClickLabel"]
        self.cvr_label = next_train_element["ConversionLabel"]
    
        self.global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

        # construct input embedding layer        
        vx_embed = self.construct_embedding(next_train_element)

        # init expert, gate and tower layers
        # (?, 13, 90)
        print('input layer shape: ', vx_embed.shape)
        print('input embedding dim: ', self.embedding_dim)

        self.cross_w_1 = tf.Variable(var_map['cross_w'])
        self.cross_b_1 = tf.Variable(var_map['cross_b'])

        self.cross_w_2 = tf.Variable(var_map['cross_w'])
        self.cross_b_2 = tf.Variable(var_map['cross_b'])

        self.h_w_1 = []
        self.h_b_1 = []
        self.h_w_2 = []
        self.h_b_2 = []
        for i in range(len(self.all_tower_deep_layers) - 1):
            self.h_w_1.append(tf.Variable(var_map['h%d_w' % (i + 1)]))
            self.h_b_1.append(tf.Variable(var_map['h%d_b' % (i + 1)]))
            self.h_w_2.append(tf.Variable(var_map['h%d_w' % (i + 1)]))
            self.h_b_2.append(tf.Variable(var_map['h%d_b' % (i + 1)]))

        self.experts = tf.Variable(var_map['experts'])
        self.gates = [tf.Variable(var_map['gate1']), tf.Variable(var_map['gate2'])]

        # first forward: input -> experts
        shared_expert_output = self.single_forward(vx_embed, self.experts, expert_act_func, keep_prob,
                                                    training=True,
                                                    batch_norm=batch_norm, name_scope="expert")
        self.gates_outputs = []
        for index, gate in enumerate(self.gates):
            gate_output = self.single_forward(vx_embed, gate, gate_act_func, keep_prob, training=True,
                                                batch_norm=batch_norm, name_scope="gate%d" % index)
            self.gates_outputs.append(gate_output)

        self.bottom_outputs = []
        for gate_output in self.gates_outputs:
            print('gate_output shape is %s' % gate_output.shape)
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)
            print('-------------------')
            print('expanded_gate_output shape is %s' % expanded_gate_output.shape)
            print('-------------------')
            repeated_expanded_gate_output = self.repeat_elements(expanded_gate_output, num_expert_units, axis=1)
            print('repeated_expanded_gate_output shape is %s' % repeated_expanded_gate_output.shape)
            gate_x_expert_output = tf.multiply(shared_expert_output, repeated_expanded_gate_output)
            print('**********************')
            print('gate_x_expert_output shape is %s' % gate_x_expert_output.shape)
            print('**********************')
            gate_x_expert_output_sum = tf.reduce_sum(gate_x_expert_output, axis=2)

            self.bottom_outputs.append(gate_x_expert_output_sum)

        # total two bottom_output
        assert len(self.bottom_outputs) == 2

        bottom_output1 = self.bottom_outputs[0]
        bottom_output2 = self.bottom_outputs[1]

        xl_1, final_hl_1 = self.forward(bottom_output1, num_cross_layer, self.cross_w_1, self.cross_b_1, self.h_w_1,
                                        self.h_b_1, tower_act_func, keep_prob, training=True, batch_norm=batch_norm,
                                        name_scope="ctr")

        xl_2, final_hl_2 = self.forward(bottom_output2, num_cross_layer, self.cross_w_2, self.cross_b_2, self.h_w_2,
                                        self.h_b_2, tower_act_func, keep_prob, training=True, batch_norm=batch_norm,
                                        name_scope="cvr")

        # concat the output of cross layer and deep layer
        x_stack_1 = tf.concat([xl_1, final_hl_1], 1)
        x_stack_2 = tf.concat([xl_2, final_hl_2], 1)

        print('x_stack_1 shape: %s' % x_stack_1.shape)
        print('x_stack_2 shape: %s' % x_stack_2.shape)
        # print(int(x_stack.shape[1]))

        init_acts_final = [('out_w', [int(x_stack_1.shape[1]), 1], 'random'),
                            ('out_b', [1], 'zero')]
        # ('out_w', [116, 1], 'random')
        # ('out_b', [1], 'zero')]
        var_map, log = init_var_map(init_argv, init_acts_final)

        self.log += log

        self.out_w_1 = tf.Variable(var_map['out_w'])
        self.out_b_1 = tf.Variable(var_map['out_b'])

        self.out_w_2 = tf.Variable(var_map['out_w'])
        self.out_b_2 = tf.Variable(var_map['out_b'])

        # print(self.out_w)
        # print(self.out_b)

        y_1 = self.final_forward(x_stack_1, self.out_w_1, self.out_b_1, tower_act_func, keep_prob,
                                    training=True,
                                    batch_norm=batch_norm, name_scope="task1")
        y_2 = self.final_forward(x_stack_2, self.out_w_2, self.out_b_2, tower_act_func, keep_prob,
                                    training=True,
                                    batch_norm=batch_norm, name_scope="task2")

        self.y_1 = y_1
        self.y_2 = y_2

        self.train_preds_ctr = tf.sigmoid(y_1, name='predicitons_ctr')
        self.train_preds_cvr = tf.sigmoid(y_2, name='predicitons_cvr')

        log_loss_ctr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_1, labels=self.ctr_label),
                                        name='ctr_loss')

        log_loss_cvr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_2, labels=self.cvr_label),
                                        name='cvr_loss')

        self.loss = tf.add(log_loss_ctr, log_loss_cvr)

        self.loss += tf.contrib.layers.l1_regularizer(l1_lambda)(self.cross_w_1) + tf.contrib.layers.l1_regularizer(
            l1_lambda)(self.cross_b_1) + tf.contrib.layers.l1_regularizer(l1_lambda)(
            self.cross_w_2) + tf.contrib.layers.l1_regularizer(
            l1_lambda)(self.cross_b_2) + _lambda * tf.nn.l2_loss(self.embed_v)
        
        self.test_dataset_iterator = test_dataset_iterator
        next_test_element = self.test_dataset_iterator.get_next()
        self.eval_ctr_label = next_test_element["ClickLabel"]
        self.eval_cvr_label = next_test_element["ConversionLabel"]
        eval_vx_embed = self.construct_embedding(next_test_element)

        eval_shared_expert_output = self.single_forward(eval_vx_embed, self.experts, expert_act_func, keep_prob,
                                                        training=False,
                                                        batch_norm=batch_norm, name_scope="expert")
        self.eval_gates_outputs = []
        for index, gate in enumerate(self.gates):
            gate_output = self.single_forward(eval_vx_embed, gate, gate_act_func, keep_prob, training=False,
                                                batch_norm=batch_norm, name_scope="gate%d" % index)
            self.eval_gates_outputs.append(gate_output)

        self.eval_bottom_outputs = []
        for gate_output in self.eval_gates_outputs:
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)
            gate_x_expert_output = tf.reduce_sum(tf.multiply(eval_shared_expert_output, self.repeat_elements(
                expanded_gate_output, num_expert_units, axis=1)), axis=2)
            self.eval_bottom_outputs.append(gate_x_expert_output)

        # total two bottom_output
        assert len(self.eval_bottom_outputs) == 2

        eval_bottom_output1 = self.eval_bottom_outputs[0]
        eval_bottom_output2 = self.eval_bottom_outputs[1]

        eval_xl_1, eval_final_hl_1 = self.forward(eval_bottom_output1, num_cross_layer, self.cross_w_1,
                                                    self.cross_b_1,
                                                    self.h_w_1, self.h_b_1, tower_act_func, keep_prob, training=False,
                                                    batch_norm=batch_norm, name_scope="ctr")

        eval_xl_2, eval_final_hl_2 = self.forward(eval_bottom_output2, num_cross_layer, self.cross_w_2,
                                                    self.cross_b_2,
                                                    self.h_w_2, self.h_b_2, tower_act_func, keep_prob, training=False,
                                                    batch_norm=batch_norm, name_scope="cvr")

        eval_x_stack_1 = tf.concat([eval_xl_1, eval_final_hl_1], 1)
        eval_x_stack_2 = tf.concat([eval_xl_2, eval_final_hl_2], 1)
        eval_y_1 = self.final_forward(eval_x_stack_1, self.out_w_1, self.out_b_1, tower_act_func, keep_prob,
                                        training=False,
                                        batch_norm=batch_norm, name_scope="task1")
        eval_y_2 = self.final_forward(eval_x_stack_2, self.out_w_2, self.out_b_2, tower_act_func, keep_prob,
                                        training=False,
                                        batch_norm=batch_norm, name_scope="task2")
        self.eval_preds_ctr = tf.sigmoid(eval_y_1, name='ctr_prediction_node')
        self.eval_preds_cvr = tf.sigmoid(eval_y_2, name='cvr_prediction_node')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # TODO adapt global_step
            learning_rate = tf.train.exponential_decay(learning_rate=ptmzr_argv[1], global_step=self.global_step//self.iter_per_epoch,
                                                        decay_rate=ptmzr_argv[3], decay_steps=ptmzr_argv[4],
                                                        staircase=False)
            
            if self.npu_mode:
                self.optmzr = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=ptmzr_argv[2]).minimize(
                                self.loss, global_step=self.global_step)
            else: 
                self.optmzr = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=ptmzr_argv[2])
                self.optmzr = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.optmzr)
                self.optmzr = self.optmzr.minimize(self.loss, global_step=self.global_step)

            # grads = self.optmzr.compute_gradients(self.loss)
            # self.optmzr = self.optmzr.apply_gradients(grads)

            log = 'optimizer: %s, learning rate: %g, epsilon: %g' % (ptmzr_argv[0], ptmzr_argv[1], ptmzr_argv[2])
        self.log += log

    # construct the embedding layer
    def construct_embedding(self, next_element):
        user_id = next_element["UID"]
        # [N, 8]
        user_attr = next_element["UserAttr"]

        target_item_id = next_element["ItemID"]
        # [N, 4]
        target_item_attr = next_element["ItemAttr"]

        UBS_brand_id = next_element["127_14_SeqID"]
        UBS_brand_len = next_element["127_14_SeqID_len"]
        UBS_brand_val = next_element["127_14_SeqVal"]

        UBS_cat_id = next_element["109_14_SeqID"]
        UBS_cat_len = next_element["109_14_SeqID_len"]
        UBS_cat_val = next_element["109_14_SeqVal"]

        UBS_shop_id = next_element["110_14_SeqID"]
        UBS_shop_len = next_element["110_14_SeqID_len"]
        UBS_shop_val = next_element["110_14_SeqVal"]

        UBS_intention_id = next_element["150_14_SeqID"]
        UBS_intention_len = next_element["150_14_SeqID_len"]
        UBS_intention_val = next_element["150_14_SeqVal"]

        scene = next_element["Scene"]
        # [N, 64]
        user_id_embed = embedding_lookup_npu(self.embed_v, user_id)
        # [N, 8, 64]
        user_attr_embed = embedding_lookup_npu(self.embed_v, user_attr)
        # [N, 64]
        target_item_id_embed = embedding_lookup_npu(self.embed_v, target_item_id)
        # [N, 4, 64]
        target_item_attr_embed = embedding_lookup_npu(self.embed_v, target_item_attr)
        # [N, 300, 64]
        UBS_brand_id_embed = embedding_lookup_npu(self.embed_v, UBS_brand_id)
        UBS_cat_id_embed = embedding_lookup_npu(self.embed_v, UBS_cat_id)
        UBS_shop_id_embed = embedding_lookup_npu(self.embed_v, UBS_shop_id)
        UBS_intention_id_embed = embedding_lookup_npu(self.embed_v,\
                                        UBS_intention_id)
        UBS_brand_id_embed = tf.multiply(UBS_brand_id_embed, tf.expand_dims(UBS_brand_val, -1))
        UBS_cat_id_embed = tf.multiply(UBS_cat_id_embed, tf.expand_dims(UBS_cat_val, -1))
        UBS_shop_id_embed = tf.multiply(UBS_shop_id_embed, tf.expand_dims(UBS_shop_val, -1))
        UBS_intention_id_embed = tf.multiply(UBS_intention_id_embed, tf.expand_dims(UBS_intention_val, -1))
        # [N, 300, 256]
        UBS_embed = tf.concat([UBS_brand_id_embed, UBS_cat_id_embed, UBS_shop_id_embed, UBS_intention_id_embed], -1)
        UBS_mask = tf.math.equal(UBS_brand_id + UBS_cat_id + UBS_shop_id + UBS_intention_id, 0)
        
        # [N, 64]
        scene = embedding_lookup_npu(self.embed_v, scene)

        user_embed = tf.concat([tf.expand_dims(user_id_embed, axis=1), user_attr_embed], axis=1)
        user_embed = tf.reshape(user_embed, [-1, 9*self.embedding_size])
        
        target_item_embed = tf.concat([tf.expand_dims(target_item_id_embed, axis=1), target_item_attr_embed], axis=1)
        # [N, 256]
        target_item_embed = tf.reshape(target_item_embed, [-1, 5*self.embedding_size])
        # (N, 300, 512)
        UBS_transformer_embed, _ = encode(self.transformer_params, UBS_embed, UBS_mask, target_item_embed, npu_mode=self.npu_mode)
        # (?, 153600)
        UBS_transformer_embed = tf.reshape(UBS_transformer_embed, [-1, self.max_seq_len*self.transformer_params["d_model"]])
        # (?, 154560)
        vx_embed = tf.concat([user_embed, scene, UBS_transformer_embed, target_item_embed], axis=1)

        return vx_embed

    def forward(self, vx_embed, num_cross_layer, cross_w, cross_b, h_w, h_b, act_func, keep_prob, training=True,
                batch_norm=False, name_scope="bn"):
        # embedding layer
        x0 = tf.reshape(vx_embed, [-1, self.num_expert_units])

        print('x0 shape: %s' % x0.shape)
        # cross layer
        xl = x0
        for i in range(num_cross_layer):
            xlw = tf.tensordot(xl, cross_w[i], axes=1)
            print('xlw shape: %s' % xlw.shape)
            xl = x0 * tf.expand_dims(xlw, -1) + cross_b[i] + xl
            xl.set_shape((None, self.num_expert_units))

        print('xl shape: %s' % xl.shape)

        # get final hidden layer output
        final_hl = self.deep_forward(vx_embed, h_w, h_b, act_func, keep_prob, training, batch_norm, name_scope)

        print('hidden layer shape: %s' % final_hl)

        return xl, final_hl

    def single_forward(self, vx_embed, x_tensor, act_func, keep_prob, training, batch_norm=False, name_scope="expert"):
        hidden_output = tf.reshape(vx_embed, [-1, self.embedding_dim])
        print('shape of hidden_output in single-forward of %s is %s' % (name_scope, hidden_output.shape))
        print('shape of x_tensor in single-forward of %s is %s' % (name_scope, x_tensor.shape))

        if training:
            if batch_norm:
                hidden_output = tf.contrib.layers.batch_norm(hidden_output, scale=True, is_training=True, reuse=False,\
                                fused=True, scope=(name_scope + "_single"))

            hidden_output = dropout(hidden_output, keep_prob=keep_prob, npu_mode=self.npu_mode)
        else:
            if batch_norm:
                hidden_output = tf.contrib.layers.batch_norm(hidden_output, scale=True, is_training=False, reuse=True,\
                                fused=True, scope=(name_scope + "_single"))


        # active function after matmul
        hidden_output = activate(act_func, tf.tensordot(hidden_output, x_tensor, axes=1))
        if name_scope == 'expert':
            hidden_output.set_shape((None, self.num_expert_units, self.num_experts))
        else:
            hidden_output.set_shape((None, self.num_experts))
        print('shape of hidden_output in batch_normalization of %s is %s' % (name_scope, hidden_output.shape))

        return hidden_output

    def deep_forward(self, vx_embed, h_w, h_b, act_func, keep_prob, training, batch_norm=False, name_scope="bn"):
        hidden_output = tf.reshape(vx_embed, [-1, self.num_expert_units])
        for i in range(len(h_w)):
            if training:
                hidden_output = tf.tensordot(activate(act_func, hidden_output), h_w[i], axes=1) + h_b[i]
                hidden_output.set_shape((None, h_w[i].shape[1]))
                if batch_norm:
                    print("setting bn for training stage")
                    hidden_output = tf.contrib.layers.batch_norm(hidden_output, scale=True, is_training=True, reuse=False,\
                                        fused=True, scope=(name_scope + "_%d") % i)

                hidden_output = dropout(hidden_output, keep_prob=keep_prob, npu_mode=self.npu_mode)
            else:
                hidden_output = tf.tensordot(activate(act_func, hidden_output), h_w[i], axes=1) + h_b[i]
                hidden_output.set_shape((None, h_w[i].shape[1]))
                if batch_norm:
                    print("setting bn for testing stage")
                    hidden_output = tf.contrib.layers.batch_norm(hidden_output, scale=True, is_training=False, reuse=True,\
                                        fused=True, scope=(name_scope + "_%d") % i)

        return hidden_output

    def final_forward(self, final_layer, out_w, out_b, act_func, keep_prob, training, batch_norm=False,
                      name_scope="bn"):
        hidden_output = final_layer
        if training:
            if batch_norm:
                hidden_output = tf.contrib.layers.batch_norm(final_layer, scale=True, is_training=True, reuse=False,\
                                        fused=True, scope=(name_scope + "_final"))

            hidden_output = dropout(hidden_output, keep_prob=keep_prob, npu_mode=self.npu_mode)
        else:
            if batch_norm:
                hidden_output = tf.contrib.layers.batch_norm(final_layer, scale=True, is_training=False, reuse=True,\
                                        fused=True, scope=(name_scope + "_final"))

        hidden_output = tf.matmul(activate(act_func, hidden_output), out_w) + out_b
        return tf.reshape(hidden_output, [-1])

    def repeat_elements(self, x, rep, axis):
        x_rep = tf.tile(x, [1, rep, 1])
        
        return x_rep


@tf.custom_gradient
def embedding_lookup_npu(params, indices):
    def grad(dy):
        params_shape = tf.shape(params, out_type=tf.int64)
        params_shape = tf.cast(params_shape, tf.int32)
        indices_int32 = tf.cast(indices, tf.int32)
        grad_embedding_lookup = tf.unsorted_segment_sum(dy, indices_int32, params_shape[0])
        return grad_embedding_lookup, None
    return tf.nn.embedding_lookup(params, indices), grad
