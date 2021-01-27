from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
import modeling
import tensorflow as tf

from tensorflow.python.tools import freeze_graph
from tensorflow.contrib import layers as contrib_layers


flags.DEFINE_string(
    "albert_config", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "ckpt_dir", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")




flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")


FLAGS = flags.FLAGS
def get_squad(albert_config, output,p_mask,  start_n_top, max_seq_length, end_n_top, dropout_prob=0.1):
    bsz = tf.shape(output)[0]
    output = tf.transpose(output, [1, 0, 2])



    #final_hidden_matrix = tf.reshape(output, [bsz * seq_len, hidden_size])
    p_mask = tf.cast(p_mask, dtype=tf.float32)


    # logit of the start position
    with tf.variable_scope("start_logits"):
        start_logits = tf.layers.dense(
            output,
            1,
            kernel_initializer=modeling.create_initializer(
                albert_config.initializer_range))
        start_logits = tf.transpose(tf.squeeze(start_logits, -1), [1, 0])
        start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
        start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)

    # logit of the end position
    with tf.variable_scope("end_logits"):
      start_top_log_probs, start_top_index = tf.nn.top_k(
          start_log_probs, k=start_n_top)
      start_index = tf.one_hot(start_top_index,
                               depth=max_seq_length, axis=-1, dtype=tf.float32)
      print(start_logits.shape)
      print(output.shape)
      print(start_index.shape)
      start_features = tf.einsum("lbh,bkl->bkh", output, start_index)
      end_input = tf.tile(output[:, :, None],
                          [1, 1, start_n_top, 1])
      start_features = tf.tile(start_features[None],
                               [max_seq_length, 1, 1, 1])
      end_input = tf.concat([end_input, start_features], axis=-1)
      end_logits = tf.layers.dense(
          end_input,
          albert_config.hidden_size,
          kernel_initializer=modeling.create_initializer(
              albert_config.initializer_range),
          activation=tf.tanh,
          name="dense_0")
      end_logits = contrib_layers.layer_norm(end_logits, begin_norm_axis=-1)
      end_logits = tf.layers.dense(
          end_logits,
          1,
          kernel_initializer=modeling.create_initializer(
              albert_config.initializer_range),
          name="dense_1")
      end_logits = tf.reshape(end_logits, [max_seq_length, -1, start_n_top])
      end_logits = tf.transpose(end_logits, [1, 2, 0])
      end_logits_masked = end_logits * (
          1 - p_mask[:, None]) - 1e30 * p_mask[:, None]
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
      end_top_log_probs, end_top_index = tf.nn.top_k(
          end_log_probs, k=end_n_top)
      end_top_log_probs = tf.reshape(
          end_top_log_probs,
          [-1, start_n_top * end_n_top])
      end_top_index = tf.reshape(
          end_top_index,
          [-1, start_n_top * end_n_top])
      
    with tf.variable_scope("answer_class"):
        # get the representation of CLS
      cls_index = tf.one_hot(tf.zeros([bsz], dtype=tf.int32),
                           max_seq_length,
                           axis=-1, dtype=tf.float32)
      cls_feature = tf.einsum("lbh,bl->bh", output, cls_index)

        # get the representation of START
      start_p = tf.nn.softmax(start_logits_masked, axis=-1,
                            name="softmax_start")
      start_feature = tf.einsum("lbh,bl->bh", output, start_p)

        # note(zhiliny): no dependency on end_feature so that we can obtain
        # one single `cls_logits` for each sample
      ans_feature = tf.concat([start_feature, cls_feature], -1)
      ans_feature = tf.layers.dense(
            ans_feature,
            albert_config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=modeling.create_initializer(
                albert_config.initializer_range),
            name="dense_0")
      ans_feature = tf.layers.dropout(ans_feature, dropout_prob,
                                    training=False)
      cls_logits = tf.layers.dense(
            ans_feature,
            1,
            kernel_initializer=modeling.create_initializer(
                albert_config.initializer_range),
            name="dense_1",
            use_bias=False)
      cls_logits = tf.squeeze(cls_logits, -1)
      ps = tf.cast(start_top_index, tf.int32, name="p_s")
      pe = tf.cast(end_top_index, tf.int32, name="p_e")
 

    return start_top_index, end_top_index


def main(_):
    ckpt_path = os.path.join(FLAGS.ckpt_dir,"model.ckpt-best")
    max_seq_length=384
    start_n_top=5
    end_n_top=5
    input_ids = tf.placeholder(tf.int32, [None, 384], "input_ids")
    input_mask = tf.placeholder(tf.int32, [None, 384], "input_mask")
    segment_ids = tf.placeholder(tf.int32, [None, 384], "segment_ids")

    p_mask = tf.placeholder(tf.int32, [None, 384], "p_mask")
    #s_position = tf.placeholder(tf.int32, [None], "s_position")
    #e_position = tf.placeholder(tf.int32, [None], "e_position")
    #is_impossible = tf.placeholder(tf.int32, [None], "is_impossible")


    albert_config_path = FLAGS.albert_config
    albert_config = modeling.AlbertConfig.from_json_file(albert_config_path)
    model = modeling.AlbertModel(
        config=albert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    pred_s, pred_e = get_squad(albert_config, model.get_sequence_output(),p_mask,  start_n_top, max_seq_length, end_n_top)
   #end_logits/TopKV2
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, FLAGS.output_dir, 'model.pb')
        freeze_graph.freeze_graph(
            input_graph=os.path.join(FLAGS.output_dir, "model.pb"),
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names="end_logits/TopKV2,end_logits/Reshape_1,end_logits/Reshape_2,answer_class/Squeeze",
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=os.path.join(FLAGS.output_dir, "albert.pb"),
            clear_devices=False,
            initializer_nodes='')


if __name__ == "__main__":
    app.run(main)
