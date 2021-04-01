
'\nbert-blstm-crf layer\n@Author:Macan\n'
from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

class BLSTM_CRF(object):

    def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers, dropout_rate, initializers, num_labels, seq_length, labels, lengths, is_training):
        '\n        BLSTM-CRF 网络\n        :param embedded_chars: Fine-tuning embedding input\n        :param hidden_unit: LSTM的隐含单元个数\n        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）\n        :param num_layers: RNN的层数\n        :param droupout_rate: droupout rate\n        :param initializers: variable init class\n        :param num_labels: 标签数量\n        :param seq_length: 序列最大长度\n        :param labels: 真实标签\n        :param lengths: [batch_size] 每个batch下序列的真实长度\n        :param is_training: 是否是训练过程\n        '
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.embedding_dims = embedded_chars.shape[(- 1)].value
        self.is_training = is_training

    def add_blstm_crf_layer(self, crf_only):
        '\n        blstm-crf网络\n        :return:\n        '
        if self.is_training:
            self.embedded_chars = npu_ops.dropout(self.embedded_chars, self.dropout_rate)
        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            lstm_output = self.blstm_layer(self.embedded_chars)
            logits = self.project_bilstm_layer(lstm_output)
        (loss, trans) = self.crf_layer(logits)
        (pred_ids, _) = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return (loss, logits, trans, pred_ids)

    def _witch_cell(self):
        '\n        RNN 类型\n        :return:\n        '
        cell_tmp = None
        if (self.cell_type == 'lstm'):
            cell_tmp = rnn.LSTMCell(self.hidden_unit)
        elif (self.cell_type == 'gru'):
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        return cell_tmp

    def _bi_dir_rnn(self):
        '\n        双向RNN\n        :return:\n        '
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if (self.dropout_rate is not None):
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
        return (cell_fw, cell_bw)

    def blstm_layer(self, embedding_chars):
        '\n\n        :return:\n        '
        with tf.variable_scope('rnn_layer'):
            (cell_fw, cell_bw) = self._bi_dir_rnn()
            if (self.num_layers > 1):
                cell_fw = rnn.MultiRNNCell(([cell_fw] * self.num_layers), state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell(([cell_bw] * self.num_layers), state_is_tuple=True)
            (outputs, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars, dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        '\n        hidden layer between lstm layer and logits\n        :param lstm_outputs: [batch_size, num_steps, emb_size]\n        :return: [batch_size, num_steps, num_tags]\n        '
        with tf.variable_scope(('project' if (not name) else name)):
            with tf.variable_scope('hidden'):
                W = tf.get_variable('W', shape=[(self.hidden_unit * 2), self.hidden_unit], dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                b = tf.get_variable('b', shape=[self.hidden_unit], dtype=tf.float32, initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[(- 1), (self.hidden_unit * 2)])
                hidden = tf.nn.xw_plus_b(output, W, b)
            with tf.variable_scope('logits'):
                W = tf.get_variable('W', shape=[self.hidden_unit, self.num_labels], dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                b = tf.get_variable('b', shape=[self.num_labels], dtype=tf.float32, initializer=tf.zeros_initializer())
                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [(- 1), self.seq_length, self.num_labels])

    def project_crf_layer(self, embedding_chars, name=None):
        '\n        hidden layer between input layer and logits\n        :param lstm_outputs: [batch_size, num_steps, emb_size]\n        :return: [batch_size, num_steps, num_tags]\n        '
        with tf.variable_scope(('project' if (not name) else name)):
            with tf.variable_scope('logits'):
                W = tf.get_variable('W', shape=[self.embedding_dims, self.num_labels], dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                b = tf.get_variable('b', shape=[self.num_labels], dtype=tf.float32, initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars, shape=[(- 1), self.embedding_dims])
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [(- 1), self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        '\n        calculate crf loss\n        :param project_logits: [1, num_steps, num_tags]\n        :return: scalar loss\n        '
        with tf.variable_scope('crf_loss'):
            trans = tf.get_variable('transitions', shape=[self.num_labels, self.num_labels], initializer=self.initializers.xavier_initializer())
            if (self.labels is None):
                return (None, trans)
            else:
                (log_likelihood, trans) = tf.contrib.crf.crf_log_likelihood(inputs=logits, tag_indices=self.labels, transition_params=trans, sequence_lengths=self.lengths)
                return (tf.reduce_mean((- log_likelihood)), trans)
