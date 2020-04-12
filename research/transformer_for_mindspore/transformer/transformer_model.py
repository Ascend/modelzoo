# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Transformer model."""

import json
import math
import copy
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


class TransformerConfig:
    """
    Configuration for `Transformer`.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input sequence. Default: 128.
        vocab_size (int): The shape of each embedding vector. Default: 32000.
        hidden_size (int): Size of the layers. Default: 768.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder/decoder
                           cell. Default: 12.
        num_attention_heads (int): Number of attention heads in the Transformer
                             encoder/decoder cell. Default: 12.
        intermediate_size (int): Size of intermediate layer in the Transformer
                           encoder/decoder cell. Default: 3072.
        hidden_act (str): Activation function used in the Transformer encoder/decoder
                    cell. Default: "relu".
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        attention_probs_dropout_prob (float): The dropout probability for
                                      MultiheadAttention. Default: 0.1.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        label_smoothing (float): label smoothing setting. Default: 0.1
        input_mask_from_dataset (bool): Specifies whether to use the input mask that loaded from
                                 dataset. Default: True.
        dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size,
                 seq_length=128,
                 vocab_size=32000,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="relu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 initializer_range=0.02,
                 label_smoothing=0.1,
                 input_mask_from_dataset=True,
                 dtype=mstype.float32,
                 compute_type=mstype.float32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.label_smoothing = label_smoothing
        self.input_mask_from_dataset = input_mask_from_dataset
        self.dtype = dtype
        self.compute_type = compute_type
    
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `TransformerConfig` from a Python dictionary of parameters."""
        config = TransformerConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `TransformerConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
    """
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 embedding_shape,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(initializer
                                         (TruncatedNormal(initializer_range),
                                          [vocab_size, embedding_size]),
                                         name='embedding_table')
        self.expand = P.ExpandDims()
        self.shape_flat = (embedding_shape[0]*embedding_shape[1],)
        self.gather = P.GatherV2()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = tuple(embedding_shape)

    def construct(self, input_ids):
        flat_ids = self.reshape(input_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(
                one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)
        output = self.reshape(output_for_reshape, self.shape)
        return output, self.embedding_table


def position_encoding(length, depth,
                      min_timescale=1,
                      max_timescale=1e4):
    """Create Tensor of sinusoids of different frequencies.

    Args:
      length: Length of the Tensor to create, i.e. Number of steps.
      min_timescale: a float
      max_timescale: a float
      depth: an int

    Returns:
      Tensor of shape (length, depth)
    """
    depth = depth // 2
    # depth = float(depth)
    positions = np.arange(length, dtype=np.float32)
    # correspond to log(10000^(1/(d-1)))
    log_timescale_increment = (
        np.log(max_timescale / min_timescale) / (depth - 1))
    # correspond to 1 / 10000^(i/(d-1)), i=0....d-1
    inv_timescales = min_timescale * np.exp(
        np.arange(depth, dtype=np.float32) * -log_timescale_increment)
    # pos / 10000^(i/(d-1))
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    # intead of using SIN and COS interleaved
    # it's  the same to first use SIN then COS
    # as they are applied to the same position
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional embeddings to word embeddings.

    Args:
        embedding_size (int): The size of each embedding vector.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        dropout_prob (float): The dropout probability. Default: 0.1.
    """
    def __init__(self,
                 embedding_size,
                 embedding_shape,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 max_position_embeddings=512,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()

        self.scores_mul = Tensor([math.sqrt(float(embedding_size))], dtype=mstype.float32)
        self.multiply = P.Mul()
        self.add = P.TensorAdd()
        self.dropout = nn.Dropout(1 - dropout_prob, dtype=mstype.float32)
        self.expand_dims = P.ExpandDims()
        self.position_embedding_table = Tensor(
                position_encoding(max_position_embeddings, embedding_size),
                mstype.float32)

    def construct(self, word_embeddings):

        output = self.multiply(word_embeddings, self.scores_mul)

        position_embeddings = self.expand_dims(self.position_embedding_table, 0)
        output = self.add(output, position_embeddings)

        output = self.dropout(output)
        return output


class SaturateCast(nn.Cell):
    """
    Cast wrapper.
    """
    def __init__(self, src_type=mstype.float32, dst_type=mstype.float32):
        super(SaturateCast, self).__init__()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        return self.cast(x, self.dst_type)


class LayerPreprocess(nn.Cell):
    """
    preprocess input of each layer.
    """
    def __init__(self,
                 in_channels=None):
        super(LayerPreprocess, self).__init__()
        self.layernorm = nn.LayerNorm(in_channels)

    def construct(self, input_tensor):
        output =  self.layernorm(input_tensor)
        return output


class LayerPostprocess(nn.Cell):
    """
    postprocess ouput of each layer.
    """
    def __init__(self,
                 dropout_prob=0.1):
        super(LayerPostprocess, self).__init__()

        self.add = P.TensorAdd()
        self.dropout = nn.Dropout(1 - dropout_prob)

    def construct(self, hidden_tensor, input_tensor):
        output = self.dropout(hidden_tensor)
        output = self.add(output, input_tensor)
        return output



class MultiheadAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".

    Args:
        batch_size (int): Batch size of input datasets.
        from_tensor_width (int): Size of last dim of from_tensor.
        to_tensor_width (int): Size of last dim of to_tensor.
        from_seq_length (int): Length of from_tensor sequence.
        to_seq_length (int): Length of to_tensor sequence.
        num_attention_heads (int): Number of attention heads. Default: 1.
        size_per_head (int): Size of each attention head. Default: 512.
        query_act (str): Activation function for the query transform. Default: None.
        key_act (str): Activation function for the key transform. Default: None.
        value_act (str): Activation function for the value transform. Default: None.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: False.
        attention_probs_dropout_prob (float): The dropout probability for
                                      MultiheadAttention. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        do_return_2d_tensor (bool): True for return 2d tensor. False for return 3d
                             tensor. Default: False.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in MultiheadAttention. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size,
                 from_tensor_width,
                 to_tensor_width,
                 out_tensor_width,
                 from_seq_length,
                 to_seq_length,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 out_act=None,
                 has_attention_mask=True,
                 attention_probs_dropout_prob=0.0,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 do_return_2d_tensor=True,
                 compute_type=mstype.float32):

        super(MultiheadAttention, self).__init__()
        self.batch_size = batch_size
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.has_attention_mask = has_attention_mask
        assert has_attention_mask

        self.scores_mul = Tensor([1.0 / math.sqrt(float(self.size_per_head))], dtype=compute_type)
        self.reshape = P.Reshape()
        self.shape_from_2d = (-1, from_tensor_width)
        self.shape_to_2d = (-1, to_tensor_width)
        weight = TruncatedNormal(initializer_range)
        units = num_attention_heads * size_per_head
        self.query_layer = nn.Dense(from_tensor_width,
                                    units,
                                    activation=query_act,
                                    has_bias=False,
                                    weight_init=weight).to_float(compute_type)
        self.key_layer = nn.Dense(to_tensor_width,
                                  units,
                                  activation=key_act,
                                  has_bias=False,
                                  weight_init=weight).to_float(compute_type)
        self.value_layer = nn.Dense(to_tensor_width,
                                    units,
                                    activation=value_act,
                                    has_bias=False,
                                    weight_init=weight).to_float(compute_type)
        self.out_layer = nn.Dense(units,
                                  out_tensor_width,
                                  activation=out_act,
                                  has_bias=False,
                                  weight_init=weight).to_float(compute_type)

        self.shape_from = (batch_size, from_seq_length, num_attention_heads, size_per_head)
        self.shape_to = (
            batch_size, to_seq_length, num_attention_heads, size_per_head)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = Tensor([-10000.0,], dtype=compute_type)
        self.batch_num = batch_size * num_attention_heads
        self.matmul = P.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1 - attention_probs_dropout_prob)

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.TensorAdd()
            self.cast = P.Cast()
            self.get_dtype = P.DType()
        if do_return_2d_tensor:
            self.shape_return = (batch_size * from_seq_length, num_attention_heads * size_per_head)
        else:
            self.shape_return = (batch_size, from_seq_length, num_attention_heads * size_per_head)

        self.cast_compute_type = SaturateCast(dst_type=compute_type)


    def construct(self, from_tensor, to_tensor, attention_mask=None):
        # reshape 2d/3d input tensors to 2d
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)

        query_layer = self.reshape(query_out, self.shape_from)
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, self.shape_to)
        key_layer = self.transpose(key_layer, self.trans_shape)

        attention_scores = self.matmul_trans_b(query_layer, key_layer)

        attention_scores = self.multiply(attention_scores, self.scores_mul)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))

            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, self.shape_to)
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, self.shape_return)

        context_layer = self.out_layer(context_layer)

        return context_layer


class SelfAttention(nn.Cell):
    """
    Apply self-attention.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input sequence.
        hidden_size (int): Size of attention layers.
        num_attention_heads (int): Number of attention heads. Default: 12.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one_hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in MultiheadAttention. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size,
                 from_seq_length,
                 to_seq_length,
                 hidden_size,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 has_attention_mask=True,
                 compute_type=mstype.float32):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))

        self.size_per_head = int(hidden_size / num_attention_heads)

        self.attention = MultiheadAttention(
            batch_size=batch_size,
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            out_tensor_width=hidden_size,
            from_seq_length=from_seq_length,
            to_seq_length=to_seq_length,
            num_attention_heads=num_attention_heads,
            size_per_head=self.size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            has_attention_mask=has_attention_mask,
            do_return_2d_tensor=True,
            compute_type=compute_type)

        self.preprocess = LayerPreprocess(in_channels=hidden_size)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)
        
        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor, memory_tensor=None, attention_mask=None):
        input_tensor = self.reshape(input_tensor, self.shape)
        output = self.preprocess(input_tensor)
        
        if memory_tensor is None:
            memory_tensor = output

        attention_output = self.attention(output, memory_tensor, attention_mask)
        output = self.postprocess(attention_output, input_tensor)
        return output



class FeedForward(nn.Cell):
    """
    Apply two-layer feed forward

    Args:
        in_channels (int): Size of the input layer.
        hidden_size (int): Size of the hidden layer.
        out_channels (int): Size of the output layers.
        hidden_act (str): name of the activation function. Default: relu
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        compute_type (:class:`mindspore.dtype`): Compute type in FeedForward. Default: mstype.float32.
    """
    def __init__(self,
                 in_channels,
                 hidden_size,
                 out_channels,
                 hidden_act="relu",
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 compute_type=mstype.float32):
        super(FeedForward, self).__init__()
        
        weight = TruncatedNormal(initializer_range)
        
        self.conv1 = nn.Dense(in_channels,
                              hidden_size,
                              activation=hidden_act,
                              weight_init=weight).to_float(compute_type)
        self.conv2 = nn.Dense(hidden_size,
                              out_channels,
                              weight_init=weight).to_float(compute_type)

        self.preprocess = LayerPreprocess(in_channels=in_channels)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)
        
        self.reshape = P.Reshape()
        self.shape = (-1, in_channels)
        self.dropout = nn.Dropout(1 - hidden_dropout_prob)

    def construct(self, input_tensor):
        input_tensor = self.reshape(input_tensor, self.shape)
        output = self.preprocess(input_tensor)
        output = self.conv1(output)
        output = self.dropout(output)
        output = self.conv2(output)
        output = self.postprocess(output, input_tensor)
        return output


class EncoderCell(nn.Cell):
    """
    Encoder cells used in Transformer.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers. Default: 768.
        seq_length (int): Length of input sequence. Default: 512.
        num_attention_heads (int): Number of attention heads. Default: 12.
        intermediate_size (int): Size of intermediate layer. Default: 3072.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function. Default: "relu".
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size,
                 hidden_size=768,
                 seq_length=512,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.02,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=mstype.float32):
        super(EncoderCell, self).__init__()
        self.attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            from_seq_length=seq_length,
            to_seq_length=seq_length,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.feedforward = FeedForward(
            in_channels=hidden_size,
            hidden_size=intermediate_size,
            out_channels=hidden_size,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)


    def construct(self, hidden_states, attention_mask):
        # self-attention with ln, res
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)
        # feed forward with ln, res
        output = self.feedforward(attention_output)
        return output



class TransformerEncoder(nn.Cell):
    """
    Multi-layer transformer encoder.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        seq_length (int): Length of input sequence.
        num_hidden_layers (int): Number of hidden layers in encoder cells.
        num_attention_heads (int): Number of attention heads in encoder cells. Default: 12.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 3072.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1..
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size,
                 hidden_size,
                 seq_length,
                 num_hidden_layers,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=mstype.float32):
        super(TransformerEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers

        layers = []
        for _ in range(num_hidden_layers):
            layer = EncoderCell(batch_size=batch_size,
                                hidden_size=hidden_size,
                                seq_length=seq_length,
                                num_attention_heads=num_attention_heads,
                                intermediate_size=intermediate_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                use_one_hot_embeddings=use_one_hot_embeddings,
                                initializer_range=initializer_range,
                                hidden_dropout_prob=hidden_dropout_prob,
                                hidden_act=hidden_act,
                                compute_type=compute_type)
            layers.append(layer)

        self.layers = nn.CellList(layers)

        self.layer_preprocess = LayerPreprocess(
            in_channels=hidden_size)

        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)
        self.out_shape = (batch_size, seq_length, hidden_size)

    def construct(self, input_tensor, attention_mask):
        prev_output = self.reshape(input_tensor, self.shape)

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask)
            prev_output = layer_output

        prev_output = self.layer_preprocess(prev_output)
        output = self.reshape(prev_output, self.out_shape)
        return output



class DecoderCell(nn.Cell):
    """
    decoder cells used in Transformer.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the Transformer decoder layers. Default: 768.
        seq_length (int): Length of input sequence. Default: 512.
        num_attention_heads (int): Number of attention heads. Default: 12.
        intermediate_size (int): Size of intermediate layer. Default: 3072.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function. Default: "relu".
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size,
                 hidden_size=768,
                 seq_length=512,
                 enc_seq_length=512,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.02,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=mstype.float32):
        super(DecoderCell, self).__init__()
        self.self_attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            from_seq_length=seq_length,
            to_seq_length=seq_length,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.cross_attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            from_seq_length=seq_length,
            to_seq_length=enc_seq_length,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.feedforward = FeedForward(
            in_channels=hidden_size,
            hidden_size=intermediate_size,
            out_channels=hidden_size,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)


    def construct(self, hidden_states, attention_mask, enc_states, enc_attention_mask):
        # self-attention with ln, res
        attention_output = self.self_attention(hidden_states, attention_mask=attention_mask)
        # cross-attention with ln, res
        attention_output = self.cross_attention(attention_output, memory_tensor=enc_states, attention_mask=enc_attention_mask)
        # feed forward with ln, res
        output = self.feedforward(attention_output)
        return output



class TransformerDecoder(nn.Cell):
    """
    Multi-layer transformer decoder.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        seq_length (int): Length of input sequence.
        num_hidden_layers (int): Number of hidden layers in encoder cells.
        num_attention_heads (int): Number of attention heads in encoder cells. Default: 12.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 3072.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size,
                 hidden_size,
                 seq_length,
                 enc_seq_length,
                 num_hidden_layers,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=mstype.float32):
        super(TransformerDecoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers

        layers = []
        for _ in range(num_hidden_layers):
            layer = DecoderCell(batch_size=batch_size,
                                hidden_size=hidden_size,
                                seq_length=seq_length,
                                enc_seq_length=enc_seq_length,
                                num_attention_heads=num_attention_heads,
                                intermediate_size=intermediate_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                use_one_hot_embeddings=use_one_hot_embeddings,
                                initializer_range=initializer_range,
                                hidden_dropout_prob=hidden_dropout_prob,
                                hidden_act=hidden_act,
                                compute_type=compute_type)
            layers.append(layer)

        self.layers = nn.CellList(layers)

        self.layer_preprocess = LayerPreprocess(
            in_channels=hidden_size)

        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)
        self.out_shape = (batch_size, seq_length, hidden_size)


    def construct(self, input_tensor, attention_mask, enc_states, enc_attention_mask):
        prev_output = self.reshape(input_tensor, self.shape)

        all_decoder_layers = ()
        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, enc_states, enc_attention_mask)
            prev_output = layer_output

        prev_output = self.layer_preprocess(prev_output)
        output = self.reshape(prev_output, self.out_shape)
        return output


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (Class): Configuration for Transformer.
    """
    def __init__(self, config):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.input_mask_from_dataset = config.input_mask_from_dataset
        self.input_mask = None

        assert self.input_mask_from_dataset
        
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape_right = (config.batch_size, 1, config.seq_length)
        self.shape_left = (config.batch_size, config.seq_length, 1)
        self.batch_matmul = P.BatchMatMul()
        self.multiply = P.Mul()

        # mask future positions
        ones = np.ones(shape=(config.batch_size, config.seq_length, config.seq_length))
        self.tril_mask = Tensor(np.tril(ones), dtype=mstype.float32)


    def construct(self, input_mask, mask_future=False):
        input_mask = self.cast(input_mask, mstype.float32)

        mask_left = self.reshape(input_mask, self.shape_left)
        mask_right = self.reshape(input_mask, self.shape_right)
        
        attention_mask = self.batch_matmul(mask_left, mask_right)

        if mask_future:
            attention_mask = self.multiply(attention_mask, self.tril_mask)

        return attention_mask


class TransformerModel(nn.Cell):
    """
    Transformer with encoder and decoder.

    Args:
        config (Class): Configuration for Transformer.
        is_training (bool): True for training mode. False for eval mode.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=False):
        super(TransformerModel, self).__init__()
        config = copy.deepcopy(config)
        self.is_training = is_training
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.input_mask_from_dataset = config.input_mask_from_dataset
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size

        self.last_idx = self.num_hidden_layers - 1
        output_embedding_shape = [self.batch_size, self.seq_length,
                                  self.embedding_size]

        self.tfm_embedding_lookup = EmbeddingLookup(
            vocab_size=config.vocab_size,
            embedding_size=self.embedding_size,
            embedding_shape=output_embedding_shape,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=config.initializer_range)

        self.tfm_embedding_postprocessor = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            embedding_shape=output_embedding_shape,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=0.02,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

        self.tfm_encoder = TransformerEncoder(
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=config.initializer_range,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_act=config.hidden_act,
            compute_type=config.compute_type)
            #return_all_encoders=False)
        
        self.tfm_decoder = TransformerDecoder(
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
            enc_seq_length=self.seq_length,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=config.initializer_range,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_act=config.hidden_act,
            compute_type=config.compute_type)

        self.cast = P.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = SaturateCast(dst_type=config.compute_type)
        self.slice = P.StridedSlice()

        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)


    def construct(self, source_ids, source_mask, target_ids, target_mask):
        # process source sentence
        src_word_embeddings, embedding_tables = self.tfm_embedding_lookup(source_ids)
        src_embedding_output = self.tfm_embedding_postprocessor(src_word_embeddings)
        # attention mask [batch_size, seq_length, seq_length]
        enc_attention_mask = self._create_attention_mask_from_input_mask(source_mask)
        # transformer encoder
        encoder_output = self.tfm_encoder(self.cast_compute_type(src_embedding_output),
                                          self.cast_compute_type(enc_attention_mask))
        
        # process target sentence
        tgt_word_embeddings, _ = self.tfm_embedding_lookup(target_ids)
        tgt_embedding_output = self.tfm_embedding_postprocessor(tgt_word_embeddings)
        # attention mask [batch_size, seq_length, seq_length]
        tgt_attention_mask = self._create_attention_mask_from_input_mask(target_mask, mask_future=True)
        # transformer decoder
        decoder_output = self.tfm_decoder(self.cast_compute_type(tgt_embedding_output),
                                          self.cast_compute_type(tgt_attention_mask),
                                          encoder_output, enc_attention_mask)
        
        return encoder_output, decoder_output, embedding_tables
   