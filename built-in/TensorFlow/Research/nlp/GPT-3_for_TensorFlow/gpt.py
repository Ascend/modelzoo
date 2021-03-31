import math
import numpy as np
import tensorflow as tf

from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu_unary_ops import npu_unary_ops

from mde.training.model_parallel_transformer import ParallelTransformerLayer, TransformerLayer


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


def gelu(x):
    return npu_unary_ops.gelu(x)


def norm(x, scope, *, axis=-1, epsilon=1e-5, params=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x * g + b
        return x


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])


def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])


def fc_matmul(x, scope, nf, *, w_init_stdev=0.02, params=None, init_scale_by_depth=False):
    if init_scale_by_depth:  # Scale by sqrt(2*num_layers), only happens at the final projection before a res block output,参考megatron pytorch脚本
        w_init_stdev = w_init_stdev * (1. / math.sqrt(2.0* params["n_layer"]))

    with tf.variable_scope(scope):
        *start, nx = shape_list(x)

        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev,seed=1))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b, start + [nf])
        return c


def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, params, train=False):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % params["n_head"] == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, params["n_head"]), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(1e4, w.dtype) * (1 - b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)

        w = dropout(w, params["attn_dropout"], train)

        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = fc_matmul(x, 'c_attn', n_state * 3, params=params, init_scale_by_depth=False)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a) 
        a = fc_matmul(a, 'c_proj', n_state, params=params, init_scale_by_depth=True)
        a = dropout(a, params["res_dropout"], train)
        return a, present


def mlp(x, scope, n_state, *, params, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(fc_matmul(x, 'c_fc', n_state, params=params, init_scale_by_depth=False))
        h2 = fc_matmul(h, 'c_proj', nx, params=params, init_scale_by_depth=True)
        h2 = dropout(h2, params["res_dropout"], train)
        return h2


def block(x, scope, *, past, params, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        if params["model_parallel"] and params["model_parallel_dim"] >1:
            x = ParallelTransformerLayer(params, x, train)         
            present=None
        else:
        #    a, present = attn(norm(x, 'attn/ln_1', params=params), 'attn', nx, past=past, params=params, train=train)
        #    with tf.variable_scope('attn_shared'):
        #        x = x + a
        #    m = mlp(norm(x, 'mlp/ln_2', params=params), 'mlp', nx * 4, params=params, train=train)
        #    with tf.variable_scope('mlp_shared'):
        #        x = x + m
            x = TransformerLayer(params, x, train )         
            present=None
        return x, present


def past_shape(*, params, batch_size=None, sequence=None):
    return [batch_size, params["n_layer"], 2, params["n_head"], sequence, params["n_embd"] // params["n_head"]]


def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)


def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def dropout(x, pdrop, train):                                ###########################################################注释掉，用于调试
    # if train and pdrop > 0:
    #     x = npu_ops.dropout(x, keep_prob=1.0-pdrop)

    return x


def _assert_float_dtype(dtype):
    if not dtype.is_floating:
        raise ValueError("Expected floating point type, got %s." % dtype)
    return dtype


def megatron(features, params, past=None, is_training=False):
    with tf.variable_scope('megatron', reuse=tf.AUTO_REUSE):
        results = {}
        batch, sequence = shape_list(features)

        wpe = tf.get_variable('wpe_embedding', [params["n_ctx"], params["n_embd"]],  # Position encoding
                                initializer=tf.random_normal_initializer(stddev=0.02,seed=1))    #原先是0.01
        wte = tf.get_variable('wte_embedding', [params["n_vocab"], params["n_embd"]],  # Text encoding
                                  initializer=tf.random_normal_initializer(stddev=0.02,seed=1))
        past_length = 0 if past is None else tf.shape(past)[-2]

        # wpe = dropout(wpe, params["embed_dropout"], is_training)
        # wte = dropout(wte, params["embed_dropout"], is_training)

        h = tf.gather(wte, features) + tf.gather(wpe, positions_for(features, past_length))
        h = dropout(h, params["embed_dropout"], is_training)     #修改1：在embedding输出上做dropout

        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * params["n_layer"]
        assert len(pasts) == params["n_layer"]


        for layer, past in enumerate(pasts):
            h, present = block(h, 'gpt2_block_%d' % layer, past=past, params=params, train=is_training)
            h = tf.identity(h, name='Checkpoints_'+str(layer))
   #         presents.append(present)
   #     results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f', params=params)

        h_flat = tf.reshape(h, [batch * sequence, params["n_embd"]])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, params["n_vocab"]])
        results['logits'] = logits
        # return results
        return logits
