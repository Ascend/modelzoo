import tensorflow as tf
from .model_parallel_layers import ColumnParallelLinear, RowParallelLinear, ColumnLinear, RowLinear
import numpy as np

from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu_unary_ops import npu_unary_ops


from mde.distribute.mix_parallel_init import get_model_parallel_world_size

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])

def norm_1( x, axis=-1, epsilon=1e-5 ):
    with tf.variable_scope('layer_normalization_1', reuse=tf.AUTO_REUSE):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x * g + b
        return x

def norm_2( x, axis=-1, epsilon=1e-5 ):
    with tf.variable_scope('layer_normalization_2', reuse=tf.AUTO_REUSE):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x * g + b
        return x





def dropout(x, pdrop, train):                                ###########################################################注释掉，用于调试
   # if train and pdrop > 0:
 #     x = npu_ops.dropout(x, keep_prob=1.0-pdrop)
    return x

def gelu(x):
    return npu_unary_ops.gelu(x)



def ParallelMLP(args,  x, train):
    with tf.name_scope('ParallelMLP'):
        # dense h to 4h ( ColumnParallelLinear ), & activation
        hidden_size = args['hidden_size']
        x = ColumnParallelLinear(args, 'c_fc1', x, hidden_size*4, gather_output=False, init_scale_by_depth=False)

        x = gelu(x)
        x = RowParallelLinear(args, 'c_fc2',x, hidden_size, input_is_parallel=True, init_scale_by_depth=True)

        # add dropout
        x = dropout(x, args["res_dropout"], train)
        return x 

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])



def ParallelSelfAttention(args,  x, train):

    def split_heads(x):
        assert args['n_head'] % get_model_parallel_world_size() == 0, 'num heads should be equally partitioned by group size!'
        num_heads_per_partition = args['n_head'] // get_model_parallel_world_size()
        return tf.transpose(split_states(x, num_heads_per_partition), [0, 2, 1, 3])

    def merge_heads(x):
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(10000, w.dtype) * (1 - b) 
        return w

    def multihead_attention( q,k,v ):
      with tf.name_scope('multihead_attention'):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w, axis=-1)
        w = dropout(w, args["attn_dropout"], train)

        x = tf.matmul(w, v)
        return x

    with tf.name_scope('ParallelSelfAttention'):
        hidden_size = args['hidden_size']
        print ('----- in Parallel self atten , input x:', x)
        x = ColumnParallelLinear(args, 'c_attn', x, hidden_size * 3, gather_output=False, init_scale_by_depth=False)

        q, k, v = map( split_heads, tf.split(x, 3, axis=2 ))   

        x = multihead_attention( q,k,v )

        x = merge_heads( x )

        x = RowParallelLinear(args, 'c_proj', x, hidden_size, input_is_parallel=True, init_scale_by_depth=True)

        x = dropout(x, args["res_dropout"], train)

        return x


def ParallelTransformerLayer(args,  x, train):
    with tf.name_scope('ParallelTransformerLayer'):
        hidden_size = x.shape[-1].value  #hidden_size
        a = norm_1(x)  #[batchsize, sentence_length, hiddensize]
        a = ParallelSelfAttention(args, a, train)
        x = x + a # shortcut
        m = norm_2(x)
        m = ParallelMLP(args, m, train)
        x = x + m # shortcut
        return x

#------------------------------------------------------------------------------ No Parallel -------------------------------------

def MLP(args,  x, train):
    with tf.name_scope('MLP'):
        # dense h to 4h ( ColumnParallelLinear ), & activation
        hidden_size = args['hidden_size']
        x = ColumnLinear(args, 'c_fc1', x, hidden_size*4, gather_output=False, init_scale_by_depth=False )
        # dense 4h to h(RowParallelLinear), & activation
       # x = tf.Print(x, [x], message='Column Linear:', summarize=50 )
        x = gelu(x)
        x = RowLinear(args, 'c_fc2', x, hidden_size, input_is_parallel=True, init_scale_by_depth=True )
       # x = tf.Print(x, [x], message='Row Linear:', summarize=50 )
        # dropout, currently no dropout
        x = dropout(x, args["res_dropout"], train)
        return x 

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def SelfAttention(args,  x, train): 

    def split_heads(x):
        return tf.transpose(split_states(x, args["n_head"]), [0, 2, 1, 3])

    def merge_heads(x):
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        print ('--- in mask atten: nd, ns:', nd,ns)
        b = attention_mask(nd, ns, dtype=w.dtype)
        print ('--- in mask atten: b before reshape:', b)
        b = tf.reshape(b, [1, 1, nd, ns])
        print ('--- in mask atten:b:', b)
        w = w * b - tf.cast(10000, w.dtype) * (1 - b)
        return w

    def multihead_attention( q,k,v ):
      with tf.name_scope('multihead_attention'):
        # q, k, v have shape [batch, heads, sequence, features]
        with tf.name_scope('multihead_attention_1'):
            w = tf.matmul(q, k, transpose_b=True)
        with tf.name_scope('multihead_attention_2'):
            w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))
        with tf.name_scope('multihead_attention_3'):
            w = mask_attn_weights(w)
        print ('--- before softmax:', w)
      #  w = tf.Print(w, [w], message='before_softmax:')
        w = softmax(w, axis=-1)
    #    w = tf.Print(w, [w], message='after_softmax:')
        w = dropout(w, args["attn_dropout"], train)
        with tf.name_scope('multihead_attention_4'):
            x = tf.matmul(w, v)
      #  x = tf.Print(x, [x], message='After multihead atten:')
        return x

    with tf.name_scope('SelfAttention'):
        hidden_size = args['hidden_size']
        x = ColumnLinear(args,'c_attn', x, hidden_size * 3, gather_output=False, init_scale_by_depth=False )

        q, k, v = map( split_heads, tf.split(x, 3, axis=2 ))
        print ('---- in Self Attention, q,k,v:', q,k,v)
        x = multihead_attention( q,k,v )
        print ('--- before merge_head:', x)
        x = merge_heads( x )
 #       x = tf.Print(x, [x], message='!!!!! after merge heads:', summarize=50)
        print ('--- after merge_head:', x)
        x = RowLinear(args,'c_proj', x, hidden_size, input_is_parallel=True, init_scale_by_depth=True )
#        x = tf.Print(x, [x], message='!!!!! after RowLinear:', summarize=50)

        x = dropout(x, args["res_dropout"], train)
        return x


def TransformerLayer(args,  x , train):
    with tf.name_scope('TransformerLayer'):
        hidden_size = x.shape[-1].value  #hidden_size
        a = norm_1(x)  #[batchsize, sentence_length, hiddensize]
        a = SelfAttention(args, a, train)
        x = x + a # shortcut
        m = norm_2(x)
        m = MLP(args, m, train)
        x = x + m # shortcut
        return x


