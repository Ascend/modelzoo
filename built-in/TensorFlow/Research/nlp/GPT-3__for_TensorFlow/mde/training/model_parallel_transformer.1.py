import tensorflow as tf
from .model_parallel_layers import ColumnParallelLinear, RowParallelLinear
import numpy as np 

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])

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

def norm( x, axis=-1, epsilon=1e-5 ):
    with tf.variable_scope('layer_normalization', reuse=tf.AUTO_REUSE):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x * g + b
        return x

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def ParallelMLP(args,  x):
    with tf.name_scope('ParallelMLP'):
        # dense h to 4h ( ColumnParallelLinear ), & activation
        hidden_size = args['hidden_size']
        x = ColumnParallelLinear( x, hidden_size*4, gather_output=False )

#        shape = shape_list(x)
#        print ('--- shape:', shape)
#        *start, m = shape_list(x)
#        print ('--- shape start:', *start, start)
#        print ('--- shape m:', m)
#        x = split_states(x, 8)
#        print ('--- after split_states:', x)

        # dense 4h to h(RowParallelLinear), & activation
        x = RowParallelLinear( x, hidden_size, input_is_parallel=True )
        # dropout, currently no dropout
        return x 



def ParallelSelfAttention(args,  x ):

    def split_heads(x):
        return tf.transpose(split_states(x, args["n_head"]), [0, 2, 1, 3])

    def merge_heads(x):
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(1e4, w.dtype) * (1 - b)
        return w

    def multihead_attention( q,k,v ):
      with tf.name_scope('multihead_attention'):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = tf.nn.softmax(w)
    #    w = dropout(w, params["attn_dropout"], train)

        x = tf.matmul(w, v)
        return x

    with tf.name_scope('ParallelSelfAttention'):
        hidden_size = args['hidden_size']
        x = ColumnParallelLinear( x, hidden_size * 3, gather_output=False )
        print ('-- in Parallel self atten, x:', x)

        q, k, v = map( split_heads, tf.split(x, 3, axis=2 ))
        print ('-- in Parallel self atten, q k v:', q, k, v)
        x = multihead_attention( q,k,v )
        print ('-- in Parallel self atten, after multihead atten:', x)

        x = merge_heads( x )
        x = RowParallelLinear( x, hidden_size, input_is_parallel=True )

        # dropout, currently no dropout
        return x


def ParallelTransformerLayer(args,  x ): 
    with tf.name_scope('ParallelTransformerLayer'):
        hidden_size = x.shape[-1].value  #hidden_size
        a = norm(x)  #[batchsize, sentence_length, hiddensize]
        a = ParallelSelfAttention(args, a)
        x = x + a # shortcut
        m = norm(x)
        m = ParallelMLP(args, m)
        x = x + m # shortcut
        return x





