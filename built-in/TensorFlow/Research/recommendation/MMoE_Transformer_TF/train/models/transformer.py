# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf

from .modules import get_token_embeddings, ff, positional_encoding, multihead_attention, dropout
import logging

logging.basicConfig(level=logging.INFO)

class Sdict:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

'''
xs: tuple of
    x: int32 tensor. (N, T1)
ys: tuple of
    decoder_input: int32 tensor. (N, T2)
    y: int32 tensor. (N, T2)
    y_seqlen: int32 tensor. (N, )
training: boolean.
'''

def encode(hp, UBS_embed, UBS_mask, target_item_embed, training=True, npu_mode=False):
    '''
    hp: vocab_size
        maxlen2
        num_heads
        num_blocks
        d_model
        d_ff
        keep_prob
    Returns
    memory: encoder outputs. (N, T1, d_model)
    '''
    hp = Sdict(**hp)
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        # [N, 300, 64]
        enc = UBS_embed

        # src_masks
        src_masks = UBS_mask # (N, T1)

        # embedding
        enc *= hp.d_model**0.5 # scale

        pos_embed = positional_encoding(enc, hp.maxlen1)
        enc = tf.concat([enc, pos_embed], axis=-1)
        if training:
            enc = dropout(enc, hp.keep_prob, npu_mode=npu_mode)
        
        ## Blocks
        for i in range(hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                enc = multihead_attention(queries=target_item_embed,
                                          keys=enc,
                                          values=enc,
                                          key_masks=src_masks,
                                          num_heads=hp.num_heads,
                                          keep_prob=hp.keep_prob,
                                          training=training,
                                          causality=False,
                                          npu_mode=npu_mode)
                # feed forward
                enc = ff(enc, num_units=[hp.d_ff, hp.d_model])
    memory = enc
    return memory, src_masks

def decode(hp, ys, memory, src_masks, training=True, npu_mode=False):
    '''
    memory: encoder outputs. (N, T1, d_model)
    src_masks: (N, T1)

    Returns
    logits: (N, T2, V). float32.
    y_hat: (N, T2). int32
    y: (N, T2). int32
    '''
    hp = Sdict(**hp)
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        embeddings, decoder_inputs, y = ys

        # tgt_masks
        tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)

        # embedding
        dec = tf.nn.embedding_lookup(embeddings, decoder_inputs)  # (N, T2, d_model)
        dec *= hp.d_model ** 0.5  # scale

        dec += positional_encoding(dec, hp.maxlen2)
        if training:
            dec = dropout(dec, hp.keep_prob, npu_mode=npu_mode)

        # Blocks
        for i in range(hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # Masked self-attention (Note that causality is True at this time)
                dec = multihead_attention(queries=dec,
                                          keys=dec,
                                          values=dec,
                                          key_masks=tgt_masks,
                                          num_heads=hp.num_heads,
                                          keep_prob=hp.keep_prob,
                                          training=training,
                                          causality=True,
                                          npu_mode=npu_mode,
                                          scope="self_attention")

                # Vanilla attention
                dec = multihead_attention(queries=dec,
                                          keys=memory,
                                          values=memory,
                                          key_masks=src_masks,
                                          num_heads=hp.num_heads,
                                          keep_prob=hp.keep_prob,
                                          training=training,
                                          causality=False,
                                          npu_mode=npu_mode,
                                          scope="vanilla_attention")
                ### Feed Forward
                dec = ff(dec, num_units=[hp.d_ff, hp.d_model])

    # Final linear projection (embedding weights are shared)
    weights = tf.transpose(embeddings) # (d_model, vocab_size)
    logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
    y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

    return logits, y_hat, y
