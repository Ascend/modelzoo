# coding=utf-8
# Copyright Huawei Noah's Ark Lab.

import codecs
import numpy as np

import tensorflow as tf

from noahnmt.utils import constant_utils


def get_logits_dataset(logit_file):
  """
  a dataset containing logit SparseTensor indices and values

  Args:
    logit_file: format 
      line := pos1_values pos2_values ... posN_values
      posi_values := vocab1_values,vocab2_values,...vocabM_values
      vocabj_values := id:value
  
  Returns:
    a tf.data.Dataset object
  """
  def _generator():
    if tf.gfile.Exists(logit_file):
      with codecs.getreader("utf-8")(tf.gfile.GFile(logit_file, "rb")) as f:
        for line in f:
          logit_pairs = [[value.split(":") for value in pos.split(",")] for pos in line.strip().split()]
          # logit_pairs = [(int(pos), int(vocab), float(value)) for pos, vocab, value in logit_pairs]
          # sparse_tensor = create_sparse_tensor(align_pairs)
          indices = [[vocab_id for vocab_id, value in pos] for pos in logit_pairs]
          values = [[value for vocab_id, value in pos] for pos in logit_pairs]
          yield (np.array(indices), np.array(values))
    else:
      raise ValueError("logit_file does not exist.")

  dataset = tf.data.Dataset.from_generator(
      _generator, (constant_utils.DT_INT(), 
                   constant_utils.DT_FLOAT()))
  return dataset



def compute_kd_loss(logits, 
                    kd_logit_indices,
                    kd_logit_values,
                    target_len=None):
  """Computes the kd loss for this model.

  Returns a tuple `(losses, loss)`, where `losses` are the per-batch
  losses and loss is a single scalar tensor to minimize.
  """
  # in case float16 training    
  if logits.dtype != tf.float32:
    logits = tf.cast(logits, dtype=tf.float32)
  
  batch, length, vocab = tf.unstack(tf.shape(logits))
  k = tf.shape(kd_logit_indices)[-1]
  
  # select logits
  indices = tf.tile(
      tf.expand_dims(tf.range(batch * length) * vocab, -1), 
      multiples=[1, k])
  indices = tf.reshape(indices, [-1]) + tf.reshape(kd_logit_indices, [-1])

  logits = tf.reshape(
      tf.gather(tf.reshape(logits, [-1]), indices),
      [batch, length, k])
  
  soft_labels = tf.nn.softmax(kd_logit_values)
  crossent = tf.nn.softmax_cross_entropy_with_logits_v2(
      logits=logits, labels=tf.stop_gradient(soft_labels))
  
  if target_len is not None:
    target_weights = tf.sequence_mask(
      target_len, length, dtype=logits.dtype)
    crossent *= target_weights
  
  return tf.reduce_sum(crossent)
