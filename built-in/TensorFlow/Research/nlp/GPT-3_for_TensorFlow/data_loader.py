import tensorflow as tf
from tensorflow.python.util import nest
import os 
import queue
import threading
import numpy as np
import os
import glob

from mde.distribute.mix_parallel_init import get_data_parallel_world_size,get_data_parallel_rank

USE_FAKE_DATA = True



class DataLoader(object):

  def __init__(self, config):
    self.config = config

  def get_train_input_fn(self):
    if USE_FAKE_DATA:
      ds = self.get_train_input_fn_fakedata()
    else:
      ds = self.get_train_input_fn_realdata()
    return ds

  def get_train_input_fn_fakedata(self):
      input_shape = [self.config['n_ctx']]
      input_element = nest.map_structure(lambda s: tf.constant(2, tf.int32, s), tf.TensorShape(input_shape))
      label_element = nest.map_structure(lambda s: tf.constant(1, tf.int32, s), tf.TensorShape(input_shape))
      element = (input_element, label_element)
      ds = tf.data.Dataset.from_tensors(element).repeat()
      ds = ds.batch(self.config['batch_size'], drop_remainder=True)
      return ds


  def get_train_input_fn_realdata(self):
      train = True
      batch_size = self.config['batch_size']
      max_seq_len = self.config['n_ctx']
      max_preds_per_seq = self.config['n_ctx'] - 1
      num_workers= 4
      seed=1 

      files = glob.glob(os.path.join(self.config['data_path'], "*.tfrecord"))
      assert max_preds_per_seq is not None, "--max-preds-per-seq MUST BE SPECIFIED when using tfrecords"
      tf.set_random_seed(seed)

      record_converter = Record2Example({"input_ids": tf.FixedLenFeature([max_seq_len], tf.int64),
                                          "input_mask": tf.FixedLenFeature([max_seq_len], tf.int64),
                                          "masked_lm_positions": tf.FixedLenFeature([max_preds_per_seq], tf.int64),
                                          "masked_lm_ids": tf.FixedLenFeature([max_preds_per_seq], tf.int64),
                                          "masked_lm_weights": tf.FixedLenFeature([max_preds_per_seq], tf.float32)})

      #Instantiate dataset according to original BERT implementation
      if train:
          ds = tf.data.Dataset.from_tensor_slices(tf.constant(files))
          
          ds = ds.shard(get_data_parallel_world_size(), get_data_parallel_rank())
        #   ds = ds.repeat()                                               #此处加repeat会导致样本经常重复，loss波动很大
          ds = ds.shuffle(buffer_size=len(files),seed=1)                         #################

          # use sloppy tfrecord dataset
          ds = ds.apply(
              tf.contrib.data.parallel_interleave(
                  tf.data.TFRecordDataset,
                  sloppy=train,
                  cycle_length=min(num_workers, len(files)))) #min(num_workers, len(files)
          ds = ds.shuffle(buffer_size=100, seed=1)
      else:
          ds = tf.data.TFRecordDataset(files)
          ds = ds.repeat()

      # Instantiate dataloader (do not drop remainder for eval)
      ds = ds.map(record_converter)   #解码


      loader_args = {'batch_size': batch_size, 
                      'num_parallel_batches': num_workers,
                      'drop_remainder': True}

      def generate_labels(x):
        #   sq_len = tf.size(x)
          sq_len = self.config['n_ctx']-1
          r1 = tf.range(0, sq_len)   #feature!!!
          r2 = tf.range(1, sq_len+1)  #label!!!
          feature = tf.gather(x, r1)
          label = tf.gather(x, r2)
          return feature, label

          # return tf.constant(2, dtype=tf.int64, shape=[511,], name='labels'), tf.constant(1, dtype=tf.int64, shape=[511,], name='labels')

      ds = ds.apply(tf.contrib.data.map_and_batch(generate_labels, **loader_args))
      return ds

class Record2Example(object):
    def __init__(self, feature_map):
        self.feature_map = feature_map

    def __call__(self, record):
        """Decodes a BERT TF record to a TF example."""
        example = tf.parse_single_example(record, self.feature_map)
        for k, v in list(example.items()):
            if v.dtype == tf.int64:
                example[k] = tf.to_int32(v)
        return example["input_ids"]
