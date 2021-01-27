import pickle
import tensorflow as tf
import json
from tensorflow.contrib import data as contrib_data
import numpy as np

import os
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_dir", None,
    "The inputput directory where the model checkpoints will be written.")


class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               paragraph_text,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.paragraph_text = paragraph_text
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

def input_fn_builder(input_file, seq_length, bsz):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "p_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }
    # p_mask is not required for SQuAD v1.1

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    batch_size = bsz

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=False))
    return d

def data_from_record(max_seq_length, predict_file, predict_feature_left_file):
    #with tf.gfile.Open(predict_feature_left_file, "rb") as fin:
    #    eval_features = pickle.load(fin)
    predict_input = input_fn_builder(
        input_file=predict_file,
        seq_length=max_seq_length,
        bsz=1)
    predict_iterator = predict_input.make_initializable_iterator()
    predict_next_element = predict_iterator.get_next()
    input_id = []
    input_mask = []
    segment_id = []
    p_mask = []
    unique_ids = []
    with tf.Session() as sess:
        sess.run(predict_iterator.initializer)
        idx = 0
        while True:
            try:
                feature = sess.run(predict_next_element)
                input_ids = feature["input_ids"]
                input_masks = feature["input_mask"]
                segment_ids = feature["segment_ids"]
                p_masks = feature["p_mask"]
                unique_id = feature["unique_ids"]
                input_id = np.array(input_ids)
                input_mask = np.array(input_masks)
                segment_id = np.array(segment_ids)
                p_mask = np.array(p_masks)
                unique_ids.append(unique_id)



                input_id.tofile("./input_ids/{0:05d}.bin".format(idx))
                input_mask.tofile("./input_masks/{0:05d}.bin".format(idx))
                segment_id.tofile("./segment_ids/{0:05d}.bin".format(idx))
                p_mask.tofile("./p_masks/{0:05d}.bin".format(idx))
                idx += 1
                '''

                input_id.extend(input_ids)
                input_mask.extend(input_masks)
                segment_id.extend(segment_ids)
                p_mask.extend(p_masks)
                '''
            except:
                break

    with open("./idx.txt", "w") as f:
        for id in unique_ids:
            f.write(str(id[0]) + '\n')
    f.close()

    '''
    input_id = np.array(input_id)
    input_mask = np.array(input_mask)
    segment_id = np.array(segment_id)
    p_mask = np.array(p_mask)
    print(input_id.shape)
    print(input_mask.shape)
    print(segment_id.shape)
    print(p_mask.shape)

    input_id.tofile("./input_id.bin")
    input_mask.tofile("./input_mask.bin")
    segment_id.tofile("./segment_id.bin")
    p_mask.tofile("./p_mask.bin")
    '''


def main():
    """main function to receive params them change data to bin.
    """
    predict_feature_file = os.path.join(FLAGS.input_dir, "dev.tfrecord")
    predict_feature_left_file = os.path.join(FLAGS.input_dir, "pred_left_file.pkl")
 
    data_from_record(384, predict_feature_file, predict_feature_left_file)

if __name__ == '__main__':
    main()



