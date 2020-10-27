# coding=utf-8
# Copyright Huawei Noah's Ark Lab.

import tensorflow as tf

from noahnmt.configurable import _create_from_dict


def make_estimator_input_fn(def_dict, mode, hparams, **kwargs):
  """Return input_fn wrapped for Estimator.
  """
  # if mode == tf.estimator.ModeKeys.TRAIN:
  if not "params" in def_dict:
    def_dict["params"] = {}
  if mode == tf.estimator.ModeKeys.TRAIN:
    def_dict["params"]["batch_multiplier"] = hparams.worker_gpu
  if mode != tf.estimator.ModeKeys.EVAL:
    def_dict["params"]["batch_size"] = hparams.batch_size

  pipeline = _create_from_dict(
      def_dict, mode, **kwargs)

  def estimator_input_fn(params, config):
    return pipeline.read_data(), None
  
  return estimator_input_fn


def serving_input_fn(hparams=None):
    """Input fn for serving export, starting from serialized example.
    """
    # TODO
    # mode = tf.estimator.ModeKeys.PREDICT
    # serialized_example = tf.placeholder(
    #     dtype=tf.string, shape=[None], name="serialized_example")
    # dataset = tf.data.Dataset.from_tensor_slices(serialized_example)
    # dataset = dataset.map(self.decode_example)
    # dataset = dataset.map(lambda ex: self.preprocess_example(ex, mode, hparams))
    # dataset = dataset.map(self.maybe_reverse_and_copy)
    # dataset = dataset.map(data_reader.cast_ints_to_int32)
    # dataset = dataset.padded_batch(
    #     tf.shape(serialized_example, out_type=tf.int64)[0],
    #     dataset.output_shapes)
    # dataset = dataset.map(standardize_shapes)
    # features = tf.contrib.data.get_single_element(dataset)

    # if self.has_inputs:
    #   features.pop("targets", None)

    # return tf.estimator.export.ServingInputReceiver(
    #     features=features, receiver_tensors=serialized_example)
    pass