# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""KeypointNet!!

A reimplementation of 'Discovery of Latent 3D Keypoints via End-to-end
Geometric Reasoning' keypoint network. Given a single 2D image of a known class,
this network can predict a set of 3D keypoints that are consistent across
viewing angles of the same object and across object instances. These keypoints
and their detectors are discovered and learned automatically without
keypoint location supervision.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import tensorflow as tf

from network import model_fn
from train_utils import train_and_eval


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("test", False, "Running testing if true")
tf.app.flags.DEFINE_string("model_dir", "", "Estimator model_dir")
tf.app.flags.DEFINE_string("tf_log_dir", "./", "Path to save the  tf summary file.")
tf.app.flags.DEFINE_string("data_url", "",
                           "Path to the directory containing the dataset.")
tf.app.flags.DEFINE_integer("steps", 700000, "Training steps")
tf.app.flags.DEFINE_integer("batch_size", 16, "Size of mini-batch.")
tf.app.flags.DEFINE_string(
    "hparams", "",
    "A comma-separated list of `name=value` hyperparameter values. This flag "
    "is used to override hyperparameter settings either when manually "
    "selecting hyperparameters or when using Vizier.")
tf.app.flags.DEFINE_integer(
    "sync_replicas", -1,
    "If > 0, use SyncReplicasOptimizer and use this many replicas per sync.")


# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler(os.path.join(FLAGS.tf_log_dir, 'tensorflow.log'))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)


# Fixed input size 128 x 128.
vw = vh = 128


def create_input_fn(split, batch_size):
    """Returns input_fn for tf.estimator.Estimator.

    Reads tfrecords and construts input_fn for either training or eval. All
    tfrecords not in test.txt or dev.txt will be assigned to training set.

    Args:
      split: A string indicating the split. Can be either 'train' or 'validation'.
      batch_size: The batch size!

    Returns:
      input_fn for tf.estimator.Estimator.

    Raises:
      IOError: If test.txt or dev.txt are not found.
    """

    if (not os.path.exists(os.path.join(FLAGS.data_url, "test.txt"))
            or not os.path.exists(
                os.path.join(FLAGS.data_url, "dev.txt"))):
        raise IOError("test.txt or dev.txt not found")

    with open(os.path.join(FLAGS.data_url, "test.txt"), "r") as f:
        testset = [x.strip() for x in f.readlines()]

    with open(os.path.join(FLAGS.data_url, "dev.txt"), "r") as f:
        validset = [x.strip() for x in f.readlines()]

    files = os.listdir(FLAGS.data_url)
    filenames = []
    for f in files:
        sp = os.path.splitext(f)
        if sp[1] != ".tfrecord" or sp[0] in testset:
            continue

        if ((split == "validation" and sp[0] in validset)
                or (split == "train" and sp[0] not in validset)):
            filenames.append(os.path.join(FLAGS.data_url, f))

    def input_fn():
        """input_fn for tf.estimator.Estimator."""
        def parser(serialized_example):
            """Parses a single tf.Example into image and label tensors."""
            fs = tf.parse_single_example(serialized_example,
                                         features={
                                             "img0":
                                             tf.FixedLenFeature([], tf.string),
                                             "img1":
                                             tf.FixedLenFeature([], tf.string),
                                             "mv0":
                                             tf.FixedLenFeature([16],
                                                                tf.float32),
                                             "mvi0":
                                             tf.FixedLenFeature([16],
                                                                tf.float32),
                                             "mv1":
                                             tf.FixedLenFeature([16],
                                                                tf.float32),
                                             "mvi1":
                                             tf.FixedLenFeature([16],
                                                                tf.float32),
                                         })

            fs["img0"] = tf.div(
                tf.to_float(tf.image.decode_png(fs["img0"], 4)), 255)
            fs["img1"] = tf.div(
                tf.to_float(tf.image.decode_png(fs["img1"], 4)), 255)

            fs["img0"].set_shape([vh, vw, 4])
            fs["img1"].set_shape([vh, vw, 4])

            # fs["lr0"] = [fs["mv0"][0]]
            # fs["lr1"] = [fs["mv1"][0]]

            fs["lr0"] = tf.convert_to_tensor([fs["mv0"][0]])
            fs["lr1"] = tf.convert_to_tensor([fs["mv1"][0]])

            return fs

        np.random.shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.shuffle(400).repeat().batch(batch_size,
                                                      drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=256)

        return dataset.make_one_shot_iterator().get_next(), None

    return input_fn


def _default_hparams(FLAGS):
    """Returns default or overridden user-specified hyperparameters."""

    hparams = tf.contrib.training.HParams(
        num_filters=64,  # Number of filters.
        num_kp=10,  # Numer of keypoints.
        loss_pose=0.2,  # Pose Loss.
        loss_con=1.0,  # Multiview consistency Loss.
        loss_sep=1.0,  # Seperation Loss.
        loss_sill=1.0,  # Sillhouette Loss.
        loss_lr=1.0,  # Orientation Loss.
        loss_variance=0.5,  # Variance Loss (part of Sillhouette loss).
        sep_delta=0.05,  # Seperation threshold.
        noise=0.1,  # Noise added during estimating rotation.
        learning_rate=1.0e-4,  # Learning rate
        lr_anneal_start=30000,  # When to anneal in the orientation prediction.
        lr_anneal_end=60000,  # When to use the prediction completely.
        batch_size=FLAGS.batch_size,  # Batch size
        data_url=FLAGS.data_url  # Path to dataset
    )
    if FLAGS.hparams:
        hparams = hparams.parse(FLAGS.hparams)
    return hparams


def train(argv):
    """main function to train the model."""
    del argv
    hparams = _default_hparams(FLAGS)

    train_and_eval(
        model_dir=FLAGS.model_dir,
        model_fn=model_fn,
        input_fn=create_input_fn,
        hparams=hparams,
        steps=FLAGS.steps,
        batch_size=FLAGS.batch_size,
        save_checkpoints_secs=600,
        sync_replicas=FLAGS.sync_replicas,
    )


if __name__ == "__main__":
    tf.app.run(train)
