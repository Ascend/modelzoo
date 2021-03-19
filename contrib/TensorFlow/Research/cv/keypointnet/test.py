import os
import time
import math
import numpy as np
import tensorflow as tf
from network import model_fn


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("model_dir", "", "Estimator model_dir")
tf.app.flags.DEFINE_string("path", None, "Path to the specific model.")
tf.app.flags.DEFINE_string("data_url", "",
                           "Path to the directory containing the dataset.")
tf.app.flags.DEFINE_integer("steps", 700000, "Training steps")
tf.app.flags.DEFINE_integer("batch_size", 16, "Size of mini-batch.")
tf.app.flags.DEFINE_string(
    "hparams", "",
    "A comma-separated list of `name=value` hyperparameter values. This flag "
    "is used to override hyperparameter settings either when manually "
    "selecting hyperparameters or when using Vizier.")
tf.app.flags.DEFINE_string("gpu", '0', "GPU Id.")


os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu


# Fixed input size 128 x 128.
vw = vh = 128


def create_test_input_fn(batch_size, task="test"):
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

    if not os.path.exists(os.path.join(FLAGS.data_url, "test.txt")):
        raise IOError("test.txt or dev.txt not found")

    with open(os.path.join(FLAGS.data_url, "test.txt"), "r") as f:
        testset = [x.strip() for x in f.readlines()]

    with open(os.path.join(FLAGS.data_url, "dev.txt"), "r") as f:
        validset = [x.strip() for x in f.readlines()]

    files = os.listdir(FLAGS.data_url)
    filenames = []
    if task == "test":
        for f in files:
            sp = os.path.splitext(f)
            if sp[1] == ".tfrecord" and sp[0] in testset:
                filenames.append(os.path.join(FLAGS.data_url, f))
    elif task == "eval":
        for f in files:
            sp = os.path.splitext(f)
            if sp[1] == ".tfrecord" and sp[0] in validset:
                filenames.append(os.path.join(FLAGS.data_url, f))

    print(filenames)

    def input_fn():
        """input_fn for tf.estimator.Estimator."""

        def parser(serialized_example):
            """Parses a single tf.Example into image and label tensors."""
            fs = tf.parse_single_example(
                serialized_example,
                features={
                    "img0": tf.FixedLenFeature([], tf.string),
                    "img1": tf.FixedLenFeature([], tf.string),
                    "mv0": tf.FixedLenFeature([16], tf.float32),
                    "mvi0": tf.FixedLenFeature([16], tf.float32),
                    "mv1": tf.FixedLenFeature([16], tf.float32),
                    "mvi1": tf.FixedLenFeature([16], tf.float32),
                })

            fs["img0"] = tf.div(tf.to_float(
                tf.image.decode_png(fs["img0"], 4)), 255)
            fs["img1"] = tf.div(tf.to_float(
                tf.image.decode_png(fs["img1"], 4)), 255)

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
        dataset = dataset.shuffle(400).repeat(1).batch(batch_size)
        dataset = dataset.prefetch(buffer_size=256)

        return dataset.make_one_shot_iterator().get_next(), None

    return input_fn


def test_model_fn(func):
    """Creates model_fn for tf.Estimator.

    Args:
      func: A model_fn with prototype model_fn(features, labels, mode, hparams).

    Returns:
      model_fn for tf.estimator.Estimator.
    """

    def fn(features, labels, mode, params):
        """Returns model_fn for tf.estimator.Estimator."""

        ret = func(features, labels, mode, params)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=ret["loss"],
            predictions=ret["predictions"],
            eval_metric_ops=ret["eval_metric_ops"],)
    return fn


def test(
        model_dir,
        steps,
        batch_size,
        model_fn,
        input_fn,
        hparams,
        keep_checkpoint_every_n_hours=0.5,
        save_checkpoints_secs=180,
        save_summary_steps=50,
        eval_steps=20,
        eval_start_delay_secs=10,
        eval_throttle_secs=300,
        task="test",
        path=None):
    """Evaluate the pre_trained model"""

    run_config = tf.estimator.RunConfig(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        save_checkpoints_secs=save_checkpoints_secs,
        save_summary_steps=save_summary_steps)

    estimator = tf.estimator.Estimator(
        model_dir=model_dir,
        model_fn=test_model_fn(model_fn),
        params=hparams,
        config=run_config)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "  Test started...")
    if path is not None:
        print('\n')
        print(path)
        output = estimator.evaluate(input_fn=input_fn(batch_size=batch_size, task=task), checkpoint_path=path)
    else:
        print('\n')
        print("Loading Default Path")
        output = estimator.evaluate(input_fn=input_fn(batch_size=batch_size, task=task))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "  Test finished.")

    return output


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


if __name__ == "__main__":
    print("Testing")
    hparams = _default_hparams(FLAGS)
    output = test(
        model_dir=FLAGS.model_dir,
        model_fn=model_fn,
        input_fn=create_test_input_fn,
        hparams=hparams,
        steps=FLAGS.steps,
        batch_size=FLAGS.batch_size,
        save_checkpoints_secs=600,
        eval_throttle_secs=1800,
        eval_steps=5,
        task="test",
        path=FLAGS.path)
    print("Angular mean: ", np.mean(output['distances'])/math.pi*180)
