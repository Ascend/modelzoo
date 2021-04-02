
'Creates and runs `Estimator` for object detection model on TPUs.\n\nThis uses the TPUEstimator API to define and run a model in TRAIN/EVAL modes.\n'
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
import os
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.core.protobuf import config_pb2
from npu_bridge.estimator.npu.npu_hook import NPUBroadcastGlobalVariablesHook
from absl import flags
import tensorflow as tf
from object_detection import model_hparams
from object_detection import model_lib

def npu_init(config):
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return config

def npu_hooks_append(hooks_list=[]):
    if (not isinstance(hooks_list, list)):
        hooks_list = []
    hooks_list.append(NPUBroadcastGlobalVariablesHook(0, int(os.getenv('RANK_ID', '0'))))
    return hooks_list
tf.flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than plain CPUs')
flags.DEFINE_string('gcp_project', default=None, help='Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string('tpu_zone', default=None, help='GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string('tpu_name', default=None, help='Name of the Cloud TPU for Cluster Resolvers.')
flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU cores).')
flags.DEFINE_integer('iterations_per_loop', 100, 'Number of iterations per TPU training loop.')
flags.DEFINE_string('mode', 'train', 'Mode to run: train, eval')
flags.DEFINE_integer('train_batch_size', None, 'Batch size for training. If this is not provided, batch size is read from training config.')
flags.DEFINE_string('hparams_overrides', None, 'Comma-separated list of hyperparameters to override defaults.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('eval_training_data', False, 'If training data should be evaluated for this job.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample one of every n train input examples for evaluation, where n is provided. This is only used if `eval_training_data` is True.')
flags.DEFINE_string('model_dir', None, 'Path to output model directory where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file.')
FLAGS = tf.flags.FLAGS

def main(unused_argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=[FLAGS.tpu_name], zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tpu_grpc_url = tpu_cluster_resolver.get_master()
    config = tf.contrib.tpu.RunConfig(master=tpu_grpc_url, evaluation_master=tpu_grpc_url, model_dir=FLAGS.model_dir, tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=FLAGS.iterations_per_loop, num_shards=FLAGS.num_shards), session_config=npu_init(config_pb2.ConfigProto()))
    kwargs = {}
    if FLAGS.train_batch_size:
        kwargs['batch_size'] = FLAGS.train_batch_size
    train_and_eval_dict = model_lib.create_estimator_and_inputs(run_config=config, hparams=model_hparams.create_hparams(FLAGS.hparams_overrides), pipeline_config_path=FLAGS.pipeline_config_path, train_steps=FLAGS.num_train_steps, sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples, sample_1_of_n_eval_on_train_examples=FLAGS.sample_1_of_n_eval_on_train_examples, use_tpu_estimator=True, use_tpu=FLAGS.use_tpu, num_shards=FLAGS.num_shards, save_final_config=(FLAGS.mode == 'train'), **kwargs)
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    train_steps = train_and_eval_dict['train_steps']
    if (FLAGS.mode == 'train'):
        estimator.train(input_fn=train_input_fn, max_steps=train_steps, hooks=npu_hooks_append())
    if (FLAGS.mode == 'eval'):
        if FLAGS.eval_training_data:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            input_fn = eval_input_fns[0]
        model_lib.continuous_eval(estimator, FLAGS.model_dir, input_fn, train_steps, name)
if (__name__ == '__main__'):
    tf.app.run()
