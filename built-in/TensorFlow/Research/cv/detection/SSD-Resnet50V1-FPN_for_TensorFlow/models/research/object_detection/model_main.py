'Binary to run train and evaluation on object detection model.'
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
from tensorflow.core.protobuf import config_pb2
from absl import flags
import tensorflow as tf
#import horovod.tensorflow as hvd
import dllogger
import time
import os
from object_detection import model_hparams
from object_detection import model_lib
from object_detection.utils.exp_utils import AverageMeter, setup_dllogger



class NpuEmptyHook(tf.train.SessionRunHook):
    pass
flags.DEFINE_string('model_dir', None, 'Path to output model directory where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file.')
flags.DEFINE_string('raport_file', default='summary.json', help='Path to dlloger json')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('eval_training_data', False, 'If training data should be evaluated for this job. Note that one call only use this in eval-only mode, and `checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample one of every n train input examples for evaluation, where n is provided. This is only used if `eval_training_data` is True.')
flags.DEFINE_integer('eval_count', 1, 'How many times the evaluation should be run')
flags.DEFINE_string('hparams_overrides', None, 'Hyperparameter overrides, represented as a string containing comma-separated hparam_name=value pairs.')
flags.DEFINE_string('checkpoint_dir', None, 'Path to directory holding a checkpoint.  If `checkpoint_dir` is provided, this binary operates in eval-only mode, writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean('allow_xla', False, 'Enable XLA compilation')
flags.DEFINE_boolean('amp', False, 'Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.')
flags.DEFINE_boolean('run_once', False, 'If running in eval-only mode, whether to run just one round of eval vs running continuously (default).')
FLAGS = flags.FLAGS




class DLLoggerHook(tf.estimator.SessionRunHook):

    def __init__(self, global_batch_size, rank=(- 1)):
        self.global_batch_size = global_batch_size
        self.rank = rank
        setup_dllogger(enabled=True, filename=FLAGS.raport_file, rank=rank)

    def after_create_session(self, session, coord):
        self.meters = {}
        warmup = 100
        self.meters['train_throughput'] = AverageMeter(warmup=warmup)

    def before_run(self, run_context):
        self.t0 = time.time()
        return tf.estimator.SessionRunArgs(fetches=['global_step:0', 'learning_rate:0'])

    def after_run(self, run_context, run_values):
        throughput = (self.global_batch_size / (time.time() - self.t0))
        (global_step, lr) = run_values.results
        self.meters['train_throughput'].update(throughput)

    def end(self, session):
        summary = {'train_throughput': self.meters['train_throughput'].avg}
        dllogger.log(step=tuple(), data=summary)

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    #关闭tf的混合精度
    if FLAGS.amp:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    else:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    if True:
        session_config = tf.ConfigProto()
        #session_config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
    #### 增加集合通信初始化
    npu_int = npu_ops.initialize_system()
    npu_shutdown = npu_ops.shutdown_system()

    custom_op =  session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["mix_compile_mode"].b = True
    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    init_sess = tf.Session(config=session_config)
    init_sess.run(npu_int)
    init_sess.run(tf.tables_initializer())
    session_config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
    #### end

  
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session_config.gpu_options.visible_device_list = str(get_local_rank_id())
    if FLAGS.allow_xla:
        if True:
            session_config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
    model_dir = (FLAGS.model_dir if (get_rank_id() == 0) else None)
    #config = tf.estimator.RunConfig(model_dir=model_dir, session_config=session_config)
    #开启混合计算
    config = NPURunConfig(model_dir=model_dir, session_config=session_config, mix_compile_mode=True, iterations_per_loop=1)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(run_config=config, eval_count=FLAGS.eval_count, hparams=model_hparams.create_hparams(FLAGS.hparams_overrides), pipeline_config_path=FLAGS.pipeline_config_path, train_steps=FLAGS.num_train_steps, sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples, sample_1_of_n_eval_on_train_examples=FLAGS.sample_1_of_n_eval_on_train_examples)
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']
    if FLAGS.checkpoint_dir:
        if FLAGS.eval_training_data:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            input_fn = eval_input_fns[0]
        if FLAGS.run_once:
            estimator.evaluate(input_fn, steps=None, checkpoint_path=tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        else:
            model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn, train_steps, name)
    else:
        (train_spec, eval_specs) = model_lib.create_train_and_eval_specs(train_input_fn, eval_input_fns, eval_on_train_input_fn, predict_input_fn, train_steps, eval_on_train_data=False)
        train_hooks = [NpuEmptyHook(), DLLoggerHook((get_rank_size() * train_and_eval_dict['train_batch_size']), get_rank_id())]
        eval_hooks = []
        for x in range(FLAGS.eval_count):
            estimator.train(train_input_fn, hooks=train_hooks, steps=(train_steps // FLAGS.eval_count))
            #集合通信初始化
            init_sess = tf.Session(config=session_config)
            init_sess.run(npu_int)
            if (get_rank_id() == 0):
                eval_input_fn = eval_input_fns[0]
                #eval阻塞，临时规避
                #results = estimator.evaluate(eval_input_fn, steps=None, hooks=eval_hooks)
if (__name__ == '__main__'):
    tf.app.run()
