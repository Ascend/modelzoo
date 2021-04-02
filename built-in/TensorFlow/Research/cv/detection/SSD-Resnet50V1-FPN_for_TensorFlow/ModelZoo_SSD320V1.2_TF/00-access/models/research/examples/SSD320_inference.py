
from npu_bridge.npu_init import *
from tensorflow.core.protobuf import config_pb2
import sys
from absl import flags
from time import time
import tensorflow as tf
import dllogger
from object_detection import model_hparams
from object_detection import model_lib
from object_detection.utils.exp_utils import setup_dllogger
import numpy as np
flags.DEFINE_string('checkpoint_dir', None, 'Path to directory holding a checkpoint.  If `checkpoint_dir` is not provided, benchmark is running on random model')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file.')
flags.DEFINE_string('raport_file', default='summary.json', help='Path to dlloger json')
flags.DEFINE_integer('warmup_iters', 100, 'Number of iterations skipped during benchmark')
flags.DEFINE_integer('benchmark_iters', 300, 'Number of iterations measured by benchmark')
flags.DEFINE_integer('batch_size', 1, 'Number of inputs processed paralelly')
flags.DEFINE_list('percentiles', default=['90', '95', '99'], help='percentiles for latency confidence intervals')
FLAGS = flags.FLAGS
flags.mark_flag_as_required('pipeline_config_path')

def build_estimator():
    if True:
        session_config = tf.ConfigProto()
        session_config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
    config = tf.estimator.RunConfig(session_config=session_config)
    train_and_eval_dict = model_lib.create_estimator_and_inputs(run_config=config, hparams=model_hparams.create_hparams(None), pipeline_config_path=FLAGS.pipeline_config_path)
    estimator = train_and_eval_dict['estimator']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    return (estimator, eval_input_fns[0])

def build_benchmark_input_fn(input_fn):

    def benchmark_input_fn(params={}):
        params['batch_size'] = FLAGS.batch_size
        return input_fn(params).repeat().take((FLAGS.warmup_iters + FLAGS.benchmark_iters))
    return benchmark_input_fn

class TimingHook(tf.train.SessionRunHook):

    def __init__(self):
        super(TimingHook, self).__init__()
        setup_dllogger(enabled=True, filename=FLAGS.raport_file)
        self.times = []

    def before_run(self, *args, **kwargs):
        super(TimingHook, self).before_run(*args, **kwargs)
        self.start_time = time()

    def log_progress(self):
        if sys.stdout.isatty():
            print((len(self.times) - FLAGS.warmup_iters), '/', FLAGS.benchmark_iters, (' ' * 10), end='\r')

    def after_run(self, *args, **kwargs):
        super(TimingHook, self).after_run(*args, **kwargs)
        self.times.append((time() - self.start_time))
        self.log_progress()

    def end(self, *args, **kwargs):
        super(TimingHook, self).end(*args, **kwargs)
        throughput = ((sum([(1 / x) for x in self.times[FLAGS.warmup_iters:]]) * FLAGS.batch_size) / FLAGS.benchmark_iters)
        latency_avg = ((1000 * sum(self.times[FLAGS.warmup_iters:])) / FLAGS.benchmark_iters)
        latency_data = (1000 * np.array(self.times[FLAGS.warmup_iters:]))
        summary = {'infer_throughput': throughput, 'eval_avg_latency': latency_avg}
        print()
        print('Benchmark result:', throughput, 'img/s')
        for p in FLAGS.percentiles:
            p = int(p)
            tf.logging.info('Latency {}%: {:>4.2f} ms'.format(p, np.percentile(latency_data, p)))
            summary[f'eval_{p}%_latency'] = np.percentile(latency_data, p)
        dllogger.log(step=tuple(), data=summary)

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    (estimator, eval_input_fn) = build_estimator()
    checkpoint_path = (tf.train.latest_checkpoint(FLAGS.checkpoint_dir) if FLAGS.checkpoint_dir else None)
    results = estimator.predict(input_fn=build_benchmark_input_fn(eval_input_fn), checkpoint_path=checkpoint_path, hooks=[TimingHook()], yield_single_examples=False)
    list(results)
if (__name__ == '__main__'):
    tf.app.run()
