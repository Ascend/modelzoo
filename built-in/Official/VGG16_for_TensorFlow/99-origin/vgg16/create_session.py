import tensorflow as tf
import os,sys


class CreateSession():
    def __init__(self, config): 
        self.config = config

        self.estimator_config = tf.ConfigProto(allow_soft_placement=False)

        import horovod.tensorflow as hvd
        self.estimator_config.gpu_options.visible_device_list = str(hvd.local_rank())
        self.estimator_config.intra_op_parallelism_threads = 1
        self.estimator_config.inter_op_parallelism_threads = 5

        print("XLA is activated.")
        self.estimator_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.estimator_config.gpu_options.allow_growth = True

        self.set_env()

    def set_env(self):
        gpu_thread_count = 2
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

