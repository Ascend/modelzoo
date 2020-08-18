import tensorflow as tf
import os,sys


class CreateSession():
    def __init__(self):
        self.estimator_config = tf.ConfigProto(
            inter_op_parallelism_threads=10,
            intra_op_parallelism_threads=10,
            allow_soft_placement=True)

        self.estimator_config.gpu_options.allow_growth = True

        self.set_env()

    def set_env(self):
        gpu_thread_count = 2
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

