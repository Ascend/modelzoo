import tensorflow as tf
import os,sys

# for MultiGPU
#import horovod.tensorflow as hvd

from hccl.manage.api import get_local_rank_id

class CreateSession():
    def __init__(self):
        from tensorflow.python.client import device_lib
        from npu_bridge.estimator import npu_ops
        self.estimator_config = tf.ConfigProto(allow_soft_placement=True)
        self.estimator_config.gpu_options.allow_growth = True

        local_device_protos = device_lib.list_local_devices(self.estimator_config)


        self.set_env()

    def set_env(self):
        # TODO, get env from config file
        gpu_thread_count = 2
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    def get_config_gpu(self):
        self.estimator_config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads
        self.estimator_config.inter_op_parallelism_threads = 5

        # enable XLA
        print("XLA is activated - Experimental Feature")
        self.estimator_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        return self.estimator_config


