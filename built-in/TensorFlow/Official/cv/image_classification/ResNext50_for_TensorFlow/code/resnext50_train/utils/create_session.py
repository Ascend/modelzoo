import tensorflow as tf
import os,sys

class CreateSession():
    def __init__(self, config): 
        self.config = config

        if self.config['accelerator'] == '1980':
            from tensorflow.python.client import device_lib
            #from tensorflow.contrib.offline_train.python import npu_ops
            from npu_bridge.estimator import npu_ops
            #self.estimator_config = tf.ConfigProto(allow_soft_placement=True, min_group_size=20, use_off_line=True)
            self.estimator_config = tf.ConfigProto(allow_soft_placement=True)
            custom_op = self.estimator_config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name = "NpuOptimizer"
            custom_op.parameter_map["use_off_line"].b = True
            custom_op.parameter_map["min_group_size"].b = 20
        else:
            self.estimator_config = tf.ConfigProto(allow_soft_placement=False)

        self.estimator_config.gpu_options.allow_growth = True

        if self.config['accelerator'] == '1980':
            local_device_protos = device_lib.list_local_devices(self.estimator_config)

        self.set_env()
      

    def set_env(self):
        # TODO, get env from config file
        gpu_thread_count = 2
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        # barrier = self.hvd.allreduce(tf.constant(0, dtype=tf.float32))
        # tf.Session(config=self.estimator_config).run(barrier)


    def get_config(self):
        self.estimator_config.gpu_options.visible_device_list = str(0)
#        self.estimator_config.gpu_options.force_gpu_compatible = True  # Force pinned memory
        self.estimator_config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads
        self.estimator_config.inter_op_parallelism_threads = 5
        return self.estimator_config


