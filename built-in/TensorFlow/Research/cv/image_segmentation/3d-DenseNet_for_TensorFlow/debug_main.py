
'This is the main class for debugging purpose'
from npu_bridge.npu_init import *
import os
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.core.protobuf import config_pb2
from npu_bridge.estimator.npu.npu_hook import NPUBroadcastGlobalVariablesHook
import tensorflow as tf
import source_dir.densenet_3d_estimator as estimator
import datetime
import sys

def npu_session_config_init(session_config=None):
    if ((not isinstance(session_config, config_pb2.ConfigProto)) and (not issubclass(type(session_config), config_pb2.ConfigProto))):
        session_config = config_pb2.ConfigProto()
    if (isinstance(session_config, config_pb2.ConfigProto) or issubclass(type(session_config), config_pb2.ConfigProto)):
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return session_config

def npu_run_config_init(run_config=None):
    if ((not isinstance(run_config, tf.estimator.RunConfig)) and (not issubclass(type(run_config), tf.estimator.RunConfig))):
        run_config = tf.estimator.RunConfig()
    if (isinstance(run_config, tf.estimator.RunConfig) or issubclass(type(run_config), tf.estimator.RunConfig)):
        run_config.__dict__['_session_config'] = npu_session_config_init(run_config.session_config)
    return run_config

def npu_hooks_append(hooks_list=[]):
    if (not isinstance(hooks_list, list)):
        hooks_list = []
    hooks_list.append(NPUBroadcastGlobalVariablesHook(0, int(os.getenv('RANK_ID', '0'))))
    return hooks_list
def get_time(start_time, end_time):
    start_time = str(start_time)
    end_time = str(end_time)
    # 日
    st_day = start_time[6:8]
    end_day = end_time[6:8]
    if st_day == end_day:
        st_sec = int(start_time[9:11]) * 3600 + int(start_time[12:14]) * 60 + int(start_time[15:])
        end_sec = int(end_time[9:11]) * 3600 + int(end_time[12:14]) * 60 + int(end_time[15:])
        sec_time = end_sec - st_sec
        hour = sec_time // 3600
        min = (sec_time - hour * 3600) // 60
        sec = sec_time - hour * 3600 - min * 60
        print("耗时 {} 时 {} 分 {} 秒" .format(hour, min, sec))
    elif end_day > st_day:
        st_sec = int(start_time[6:8]) * 24 * 3600 + int(start_time[9:11]) * 3600 + int(start_time[12:14]) * 60 + int(start_time[15:])
        end_sec = int(end_time[6:8]) * 24 * 3600 + int(end_time[9:11]) * 3600 + int(end_time[12:14]) * 60 + int(end_time[15:])
        sec_time = end_sec - st_sec
        day = sec_time // 24 // 3600
        hour = (sec_time - day * 24 *3600) // 3600
        min = (sec_time - day * 24 *3600 - hour * 3600) // 60
        sec = sec_time - day * 24 *3600 - hour * 3600 - min *60
        print("耗时 {} 天 {} 时 {} 分 {} 秒".format(day, hour, min, sec))
time1 = datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')
print(time1)
MODEL_DIR = 'denseNet3d_result'
DATA_DIR = '/home/out/3d-DenseNet_npu/tfrecord'
HYPERPARAMETERS = {'num_classes': 6, 'batch_size': 20, 'initial_learning_rate': 0.1, 'decay_step': 5000, 'lr_decay_factor': 0.1, 'growth_rate': 12, 'network_depth': 20, 'total_blocks': 3, 'keep_prob': 0.9, 'weight_decay': 0.0001, 'model_type': 'DenseNet3D', 'reduction': 0.5, 'bc_mode': True, 'num_frames_per_clip': 16, 'width': 120, 'height': 100, 'channel': 3, 'train_total_video_clip': 1044, 'eval_total_video_clip': 588}
TFRUNCONFIG = tf.estimator.RunConfig(log_step_count_steps=1, save_summary_steps=1, model_dir=MODEL_DIR)
CLASSIFIER = tf.estimator.Estimator(model_fn=estimator.model_fn, params=HYPERPARAMETERS, config=npu_run_config_init(run_config=TFRUNCONFIG))
CLASSIFIER.train(input_fn=(lambda : estimator.train_input_fn(DATA_DIR, HYPERPARAMETERS)), steps=1000, hooks=npu_hooks_append())
CLASSIFIER.evaluate(input_fn=(lambda : estimator.eval_input_fn(DATA_DIR, HYPERPARAMETERS)), steps=100)
time2 = datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')
print(time2)
get_time(time1, time2)
