# coding=utf-8
import tensorflow as tf
import os
from model import model_fn, RestoreMovingAverageHook
from input_pipeline import Pipeline
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
#from mxnet.model import save_checkpoint
import moxing as mox
import math
mox.file.shift('os','mox')

from absl.flags import Flag

tf.flags.DEFINE_string('data_url', " ", 'dataset path')
tf.flags.DEFINE_string('train_url'," ", 'train output path')
FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity('INFO')


"""
The purpose of this script is to eval a checkpoint model.

To use it just run:
python eval.py

Parameters below is for training 0.5x version.
"""

# 1281144/128 = 10008.9375
# so 1 epoch ~ 10000 steps

GPU_TO_USE = '0'
BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 128
NUM_EPOCHS = 160  # set 166 for 1.0x version

Epochs_between_evals = 5
TRAIN_DATASET_SIZE = 1281144 #共1281167
EVAL_DATASET_SIZE = 49999 #共50000
NUM_STEPS = NUM_EPOCHS * (TRAIN_DATASET_SIZE // BATCH_SIZE)
#####配置数据集路径
# 训练集
Cache_Path = '/cache/data'
OBS_TRAIN_DATA_Path = os.path.join(FLAGS.data_url, 'train')
TMP_TRAIN_DATA_Path = os.path.join(Cache_Path,'train')
mox.file.copy_parallel(src_url=OBS_TRAIN_DATA_Path,dst_url=TMP_TRAIN_DATA_Path)
# 测试集
OBS_VAL_DATA_Path = os.path.join(FLAGS.data_url,'val')
TMP_VAL_DATA_Path = os.path.join(Cache_Path, 'val')
mox.file.copy_parallel(src_url=OBS_VAL_DATA_Path, dst_url=TMP_VAL_DATA_Path)
# 配置需要加载的模型文件的路径
OBS_MODEL_Path = FLAGS.train_url
TMP_MODEL_Path = '/cache/model'
TRAINED_MODE_PATH = os.path.join(FLAGS.data_url, 'pretrained-model')
mox.file.copy_parallel(TRAINED_MODE_PATH, TMP_MODEL_Path)
# 配置验证log保存的路径
OBS_EVALLOG_PATH = os.path.join(FLAGS.data_url, 'txt')
TMP_EVALLOG_PATH = '/cache/log'
mox.file.copy_parallel(src_url=OBS_EVALLOG_PATH, dst_url=TMP_EVALLOG_PATH)

PARAMS = {
    'train_dataset_path': TMP_TRAIN_DATA_Path,#'/mnt/datasets/imagenet/train_shards/',
    'val_dataset_path': TMP_VAL_DATA_Path,#'/mnt/datasets/imagenet/val_shards/',
    'weight_decay': 4e-5 ,
    'initial_learning_rate': 0.06 , # 0.0625,  # 0.5/8
    'decay_steps': NUM_STEPS ,
    'end_learning_rate': 1e-5,  # 1e-6
    'model_dir': TMP_MODEL_Path ,
    'num_classes': 1000,
    'depth_multiplier': '0.5'  # set '1.0' for 1.0x version
}

def get_input_fn(is_training, num_epochs):

	dataset_path = PARAMS['train_dataset_path'] if is_training else PARAMS['val_dataset_path']
	filenames = os.listdir(dataset_path)
	filenames = [n for n in filenames if n.endswith('.tfrecords')]#筛除后缀不是.tfrecords的文件
	print("-----------------------------------",filenames)
	filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]
	batch_size = BATCH_SIZE if is_training else VALIDATION_BATCH_SIZE


	def input_fn():
		pipeline = Pipeline(
			filenames, is_training,
			batch_size=batch_size,
			num_epochs=num_epochs
		)

		return pipeline.dataset

	return input_fn

#设置运行配置
session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE

run_config = NPURunConfig(
	model_dir=PARAMS['model_dir'],
	session_config=session_config,
	save_summary_steps=500,
	save_checkpoints_steps=1200,
	precision_mode='allow_mix_precision',
)

estimator =NPUEstimator(model_fn,
						model_dir=PARAMS['model_dir'],
						config=run_config,
						params=PARAMS
						)
#配置验证输出文件路径
log_path = TMP_EVALLOG_PATH + '/val_log.txt'
if not os.path.exists(log_path):
	print("File does not exist!")
f = open(log_path, 'w')

for cycle_index in range(1):

	tf.compat.v1.logging.info('====================================Starting to evaluate.')
	eval_results = estimator.evaluate(input_fn=get_input_fn(False, 1), hooks=[RestoreMovingAverageHook(PARAMS['model_dir'])])
	for key in sorted(eval_results.keys()):
	    tf.logging.info("%s = %s", key, str(eval_results[key]))
	    line = (key, '\t', str(eval_results[key]), '\n')
	    f.writelines(line)
	enter = '\n'
	f.writelines(enter)
	mox.file.copy_parallel(src_url=TMP_MODEL_Path, dst_url=OBS_MODEL_Path)
f.close()
mox.file.copy_parallel(src_url=TMP_EVALLOG_PATH, dst_url=OBS_MODEL_Path)
print("Done!!!!!!!!!")