import argparse
import os
import tensorflow as tf
from model import Model
import sys
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
"""
This script defines hyperparameters.
"""

def configure():
	flags = tf.flags
	# training

	flags.DEFINE_integer('num_steps', 20, 'maximum number of iterations')
	flags.DEFINE_integer('save_interval', 100, 'number of iterations for saving and visualization')
	flags.DEFINE_integer('random_seed', 1234, 'random seed')
	flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
	flags.DEFINE_float('learning_rate', 2.5e-4, 'learning rate')
	flags.DEFINE_float('power', 0.9, 'hyperparameter for poly learning rate')
	flags.DEFINE_float('momentum', 0.9, 'momentum')
	flags.DEFINE_string('encoder_name', 'deeplab', 'name of pre-trained model, res101, res50 or deeplab')
	flags.DEFINE_string('pretrain_file', './pretrained_model/deeplab_resnet.ckpt', 'pre-trained model filename corresponding to encoder_name')
	flags.DEFINE_string('data_list', './dataset/train.txt', 'training data list filename')

	# validation
	flags.DEFINE_integer('valid_step', 2000, 'checkpoint number for validation')
	flags.DEFINE_integer('valid_num_steps', 1447, '= number of validation samples')
	flags.DEFINE_string('valid_data_list', './dataset/val.txt', 'validation data list filename')
	flags.DEFINE_string('val_modeldir', './pretrained_model', 'the dir of thw model used to eval')


	# prediction / saving outputs for testing or validation
	flags.DEFINE_string('out_dir', 'output', 'directory for saving outputs')
	flags.DEFINE_integer('test_step', 20000, 'checkpoint number for testing/validation')
	flags.DEFINE_integer('test_num_steps', 1447, '= number of testing/validation samples')
	flags.DEFINE_string('test_data_list', './dataset/val.txt', 'testing/validation data list filename')
	flags.DEFINE_boolean('visual', True, 'whether to save predictions for visualization')

	# data
	flags.DEFINE_string('data_dir', './data', 'data directory') # 数据集路径
	flags.DEFINE_integer('batch_size', 5, 'training batch size')
	flags.DEFINE_integer('input_height', 321, 'input image height')
	flags.DEFINE_integer('input_width', 321, 'input image width')
	flags.DEFINE_integer('num_classes', 21, 'number of classes')
	flags.DEFINE_integer('ignore_label', 255, 'label pixel value that should be ignored')
	flags.DEFINE_boolean('random_scale', False, 'whether to perform random scaling data-augmentation')
	flags.DEFINE_boolean('random_mirror', False, 'whether to perform random left-right flipping data-augmentation')

	# log
	flags.DEFINE_string('modeldir', './model', 'model directory')
	flags.DEFINE_string('logfile', './log/log.txt', 'training log filename')
	flags.DEFINE_string('logdir', './log', 'training log directory')

	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_url', type=str, default=' ', help='the obs path of data')
	parser.add_argument('--train_url', type=str, default=' ', help='the obs path of output')
	parser.add_argument('--option', dest='option', type=str, default='train',help='actions: train, test, or predict')
	args = parser.parse_args()
	while len(sys.argv) > 1:
		sys.argv.pop()

	if args.option not in ['train', 'test', 'predict']:
		print('invalid option: ', args.option)
		print("Please input a option: train, test, or predict")
	else:
		config = tf.ConfigProto()
		custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
		custom_op.name = 'NpuOptimizer'
		custom_op.parameter_map["use_off_line"].b = True
		custom_op.parameter_map["mix_compile_mode"].b = True
		config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
		sess = tf.Session(config=config)
		model = Model(sess, configure())
		getattr(model, args.option)()
if __name__ == '__main__':
	#os.environ['SLOG_PRINT_TO_STDOUT'] = "1"
	main()
