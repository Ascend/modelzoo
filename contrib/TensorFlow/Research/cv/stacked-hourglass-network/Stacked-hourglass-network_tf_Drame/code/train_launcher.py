#-*- coding:utf-8 –*-
"""
TRAIN LAUNCHER 

"""

import configparser
from hourglass_tiny import HourglassModel
from datagen import DataGenerator

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# os.symlink("/home/jida/mpii/code/mpii_human_pose_v1/events.*", "/home/tensorboard/c0ca75f3-7c02-4f8f-98b4-a22215f49d96/logs/events.*")
# In -s ln -s ~/tensorboard/c0ca75f3-7c02-4f8f-98b4-a22215f49d96/logs /home/jida/mpii/code/mpii_human_pose_v1

def process_config(conf_file):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
	# change 1
	# conf_file = open(conf_file, encoding='gbk')
	config.read(conf_file)
	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


if __name__ == '__main__':
	print('--Parsing Config File')
	params = process_config('/home/jida/mpii/code_run/config.cfg')
	print('params',params)
	
	print('--Creating Dataset')
	dataset = DataGenerator(params['joint_list'], params['img_directory'], params['training_txt_file'], remove_joints=params['remove_joints'])
	dataset._create_train_table()
	dataset._randomize()
	dataset._create_sets()
	
	model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'], attention = params['mcam'],training=True, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, name=params['name'], logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], tiny= params['tiny'], w_loss=params['weighted_loss'] , joints= params['joint_list'],modif=False)
	model.generate_model()
# 	model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset = None,load = 'test_99_84.833')
	model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset = None)
