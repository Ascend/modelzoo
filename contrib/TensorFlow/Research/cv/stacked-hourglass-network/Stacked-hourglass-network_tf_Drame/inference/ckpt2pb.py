import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator import npu_ops
import os
import configparser
from datagen import DataGenerator
from hourglass_tiny import HourglassModel


def process_config(conf_file):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
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

def main():
    tf.reset_default_graph()
    
    # 定义网络的输入节点
    input1 = tf.placeholder(dtype= tf.float32, shape= [None, 256, 256, 3], name = 'input_img')
    params = process_config('/home/jida/mpii/code_run/config.cfg')
    dataset = DataGenerator(params['joint_list'], params['img_directory'], params['training_txt_file'],     remove_joints=params['remove_joints'])
    dataset._create_train_table()
    dataset._randomize()
    dataset._create_sets()
    
    model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'], attention = params['mcam'],training=True, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, name=params['name'], logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], tiny= params['tiny'], w_loss=params['weighted_loss'] , joints= params['joint_list'],modif=False)
    if params['mcam']:
        output = model._graph_mcam(input)
    else :
        output = model._graph_hourglass(input1)
    print('===========', model)
    print('-----------', output)
    
    # 初始化参数
    sess = tf.Session()
    init = tf.global_variables_initializer() 
    sess.run(init)

    predict_class = tf.add(output,output,name = 'final_output')

    tf.train.write_graph(sess.graph_def, '../pb_model', 'model.pb')    # 通过write_graph生成模型文件
    freeze_graph.freeze_graph(
		       input_graph='../pb_model/model.pb',   # 传入write_graph生成的模型文件
		       input_saver='',
		       input_binary=False, 
		       input_checkpoint='./Test318_200_89.035',  # 传入训练生成的checkpoint文件
		       output_node_names= 'final_output',  # 与定义的推理网络输出节点保持一致
		       restore_op_name='save/restore_all',
		       filename_tensor_name='save/Const:0',
		       output_graph='../pb_model/Test318_200_89.035.pb',   # 改为需要生成的推理网络的名称
		       clear_devices=False,
		       initializer_nodes='')
    print("Convert ckpt to pb successfully!")

if __name__ == '__main__': 
    main()