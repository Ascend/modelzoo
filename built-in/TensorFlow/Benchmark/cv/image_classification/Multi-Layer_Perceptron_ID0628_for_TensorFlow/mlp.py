#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from npu_bridge.npu_init import *
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("precision_mode", "allow_fp32_to_fp16", "training precision")
flags.DEFINE_float("learning_rate", 0.01, "training learning rate")
flags.DEFINE_integer("training_epochs", 5000, "train epochs")
flags.DEFINE_integer("display_steps", 100, "display steps")
flags.DEFINE_string("Autotune", "False", "Autotune switch")
flags.DEFINE_string("data_path", "./", "data path")


def label_encode(label):
	val = []
	if label == "Iris-setosa":
		val = [1,0,0]
	elif label == "Iris-versicolor":
		val = [0,1,0]
	elif label == "Iris-virginica":
		val = [0,0,1]
	return val

def data_encode(file):
	X = []
	Y = []
	train_file = open(file, 'r')
	for line in train_file.read().strip().split('\n'):
		line = line.split(',')
		X.append([line[0],line[1],line[2],line[3]])
		Y.append(label_encode(line[4]))
	return X,Y

file = os.path.join(FLAGS.data_path, "iris.train")
train_X, train_Y = data_encode(file)

#print(train_Y)

#parametros
learning_rate = FLAGS.learning_rate
training_epochs = FLAGS.training_epochs
display_steps = FLAGS.display_steps

n_input = 4
n_hidden = 10
n_output = 3

#a partir daqui construimos o modelo
X = tf.placeholder("float",[None,n_input])
Y = tf.placeholder("float",[None,n_output])

weights = {
	"hidden": tf.Variable(tf.random_normal([n_input,n_hidden])),
	"output": tf.Variable(tf.random_normal([n_hidden,n_output])),
}

bias = {
	"hidden": tf.Variable(tf.random_normal([n_hidden])),
	"output": tf.Variable(tf.random_normal([n_output])),
}

def model(X, weights, bias):
	layer1 = tf.add(tf.matmul(X, weights["hidden"]),bias["hidden"])
	layer1 = tf.nn.relu(layer1)

	output_layer = tf.matmul(layer1,weights["output"]) + bias["output"]
	return output_layer

train_X, train_Y = data_encode(file)
file_test = os.path.join(FLAGS.data_path, "iris.test")
test_X, test_Y = data_encode(file_test)

pred = model(X,weights,bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
##################NPU modify start#####################
optimizador = tf.train.AdamOptimizer(learning_rate)
if FLAGS.precision_mode == "allow_mix_precision":
	loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32,
														   incr_every_n_steps=1000, decr_every_n_nan_or_inf=2,
														   decr_ratio=0.8)
	if int(os.getenv('RANK_SIZE')) == 1:
		optimizador = NPULossScaleOptimizer(optimizador, loss_scale_manager)
	else:
		optimizador = NPULossScaleOptimizer(optimizador, loss_scale_manager, is_distributed=True)
optimizador = npu_tf_optimizer(optimizador).minimize(cost)
#optimizador = npu_tf_optimizer(tf.train.AdamOptimizer(learning_rate)).minimize(cost)
##################NPU modify end#######################


init = tf.global_variables_initializer()
##################NPU modify start#####################
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(FLAGS.precision_mode)
if FLAGS.Autotune == "True":
    custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
custom_op.parameter_map["mix_compile_mode"].b = False
custom_op.parameter_map["enable_data_pre_proc"].b = True
custom_op.parameter_map["iterations_per_loop"].i = 10
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
##################NPU modify end#######################
with tf.Session(config=config) as sess:
	sess.run(init)
	duration = 0
	train_op = util.set_iteration_per_loop(sess, optimizador, 10)
	training_epochs = training_epochs / 10
	display_steps = display_steps / 10
	for epochs in range(int(training_epochs)):
		start_time = time.time()
		##################NPU modify start#####################
		_, c= sess.run([train_op,cost],feed_dict = {X: train_X, Y: train_Y})
		# lossScale = tf.get_default_graph().get_tensor_by_name("loss_scale:0")
		# overflow_status_reduce_all = tf.get_default_graph().get_tensor_by_name("overflow_status_reduce_all:0")
		# l_s, overflow_status_reduce_all, _, c = sess.run([lossScale, overflow_status_reduce_all, optimizador, cost], feed_dict={X: train_X, Y: train_Y})
		# print('loss_scale is : ', l_s)
		# print('overflow_status_reduce_all: ', overflow_status_reduce_all)
		##################NPU modify end#######################
		duration += (time.time() - start_time)
		if(epochs + 1) % display_steps == 0:
			print("Epoch:",epochs+1,"loss: {:3f} time cost  = {:4f}".format(c, duration))
			duration = 0
		if(epochs + 1) % 100 == 0:
			tf.train.Saver().save(sess, "ckpt_gpu/model.ckpt")
			tf.io.write_graph(sess.graph, './ckpt_gpu','graph.pbtxt', as_text=True)
	print("Optimization Finished")

	test_result = sess.run(pred,feed_dict = {X: train_X})
	correct_prediction = tf.equal(tf.argmax(test_result,1),tf.argmax(train_Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

	print("accuracy:", accuracy.eval({X: test_X, Y: test_Y}))











