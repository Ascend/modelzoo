import csv
import numpy as np
import time
import os
import argparse
import datetime
from random import randint
import tensorflow as tf
from tensorflow.python.framework import dtypes
from npu_bridge.estimator.npu.npu_dynamic_rnn import DynamicRNN
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu import util
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator.npu.npu_config import ProfilingConfig

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--deviceid', type=str, default='0',help="""deviceid""")
    parser.add_argument('-r', '--learningRate', type=float, default='0.7',help="""rate""")
    parser.add_argument('-dir', '--data_path', type=str, default = './training_data',help = """the data dir path""")
    parser.add_argument('-b', '--batchsize', type=int, default = 128,help = """training batchsize""")
    parser.add_argument('-s', '--steps', type=int, default = 20000,help = """training steps""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

args = parse_args()
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
os.environ['ASCEND_DEVICE_ID'] = args.deviceid
wordVectors = np.load(args.data_path + '/' + 'wordVectors.npy')
print ('Loaded the word vectors!')
print(wordVectors.shape)
maxSeqLength = 250
ids = np.load(args.data_path + '/' + 'idsMatrix.npy')
print ('Loaded the idsMatrix.npy!')
numlist = np.loadtxt('./getTrainBatch.txt',dtype=int)
count = 0

'''
def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if ((i % 2) == 0):
            num = randint(0, 12499)
            labels.append([1, 0])
        else:
            num = randint(12500, 24999)
            labels.append([0, 1])
        arr[i] = ids[(num - 1)]
    return (arr, labels)
'''

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    global count
    for i in range(batchSize):
        num = numlist[count]
        # print('num is %d' % num)
        if (num < 12500):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        count += 1
        arr[i] = ids[num]
    return (arr, labels)

batchSize = args.batchsize
lstmUnits = 64
numClasses = 2
numDimensions = 50
iterations = args.steps

tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
data = tf.cast(data, tf.float16)
data = tf.transpose(data, [1, 0, 2], name='transpose_time_major')
lstm = DynamicRNN(lstmUnits, dtypes.float16, time_major=True, forget_bias=0, is_training=False)
value, output_h, output_c, i, j, f, o, tanhct = lstm(data, seq_length= None, init_h = None, init_c = None)
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses], seed=1))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
last = tf.gather(value, (int(value.get_shape()[0]) - 1))
last = tf.cast(last, tf.float32)
prediction = (tf.matmul(last, weight) + bias)
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
# optimizer = tf.train.GradientDescentOptimizer(args.learningRate).minimize(loss)
#loss_scale
optimizer = tf.train.GradientDescentOptimizer(args.learningRate)
# loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**7, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.7)
loss_scale_manager = FixedLossScaleManager(loss_scale=1)#使用固定LossScale
optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager, is_distributed=False)
optimizer = optimizer.minimize(loss)


sess = tf.InteractiveSession()
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
#logdir = (('tensorboard/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')) + '/')
#writer = tf.summary.FileWriter(logdir, sess.graph)

#npu_config
config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_fp32_to_fp16")
#custom_op.parameter_map["iterations_per_loop"].i = 100

# autotune
#custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
# profiling
#custom_op.parameter_map["profiling_mode"].b = True
#custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/var/log/npu/profiling","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"embedding_lookup","bp_point":"rnn/strided_slice"}')
# dump debug
# custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/mz/dwx5325834/LSTM_DC/dump")
# custom_op.parameter_map["enable_dump_debug"].b = True
# custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
# dump data
# custom_op.parameter_map["enable_dump"].b = True
# custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/mz/dwx5325834/LSTM_DC/dump")
# custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("9820|9821|9822")
# custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.optimizers.extend(["pruning",
                                                        "function",
                                                        "constfold",
                                                        "shape",
                                                        "arithmetic",
                                                        "loop",
                                                        "dependency",
                                                        "layout",
                                                        "memory",
                                                        "GradFusionOptimizer"])

with tf.Session(config=config) as sess:
    saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.global_variables_initializer())
    total_time = time.time()
    csvfile = open("./csv/loss_scale.csv", "w+")
    writer = csv.writer(csvfile)
    writer.writerow(('step', 'loss', 'acc'))
    for i in range(iterations):
       #Next Batch of reviews
        (nextBatch, nextBatchLabels) = getTrainBatch()
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

       #Write summary to Tensorboard
        if (((i+1) % 10) == 0):
            #summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            #writer.add_summary(summary, i)
            (l, acc) = sess.run([loss, accuracy], {input_data: nextBatch, labels: nextBatchLabels})
            print(('training! Step%d Training loss: %f  Training acc: %f' % (i+1, l, acc)), flush=True)
            writer.writerow((i+1, l, acc))
        
       #Save the network every 10,000 training iterations
        # if ((((i+1) % 5000) == 0) and (i != 0)):
            # save_path = saver.save(sess, './models_dc/pretrained_lstm.ckpt', global_step=i+1)
            # print(('saved to %s' % save_path), flush=True)

    #writer.close()
    csvfile.close()
    total_time = (time.time() - total_time)
    FPS = 1/(total_time/iterations/batchSize)
    print(('**** total_time is %d S fps is %d ****' % (total_time, FPS)), flush=True)