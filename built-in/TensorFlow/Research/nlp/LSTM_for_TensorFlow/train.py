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
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--ckpt_count', type=int, default='0',help="""save checkpoint counts""")
    parser.add_argument('-r', '--learning_rate', type=float, default='0.1',help="""rate""")
    parser.add_argument('-dir', '--data_path', type=str, default = './training_data',help = """the data dir path""")
    parser.add_argument('-b', '--batch_size', type=int, default = 24,help = """training batchsize""")
    parser.add_argument('-s', '--steps', type=int, default = 100000,help = """training steps""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

args = parse_args()
ASCEND_DEVICE_ID = os.getenv("ASCEND_DEVICE_ID")

wordVectors = np.load(args.data_path + '/' + 'wordVectors.npy')
print ('Loaded the word vectors!')
print(wordVectors.shape)
ids = np.load(args.data_path + '/' + 'idsMatrix.npy')
print ('Loaded the idsMatrix.npy!')

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch(j):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        #num = randint(11499, 13499)
        num = 11499+i+batchSize*j
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels

batchSize = args.batch_size
maxSeqLength = 250
lstmUnits = 64
numClasses = 2
numDimensions = 50
iterations = args.steps

tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
data = tf.transpose(data, [1, 0, 2], name='transpose_time_major')
lstm = DynamicRNN(lstmUnits, dtypes.float32, time_major=True, forget_bias=1.0, is_training=True)
value, output_h, output_c, i, j, f, o, tanhct = lstm(data, seq_length= None, init_h = None, init_c = None)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
last = tf.gather(value, (int(value.get_shape()[0]) - 1))
prediction = (tf.matmul(last, weight) + bias)
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

#lr decay
learning_rate_base = args.learning_rate
learning_rate_decay = 0.99
learning_rate_step = 900
global_step = tf.Variable(0, trainable = False)
learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, learning_rate_step, learning_rate_decay, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)#.minimize(loss, global_step=global_step)

#gradient clipping
grads = optimizer.compute_gradients(loss)
for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 0.9), v)
optimizer = optimizer.apply_gradients(grads, global_step=global_step)

#sess = tf.InteractiveSession()
#tf.summary.scalar('Loss', loss)
#tf.summary.scalar('Accuracy', accuracy)
#merged = tf.summary.merge_all()
#logdir = (('tensorboard/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')) + '/')
#writer = tf.summary.FileWriter(logdir, sess.graph)

#npu_config
config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_fp32_to_fp16")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    total_time = time.time()
    csvfile = open("./output/" + ASCEND_DEVICE_ID + "/train.csv", "w+")
    writer = csv.writer(csvfile)
    writer.writerow(('step', 'loss', 'acc'))
    for i in range(iterations):
       #Next Batch of reviews
        (nextBatch, nextBatchLabels) = getTrainBatch()
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

       #Write summary to Tensorboard
        if (((i+1) % 100) == 0):
            #summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            #writer.add_summary(summary, i)
            (l, acc) = sess.run([loss, accuracy], {input_data: nextBatch, labels: nextBatchLabels})
            print(('training! Step%d Training loss: %f  Training acc: %f' % (i+1, l, acc)), flush=True)
            writer.writerow((i+1, l, acc))
        
       #Save the network every 10,000 training iterations
        if ((((i+1) % args.ckpt_count) == 0) and (i != 0)):
            save_path = saver.save(sess, './output/' + ASCEND_DEVICE_ID + '/pretrained_lstm.ckpt', global_step=i+1)
            print(('saved to %s' % save_path), flush=True)

    #writer.close()
    csvfile.close()
    total_time = (time.time() - total_time)
    print("*******************validation for train**********************")
    terations = 83
    sum = 0
    for i in range(terations):
        nextBatch, nextBatchLabels = getTestBatch(i)
        acc = sess.run([accuracy], {input_data: nextBatch, labels: nextBatchLabels})
        #print(acc)
        sum = sum+acc[0]
    print('Final Performance TotalTimeToTrain(s) : %d' % total_time)
    print('Final Accuracy acc : %f' % (sum/terations))