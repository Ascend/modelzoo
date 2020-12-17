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
# Copyright 2020 Huawei Technologies Co., Ltd
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

from __future__ import print_function
from six.moves import xrange
import os
os.environ['EXPERIMENTAL_DYNAMIC_PARTITION']="1"
os.system("pip install better_exceptions")
import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from model import VQVAE, _cifar10_arch
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

# The codes are borrowed from
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
DATA_DIR = '/data'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
def maybe_download_and_extract():
    import sys, tarfile
    from six.moves import urllib
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(DATA_DIR)

def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    record_bytes = 1 + 32*32*3

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [1]), tf.int32)
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [1],
                         [1 + 32*32*3]),
        [3, 32, 32])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result

def get_image(train=True,num_epochs=None):
    #maybe_download_and_extract()
    if train:
        filenames = [os.path.join(DATA_DIR, 'cifar10-bin', 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    else:
        filenames = [os.path.join(DATA_DIR, 'cifar10-bin', 'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(filenames,num_epochs=num_epochs)
    read_input = read_cifar10(filename_queue)
    return tf.cast(read_input.uint8image, tf.float32) / 255.0, tf.reshape(read_input.label,[])


def main(config,
         RANDOM_SEED,
         LOG_DIR,
         TRAIN_NUM,
         BATCH_SIZE,
         LEARNING_RATE,
         DECAY_VAL,
         DECAY_STEPS,
         DECAY_STAIRCASE,
         BETA,
         K,
         D,
         SAVE_PERIOD,
         SUMMARY_PERIOD,
         **kwargs):
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # >>>>>>> DATASET
    
    images=tf.placeholder(tf.float32,[8,32,32,3])

    # <<<<<<<

    # >>>>>>> MODEL
    with tf.variable_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        with tf.variable_scope('params') as params:
            pass
        net = VQVAE(learning_rate,global_step,BETA,images,K,D,_cifar10_arch,params,True)

    #with tf.variable_scope('valid'):
    #    params.reuse_variables()
    #    valid_net = VQVAE(None,None,BETA,valid_images,K,D,_cifar10_arch,params,False)

    with tf.variable_scope('misc'):
        # Summary Operations
        tf.summary.scalar('loss',net.loss)
        tf.summary.scalar('recon',net.recon)
        tf.summary.scalar('vq',net.vq)
        tf.summary.scalar('commit',BETA*net.commit)
        tf.summary.scalar('nll',tf.reduce_mean(net.nll))
        tf.summary.image('origin',images,max_outputs=4)
        tf.summary.image('recon',net.p_x_z,max_outputs=4)
        # TODO: logliklihood

        summary_op = tf.summary.merge_all()

        # Initialize op
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

       

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True #在昇腾AI处理器执行训练
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #关闭remap开关
    #custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    summary_writer.add_summary(config_summary.eval(session=sess))

    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for step in tqdm(xrange(TRAIN_NUM),dynamic_ncols=True):
            it,loss,_ = sess.run([global_step,net.loss,net.train_op],feed_dict={images:np.random.random((8,32,32,3))})

            if( it % SAVE_PERIOD == 0 ):
                net.save(sess,LOG_DIR,step=it)

            
            tqdm.write('[%5d] Loss: %1.3f'%(it,loss))


    except Exception as e:
        coord.request_stop(e)
    finally :
        net.save(sess,LOG_DIR)

        coord.request_stop()
        coord.join(threads)
        

def extract_z(MODEL,
              BATCH_SIZE,
              BETA,
              K,
              D,
              **kwargs):

    # >>>>>>> MODEL
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        x_ph = tf.placeholder(tf.float32,[8,32,32,3])
        net= VQVAE(None,None,BETA,x_ph,K,D,_cifar10_arch,params,False)

    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True #在昇腾AI处理器执行训练
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #关闭remap开关
    #custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
    net.load(sess,MODEL)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    try:
        ks = []
        ys = []
        while not coord.should_stop():
            #x,y = sess.run([images,labels])
            k = sess.run(net.k,feed_dict={x_ph:np.random.random((8,32,32,3))})
            ks.append(k)
            ys.append(y)
            print('.', end='', flush=True)
    except tf.errors.OutOfRangeError:
        print('Extracting Finished')

    ks = np.concatenate(ks,axis=0)
    ys = np.concatenate(ys,axis=0)
    np.savez(os.path.join(os.path.dirname(MODEL),'ks_ys.npz'),ks=ks,ys=ys)

    coord.request_stop()
    coord.join(threads)

def get_default_param():
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        'LOG_DIR':'./log/cifar10/a',
        'MODEL' : './log/cifar10/a/last.ckpt',

        'TRAIN_NUM' : 20, #Size corresponds to one epoch
        'BATCH_SIZE': 128,

        'LEARNING_RATE' : 0.0002,
        'DECAY_VAL' : 1.0,
        'DECAY_STEPS' : 20000, # Half of the training procedure.
        'DECAY_STAIRCASE' : False,

        'BETA':0.25,
        'K':10,
        'D':256,

        # PixelCNN Params
        'GRAD_CLIP' : 5.0,
        'NUM_LAYERS' : 12,
        'NUM_FEATURE_MAPS' : 64,

        'SUMMARY_PERIOD' : 10,
        'SAVE_PERIOD' : 10,
        'RANDOM_SEED': 0,
    }

if __name__ == "__main__":
    class MyConfig(dict):
        pass
    params = get_default_param()
    config = MyConfig(params)
    def as_matrix() :
        return [[k, str(w)] for k, w in config.items()]
    config.as_matrix = as_matrix

    main(config=config,**config)
    #extract_z(**config)
    #config['TRAIN_NUM'] = 10
    #config['LEARNING_RATE'] = 0.001
    #config['DECAY_VAL'] = 0.5
    #config['DECAY_STEPS'] = 100000
    #train_prior(config=config,**config)

    #test(MODEL='models/cifar10/last.ckpt',**config)
