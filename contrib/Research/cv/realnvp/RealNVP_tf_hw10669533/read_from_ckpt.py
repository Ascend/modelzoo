# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf

import nn as real_nvp_nn
from model import model_spec as real_nvp_model_spec
from model import inv_model_spec as real_nvp_inv_model_spec
import cifar10_data as cifar10_data
import util
import plotting


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-z', '--num_gpus', type=int, default=1, help='Location for the dataset')  #for huawei, compat with nr_gpu

parser.add_argument('-i', '--data_url', type=str, default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--train_url', type=str, default='/tmp/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=int, default=0, help='Restore training from previous model checkpoint? 1 = Yes, 0 = No')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--sample_batch_size', type=int, default=16, help='How many images to process in paralell during sampling?')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)
# tf.random.set_seed(args.seed)
import torch as pt
pt.nn.MaxPool2d(2,2)
args.data_dir="."
args.save_dir=args.train_url

DataLoader = cifar10_data.DataLoader
train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True)
test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False)
obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
model_spec = real_nvp_model_spec
inv_model_spec = real_nvp_inv_model_spec
nn = real_nvp_nn

# create the model
model = tf.make_template('model', model_spec)
inv_model = tf.make_template('model', inv_model_spec, unique_name_='model')

x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
# run once for data dependent initialization of parameters
gen_par = model(x_init)

# sample from the model
x_sample = tf.placeholder(tf.float32, shape=(args.sample_batch_size, ) + obs_shape)
new_x_gen = inv_model(x_sample)
def sample_from_model(sess):
  x_gen = np.random.normal(0.0, 1.0, (args.sample_batch_size,) + obs_shape)
  new_x_gen_np = sess.run(new_x_gen, {x_sample: x_gen})
  return new_x_gen_np

# get loss gradients over multiple GPUs
xs = []
grads = []
loss_gen = []
loss_gen_test = []
all_params = tf.trainable_variables()

xs.append(tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape))

gen_par,jacs = model(xs[0])
loss_gen.append(nn.loss(gen_par, jacs))
# gradients
grads.append(tf.gradients(loss_gen[0], all_params))
# test
gen_par,jacs = model(xs[0])
loss_gen_test.append(nn.loss(gen_par, jacs))

# add gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])

optimizer =tf.train.AdamOptimizer(learning_rate=tf_lr)

bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)

# init & save
initializer = tf.initialize_all_variables()
saver = tf.train.Saver()

# input is scaled from uint8 [0,255] to float in range [-1,1]
def prepro(x):
  return np.cast[np.float32]((x - 127.5) / 127.5)

def compute_likelihood(xf):
  print ("computing likelihood of image with mean %f" % np.mean(xf))
  xfs = np.split(xf, args.nr_gpu)
  feed_dict = { xs[i]: xfs[i] for i in range(args.nr_gpu) }
  l = sess.run(bits_per_dim_test, feed_dict)
  return l

# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('starting training')
test_bpd = []
lr = args.learning_rate

with tf.Session() as sess:
    for epoch in range(args.max_epochs):
        begin = time.time()

        # init
        if epoch == 0:
            x = train_data.next(args.init_batch_size) # manually retrieve exactly init_batch_size examples
            train_data.reset() # rewind the iterator back to 0 to do one full epoch
            print('initializing the model...')
            sess.run(initializer,{x_init: prepro(x)})
            if args.load_params:
                ckpt_file = './tfcheck_1608121226.1290338/params_cifar.ckpt'
                print('restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)
            # util.show_all_variables() # code for debug

        # train for one epoch
        train_losses = []
        print ("Training (%d/%d) started" % (epoch, args.max_epochs))
        for t,x in enumerate(train_data):
          # prepro the data and split it for each gpu
          xf = prepro(x)
          xfs = np.split(xf, args.nr_gpu)
          lr *= args.lr_decay
          feed_dict = { tf_lr: lr }
          feed_dict.update({ xs[i]: xfs[i] for i in range(args.nr_gpu) })
          l,_ = sess.run([bits_per_dim, optimizer], feed_dict) # toooooo long
          # l = sess.run(bits_per_dim, feed_dict) # if want to debug, use this func to speed up (useless for result)
          train_losses.append(l)
        train_loss_gen = np.mean(train_losses)

        # compute likelihood over test split
        test_losses = []
        print ("Testing...")
        for x in test_data:
          xf = prepro(x)
          xfs = np.split(xf, args.nr_gpu)
          feed_dict = { xs[i]: xfs[i] for i in range(args.nr_gpu) }
          l = sess.run(bits_per_dim_test, feed_dict)
          test_losses.append(l)
        test_loss_gen = np.mean(test_losses)
        test_bpd.append(test_loss_gen)

        # log progress to console
        print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (epoch, time.time()-begin, train_loss_gen, test_loss_gen))
        sys.stdout.flush()

        if epoch % args.save_interval == 0:

            print ("Generating samples...")

      
            cwd = os.getcwd()  # current path
            t=str(time.time())
            os.makedirs("save_model_"+t,exist_ok=True)
            os.makedirs("save_testbpd_"+t,exist_ok=True)

            save_path = 'save_model_'+t+'/params_cifar.ckpt'
            osr_dir=os.path.join(cwd, 'save_model_'+t)
            osr=os.path.join(cwd, save_path)

            ret=saver.save(sess, osr) # main func  # output /home/work/workspace/device7/save_model/params_cifar.ckpt
            ck=tf.train.get_checkpoint_state(save_path)


            import boto3
            # import boto3.s3 as ss
            ACCESS_KEY = '7N2JK6JLDLW3DE3ESNWV'
            SECRET_KEY = 'cJfnmOiPNe3PSwqyadPMkPr7wyj4ltl6Ao8E6SRY'
            s3s = boto3.client(
                's3',
                aws_access_key_id=ACCESS_KEY,
                aws_secret_access_key=SECRET_KEY,
                endpoint_url='https://obs.cn-north-4.myhuaweicloud.com'
            )
            os.chdir(osr_dir)
            all_files = os.listdir(osr_dir)
            for ff in all_files:
                print(os.path.abspath(ff))
                if not os.path.isfile(os.path.abspath(ff)):
                    all_files.remove(ff)
                s3s.upload_file(os.path.abspath(ff), 'realnvp', 'oo/tfcheck_'+t+'/'+os.path.basename(ff))

            save_nppath = 'save_testbpd_'+t+'/test_bpd_cifar.npz'
            osd = os.path.join(cwd , save_nppath)
            osd_dir = os.path.join(cwd , 'save_testbpd_'+t)
            np.savez(osd, test_bpd=np.array(test_bpd)) # main func

            os.chdir(osd_dir)
            all_files = os.listdir(osd_dir)
            for ff in all_files:
                if not os.path.isfile(os.path.abspath(ff)):
                    all_files.remove(ff)
                s3s.upload_file(os.path.abspath(ff), 'realnvp', 'oo/npcheck_'+t+'/'+os.path.basename(ff))

            os.chdir(cwd)
