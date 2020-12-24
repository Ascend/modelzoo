import tensorflow as tf
from Deblur_Net import Deblur_Net
from mode import *
import argparse
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import os
import moxing as mox
parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ('true')

## Model specification
parser.add_argument("--channel", type = int, default = 3)
parser.add_argument("--n_feats", type = int, default = 64)
parser.add_argument("--num_of_down_scale", type = int, default = 2)
parser.add_argument("--gen_resblocks", type = int, default = 9)
parser.add_argument("--discrim_blocks", type = int, default = 3)

## Data specification
parser.add_argument("--data_url", type = str, default = "s3://DeblurGAN-tf/data")
parser.add_argument("--native_data", type = str, default = "./data")
parser.add_argument("--train_Sharp_path", type = str, default = "./data/train/sharp")
parser.add_argument("--train_Blur_path", type = str, default = "./data/train/blur")
parser.add_argument("--vgg_path", type = str, default = "./data/vgg19/vgg19.npy")
parser.add_argument("--model_path", type = str, default = "./data/model")
parser.add_argument("--logdir", type = str, default = "./data/log")

parser.add_argument("--train_url", type = str, default = "s3://deblur-gan/")
parser.add_argument("--ext", type = str, default = "./png")
parser.add_argument("--test_Sharp_path", type = str, default = "./val_sharp")
parser.add_argument("--test_Blur_path", type = str, default = "./val_blur")
parser.add_argument("--patch_size", type = int, default = 256)
parser.add_argument("--result_path", type = str, default = "./result")
parser.add_argument("--in_memory", type = str2bool, default = True)

## Optimization
parser.add_argument("--batch_size", type = int, default = 1)
parser.add_argument("--max_epoch", type = int, default = 300)
parser.add_argument("--learning_rate", type = float, default = 1e-4)
parser.add_argument("--decay_step", type = int, default = 150)
parser.add_argument("--test_with_train", type = str2bool, default = False)
parser.add_argument("--save_test_result", type = str2bool, default = False)

## Training or test specification
parser.add_argument("--mode", type = str, default = "train")
parser.add_argument("--critic_updates", type = int, default = 5)
parser.add_argument("--augmentation", type = str2bool, default = False)
parser.add_argument("--load_X", type = int, default = 640)
parser.add_argument("--load_Y", type = int, default = 360)
parser.add_argument("--fine_tuning", type = str2bool, default = False)
parser.add_argument("--log_freq", type = int, default = 1)
parser.add_argument("--model_save_freq", type = int, default = 20)
parser.add_argument("--test_batch", type = int, default = 1)
parser.add_argument("--pre_trained_model", type = str, default = "./")
parser.add_argument("--chop_forward", type = str2bool, default = False)
parser.add_argument("--chop_size", type = int, default = 8e4)
parser.add_argument("--chop_shave", type = int, default = 16)

args = parser.parse_args()
mox.file.copy_parallel(args.data_url, args.native_data)

model = Deblur_Net(args)
model.build_graph()

print("Build model!")

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep = 10)

if args.mode == 'train':
    train(args, model, sess, saver)

elif args.mode == 'test':
    f = open("test_results.txt", 'w')
    test(args, model, sess, saver, f, step = -1, loading = True)
    f.close()

elif args.mode == 'test_only':
    test_only(args, model, sess, saver)

print("model compute finished")
mox.file.copy_parallel(args.model_path, os.path.join(args.data_url, 'model'))
