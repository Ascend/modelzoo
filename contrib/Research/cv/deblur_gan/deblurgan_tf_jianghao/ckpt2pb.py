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

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from Deblur_Net import Deblur_Net
import argparse
import os


def str2bool(v):
    return v.lower() in ('true')


def main():
    parser = argparse.ArgumentParser()
    ## Model specification
    parser.add_argument("--channel", type=int, default=3)
    parser.add_argument("--n_feats", type=int, default=64)
    parser.add_argument("--num_of_down_scale", type=int, default=2)
    parser.add_argument("--gen_resblocks", type=int, default=9)
    parser.add_argument("--discrim_blocks", type=int, default=3)

    ## Data specification
    parser.add_argument("--train_Sharp_path",
                        type=str,
                        default="s3://deblurgan/data/sharp/")
    parser.add_argument("--train_Blur_path",
                        type=str,
                        default="s3://deblurgan/data/blur")
    parser.add_argument("--test_Sharp_path",
                        type=str,
                        default="s3://deblurgan/data/sharp")
    parser.add_argument("--test_Blur_path",
                        type=str,
                        default="s3://deblurgan/data/blur")
    parser.add_argument("--vgg_path",
                        type=str,
                        default="s3://deblurgan/pre_train_model/vgg19.npy")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--result_path", type=str, default="./result")
    parser.add_argument("--model_path", type=str, default="./ascend")
    parser.add_argument("--in_memory", type=str2bool, default=True)
    parser.add_argument("--ckpt_path", type=str, default="DeblurGAN_last")
    ## Optimization
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--decay_step", type=int, default=150)
    parser.add_argument("--test_with_train", type=str2bool, default=True)
    parser.add_argument("--save_test_result", type=str2bool, default=False)

    ## Training or test specification
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--critic_updates", type=int, default=5)
    parser.add_argument("--augmentation", type=str2bool, default=False)
    parser.add_argument("--load_X", type=int, default=640)
    parser.add_argument("--load_Y", type=int, default=360)
    parser.add_argument("--fine_tuning", type=str2bool, default=False)
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--model_save_freq", type=int, default=50)
    parser.add_argument("--test_batch", type=int, default=1)
    parser.add_argument("--pre_trained_model", type=str, default="./model")
    parser.add_argument("--chop_forward", type=str2bool, default=False)
    parser.add_argument("--chop_size", type=int, default=8e4)
    parser.add_argument("--chop_shave", type=int, default=16)

    args = parser.parse_args()

    # build Generator
    tf.reset_default_graph()
    model = Deblur_Net(args)
    model.build_graph()
    print("Build model!")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))
        saver.restore(sess, os.path.join(args.model_path, args.ckpt_path))
        tf.train.write_graph(sess.graph_def, os.path.join(args.model_path, args.ckpt_path), \
        '/media/jianghao/Elements/SR/DeblurGAN-tf/DeblurGAN.pb', as_text=True)
        freeze_graph.freeze_graph(
            input_graph='/media/jianghao/Elements/SR/DeblurGAN-tf/DeblurGAN.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=os.path.join(args.model_path, args.ckpt_path),
            output_node_names='Cast_3',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=
            '/media/jianghao/Elements/SR/DeblurGAN-tf/DeblurGAN_frozen.pb',
            clear_devices=False,
            initializer_nodes='')
    print('done')


if __name__ == '__main__':
    main()
