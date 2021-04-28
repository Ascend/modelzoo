# Copyright 2020 Huawei Technologies Co., Ltd
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

# coding:utf-8
import tensorflow as tf
import sys
from tensorflow.python.framework import graph_util
from npu_bridge.estimator import npu_ops
import efficientnet_builder
import main_npu
from tensorflow.python.tools import freeze_graph
import argparse
import os

CUR_PATH = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    """parse args from command line"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', default=1,
                        help="""set checkpoint path""")
    parser.add_argument('--output_path', default=CUR_PATH,
                        help="""set output path""")
    parser.add_argument('--num_classes', default=1000, type=int,
                        help="""number of classes for datasets """)
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def main():
    args = parse_args()
    tf.reset_default_graph()
    # input image after normalize
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")

    override_params = {'data_format': 'channels_last', 'num_classes': args.num_classes}

    logits, _ = efficientnet_builder.build_model(
        inputs,
        model_name="efficientnet-b0",
        training=False,
        override_params=override_params,
        model_dir=args.output_path)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph = os.path.join(args.output_path, "efficientnet-b0_tf.pb")

    with tf.Session() as sess:
        # try to add ema
        ema = tf.train.ExponentialMovingAverage(decay=0.0)
        ema_vars = tf.trainable_variables() + tf.get_collection("moving_vars")
        for v in tf.global_variables():
            if "moving_mean" in v.name or "moving_variance" in v.name:
                ema_vars.append(v)
        var_dict = ema.variables_to_restore(ema_vars)

        saver = tf.train.Saver(var_dict, max_to_keep=1)
        saver.restore(sess, args.ckpt_path)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=["logits"])

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    print("Done.")


if __name__ == '__main__':
    main()
