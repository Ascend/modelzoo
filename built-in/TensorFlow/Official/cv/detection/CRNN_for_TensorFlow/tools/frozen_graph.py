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
# =============================================================================
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from crnn_model import crnn_net
import argparse
from config import global_config
from data_provider import tf_io_pipline_fast_tools
import numpy as np

CFG = global_config.cfg

#set checkpoint path
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path',default="./model/shadownet.ckpt-60000",help="set checkpoint file path")
    args,unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

def main():
    args = parse_args()
    tf.reset_default_graph()
    # modify input node
    batchsize = 64
    test_images = tf.placeholder(tf.float32, shape=[batchsize, 32, 100, 3],name="test_images")
    #build inference graph
    shadownet = crnn_net.ShadowNet(
        phase = 'test',
        hidden_nums = CFG.ARCH.HIDDEN_UNITS,
        layers_nums = CFG.ARCH.HIDDEN_LAYERS,
        num_classes = CFG.ARCH.NUM_CLASSES
    )
    #compute inference result
    test_inference_ret = shadownet.inference(
        inputdata = test_images,
        name = 'shadow_net',
        reuse = False
    )
    test_decoded, test_log_prob = tf.nn.ctc_greedy_decoder(
        test_inference_ret,
        CFG.ARCH.SEQ_LENGTH * np.ones(batchsize),
        merge_repeated = True
    )

    with tf.Session() as sess:
        #save unfrozen graph
        tf.train.write_graph(sess.graph_def, './', 'model.pb')
        #start to froze graph
        freeze_graph.freeze_graph(
            input_graph = './model.pb',
            input_saver = '',
            input_binary = False,
            input_checkpoint = args.ckpt_path,
            output_node_names = 'shadow_net/Cast',
            restore_op_name = 'save/restore_all',
            filename_tensor_name = 'save/Const:0',
            output_graph = 'shadownet_tf_%dbatch.pb'%batchsize,
            clear_devices = False,
            initializer_nodes = ''
        )
    print("Done!")

if __name__ == '__main__':
    main()