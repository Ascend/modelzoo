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
from model import *
from data import *
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--train_data_path', default='./data/membrane/train')
parser.add_argument('--test_data_path', default='./data/membrane/test')
parser.add_argument('--predict_data_path', default='./data/membrane/predict')
args = parser.parse_args()

def main():
    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    myGene = trainGenerator(2,args.train_data_path,'image','label',data_gen_args,save_to_dir = None)

    model = unet()
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=2, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=300,epochs=args.epochs,callbacks=[model_checkpoint])

    testGene = testGenerator(args.test_data_path)
    results = model.predict_generator(testGene,30,verbose=2)
    saveResult(args.predict_data_path,results)

if __name__ == '__main__':
    # ***** npu modify begin *****
    global_config = tf.ConfigProto(log_device_placement=False)
    custom_op = global_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["dynamic_input"].b = 1
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    npu_keras_sess = set_keras_session_npu_config(config=global_config)
    # ***** npu modify end *****
    main()
    close_session(npu_keras_sess)

