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
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import time

int2name={}
txt=open('/home/zw/Downloads/tiny-imagenet-200/name2int.txt')
lines=txt.readlines()
for line in lines:
    line_s=line.strip('\n').split(' ')
    int2name.update({line_s[0]:line_s[1]})
txt.close()

session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = '1'
session_config.gpu_options.allow_growth = True
with tf.Session(graph=tf.Graph(), config=npu_config_proto(config_proto=session_config)) as sess:
    tf.saved_model.loader.load(sess,['serve'],'./models/imagenet-rgb/1546129741')
    graph = tf.get_default_graph()
    names=[n.name for n in tf.get_default_graph().as_graph_def().node]
    x=sess.graph.get_tensor_by_name('images:0')
    y=sess.graph.get_tensor_by_name('classes:0')
    imgl=os.listdir('/home/zw/Downloads/tiny-imagenet-200/test/images')
    for imgn in imgl:
        print(imgn)
        img=Image.open('/home/zw/Downloads/tiny-imagenet-200/test/images/'+imgn)
        #img.show('img')
        #break
        img.convert('RGB')
        bft=time.clock()
        #img=img.convert('L')
        #img=img.crop((145,153,481,489))
        image=np.array(img.resize([224,224],2),dtype=float).reshape(1,224,224,3)
        c_ = sess.run(y, feed_dict={x: image})
        aft=time.clock()
        print(aft-bft, ' ', imgn, ' class: ', int2name[str(c_[0])])
        #break

