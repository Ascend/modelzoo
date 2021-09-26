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
import matplotlib.pyplot as plt
from pylab import *
import cv2



def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
def visualize_feature_map(img_batch, path):
    feature_maps = np.squeeze(img_batch, axis=0)
    #print(feature_map.shape)
 
    feature_map_combination = []
    plt.figure()
    for l in range(10):
        feature_map=feature_maps[l]
        num_pic = feature_map.shape[2]
        row, col = get_row_col(num_pic)
    
        for i in range(0, 10):
            feature_map_split = feature_map[:, :, i]
            feature_map_combination.append(feature_map_split)
            plt.subplot(10, 10, l*10+i + 1)
            plt.imshow(feature_map_split)
            axis('off')
            #title('feature_map_{}'.format(i))
            #cv2.imwrite(path+'feature_map_{}'.format(i)+'.png',feature_map_split) 
    plt.savefig(path+'feature_map.png')
    plt.show()
 
    # 鍚勪釜鐗瑰緛鍥炬寜1锛1 鍙犲姞
    #feature_map_sum = sum(ele for ele in feature_map_combination)
    #plt.imshow(feature_map_sum)
    #plt.savefig(path+"feature_map_sum.png")


#txt=open('vis/name.txt','w')

imgs=['vis/0/C1_92.bmp','vis/1/noAug_8_22_42_397.bmp','vis/2/normMatch_3_noAug_8_22_42_397.bmp','vis/3/8_21_28_193_d.bmp']
layers=['ShuffleNetV2/Conv1/Relu:0','ShuffleNetV2/MaxPool/MaxPool:0','ShuffleNetV2/Stage2/concat:0','ShuffleNetV2/Stage3/concat:0','ShuffleNetV2/Stage4/concat:0','ShuffleNetV2/Conv5/Relu:0']

session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = '1'
session_config.gpu_options.allow_growth = True
with tf.Session(graph=tf.Graph(), config=npu_config_proto(config_proto=session_config)) as sess:
    tf.saved_model.loader.load(sess,['serve'],'./models/jiu-huan_1.0_9988/1551059747')
    graph = tf.get_default_graph()
    names=[n.name for n in tf.get_default_graph().as_graph_def().node]
    #for name in names:
     #   txt.write(name+'\n')
    #txt.close()
    x=sess.graph.get_tensor_by_name('images:0')
    featureMaps=[]
    for i in [1,2]:
        img=Image.open(imgs[i])
        img=img.convert('L')
        #img=img.crop((145,153,481,489))
        image=np.array(img.resize([224,224],Image.BILINEAR),dtype=float).reshape(224,224,1)
        for l in [0,2,3,4,5]:
            y=sess.graph.get_tensor_by_name(layers[l])
            conv_img = sess.run(y, feed_dict={x: image})
            feature_map = np.squeeze(conv_img, axis=0)
            featureMaps.append(feature_map[:,:,0:10])
    visualize_feature_map(featureMaps, 'vis/1/conv1/')
