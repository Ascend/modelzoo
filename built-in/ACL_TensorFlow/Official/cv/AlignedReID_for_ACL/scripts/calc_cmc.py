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

import tensorflow as tf
import numpy as np
import cv2
import os
import random
import cmc

from PIL import Image 

print tf.__version__
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '100', 'batch size for training')
tf.flags.DEFINE_string('mode', 'top1', 'Mode train, val, test')

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

def main(argv=None):
    with tf.Session() as sess:
        if FLAGS.mode == 'top1':
            path = 'data'
            set = 'val'
            cmc_sum=np.zeros((100, 100), dtype='f')

            cmc_total = []
            do_times = 1

            for times in range(do_times):
                query_feature = []
                test_feature = []

                for i in range(100):
                    while True:
                          index_gallery = int(random.random() * 10)
                          index_temp = index_gallery
                          filepath_gallery = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, i, index_gallery)
                          if not os.path.exists(filepath_gallery):
                             continue
                          break
                    image1 = cv2.imread(filepath_gallery)
                    image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                    query_feature.append(image1)
    
                    while True:
                          index_gallery = int(random.random() * 10)
                          if index_temp == index_gallery:
                             continue
      
                          filepath_gallery = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, i, index_gallery)
                          if not os.path.exists(filepath_gallery):
                             continue
                          break
                    image1 = cv2.imread(filepath_gallery)
                    image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                    test_feature.append(image1)
                query_feature = np.array(query_feature)
                test_feature = np.array(test_feature)
                query_feature.astype('float32').tofile(os.path.join('query_bins','query.bin'))
                test_feature.astype('float32').tofile(os.path.join('test_bins','test.bin'))
                os.system("../Benchmark/out/benchmark --om AlignedReID_100batch.om --dataDir ./query_bins --modelType AlignedReID --outDir results --batchSize 1 --imgType bin --useDvpp 0")
                q_feat = np.fromfile("results/AlignedReID/davinci_query_output0.bin",dtype='float32').reshape(100,2048)
                
                os.system("../Benchmark/out/benchmark --om AlignedReID_100batch.om --dataDir ./test_bins --modelType AlignedReID --outDir results --batchSize 1 --imgType bin --useDvpp 0")
                test_feat = np.fromfile("results/AlignedReID/davinci_test_output0.bin",dtype='float32').reshape(100,2048)
    
                cmc_array = []
                tf_q_feat = tf.constant(q_feat)
                tf_test_feat = tf.constant(test_feat)
  
                h = tf.placeholder(tf.int32)
                pick = tf_q_feat[h]
                tf_q_feat = tf.reshape(pick,[1,2048])
                feat1 = tf.tile(tf_q_feat,[100,1])
                f = tf.square(tf.subtract(feat1 , tf_test_feat))
                d = tf.sqrt(tf.reduce_sum(f,1))
                for t in range(100):
                    feed_dict = {h: t}
                    D = sess.run(d,feed_dict=feed_dict)
                    cmc_array.append(D)
                cmc_array = np.array(cmc_array)
                cmc_score = cmc.cmc(cmc_array)
                cmc_sum = cmc_score + cmc_sum
                cmc_total.append(cmc_score)
                #print(cmc_score)
            cmc_sum = cmc_sum/do_times
            #print(cmc_sum)
            print('final cmc:')
            print(cmc_total)
            print('mean cmc:',np.mean(cmc_total))
        
if __name__ == '__main__':
    tf.app.run()
