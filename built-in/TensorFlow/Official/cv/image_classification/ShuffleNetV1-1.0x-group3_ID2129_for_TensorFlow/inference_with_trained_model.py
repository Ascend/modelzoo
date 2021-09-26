
# coding: utf-8
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

# In[ ]:


from npu_bridge.npu_init import *
import tensorflow as tf
import json
import cv2
import numpy as np
from PIL import Image

from architecture import shufflenet


# # Load label decoding

# In[ ]:


with open('data/integer_encoding.json', 'r') as f:
    encoding = json.load(f)
    
with open('data/wordnet_decoder.json', 'r') as f:
    wordnet_decoder = json.load(f)


# In[ ]:


decoder = {i: wordnet_decoder[n] for n, i in encoding.items()}


# # Load an image

# In[ ]:


image = cv2.imread('panda.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224), cv2.INTER_LINEAR)

Image.fromarray(image)


# # Load a trained model

# In[ ]:


tf.reset_default_graph()

raw_images = tf.placeholder(tf.uint8, [None, 224, 224, 3])
images = tf.to_float(raw_images)/255.0

logits = shufflenet(images, is_training=False, depth_multiplier='0.5')
probabilities = tf.nn.softmax(logits, axis=1)

ema = tf.train.ExponentialMovingAverage(decay=0.995)
variables_to_restore = ema.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)


# # Predict

# In[ ]:


with tf.Session(config=npu_config_proto()) as sess:
    saver.restore(sess, 'run00/model.ckpt-1331064')
    feed_dict = {raw_images: np.expand_dims(image, axis=0)}
    result = sess.run(probabilities, feed_dict)[0]


# In[ ]:


print('The most probable labels is:')
print(decoder[np.argmax(result)])


