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

#from npu_bridge.npu_init import *
from npu_bridge import *
import sys
from collections import Counter
import numpy as np
import tensorflow.keras as kr
if (sys.version_info[0] > 2):
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding('utf-8')
    is_py3 = False

def native_word(word, encoding='utf-8'):
    '如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码'
    if (not is_py3):
        return word.encode(encoding)
    else:
        return word

def native_content(content):
    if (not is_py3):
        return content.decode('utf-8')
    else:
        return content

def open_file(filename, mode='r'):
    "\n    常用文件操作，可在python2和python3间切换.\n    mode: 'r' or 'w' for read or write\n    "
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

def read_file(filename):
    '读取文件数据'
    (contents, labels) = ([], [])
    with open_file(filename) as f:
        for line in f:
            try:
                (label, content) = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return (contents, labels)

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    '根据训练集构建词汇表，存储'
    (data_train, _) = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)
    count_pairs = counter.most_common((vocab_size - 1))
    (words, _) = list(zip(*count_pairs))
    words = (['<PAD>'] + list(words))
    open_file(vocab_dir, mode='w').write(('\n'.join(words) + '\n'))

def read_vocab(vocab_dir):
    '读取词汇表'
    with open_file(vocab_dir) as fp:
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return (words, word_to_id)

def read_category():
    '读取分类目录，固定'
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return (categories, cat_to_id)

def to_words(content, words):
    '将id表示的内容转换为文字'
    return ''.join((words[x] for x in content))

def process_file(filename, word_to_id, cat_to_id, max_length=600):
    '将文件转换为id表示'
    (contents, labels) = read_file(filename)
    (data_id, label_id) = ([], [])
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if (x in word_to_id)])
        label_id.append(cat_to_id[labels[i]])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    return (x_pad, y_pad)

def batch_iter(x, y, batch_size=64):
    '生成批次数据'
    data_len = len(x)
    num_batch = int(((data_len - 1) / batch_size))
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = (i * batch_size)
        end_id = min(((i + 1) * batch_size), data_len)
        (yield (x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]))
