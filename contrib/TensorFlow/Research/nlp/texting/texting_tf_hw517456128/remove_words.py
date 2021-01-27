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

import sys
import nltk
from nltk.corpus import stopwords
from utils import clean_str, clean_str_sst, loadWord2Vec

if len(sys.argv) < 2:
    sys.exit("Use: python remove_words.py <dataset>")

dataset = sys.argv[1]
if 'SST' in dataset:
    func = clean_str_sst
else:
    func = clean_str

try:
    least_freq = sys.argv[2]
except:
    least_freq = 5
    print('using default least word frequency = 5')


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)


doc_content_list = []
with open('data/corpus/' + dataset + '.txt', 'rb') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))


word_freq = {}  # to remove rare words

for doc_content in doc_content_list:
    temp = func(doc_content)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in doc_content_list:
    temp = func(doc_content)
    words = temp.split()
    doc_words = []
    for word in words:
        if dataset == 'mr' or 'SST' in dataset:
            doc_words.append(word)
        elif word not in stop_words and word_freq[word] >= least_freq:
            doc_words.append(word)

    doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)


clean_corpus_str = '\n'.join(clean_docs)
with open('data/corpus/' + dataset + '.clean.txt', 'w') as f:
    f.write(clean_corpus_str)


len_list = []
with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
    for line in f.readlines():
        if line == '\n':
            continue
        temp = line.strip().split()
        len_list.append(len(temp))

print('min_len : ' + str(min(len_list)))
print('max_len : ' + str(max(len_list)))
print('average_len : ' + str(sum(len_list)/len(len_list)))
