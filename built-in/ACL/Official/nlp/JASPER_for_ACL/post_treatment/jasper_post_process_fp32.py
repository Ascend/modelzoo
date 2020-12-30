# convert checkpoint 2 pb

# -*- coding: utf-8 -*-
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#   http://www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow.compat.v1 as tf
import numpy as np
import os
import string
import sys
import csv
S = string.ascii_lowercase

tf.disable_eager_execution()

def ctc(val, lenVal):
    #print(lenVal)
    val = np.transpose(val, (1,0,2))
    test_decoded, test_log_prob = tf.nn.ctc_greedy_decoder(val,lenVal * np.ones(1))
    with tf.Session() as sess:
        td, tlp = sess.run([test_decoded, test_log_prob])
        return td

def sparseToDense(t):
    return tf.Session().run(tf.sparse.to_dense(tf.sparse.reorder(t)))

def toString(v):
    res = ""
    for i in v:
       if i == 0:
           res += " "
       elif i == 27:
           res += "'"
       else:       
           res += S[i-1]
    return res
    
def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b.
    The code was copied from: http://hetland.org/coding/python/levenshtein.py
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def calc_jasper_infer_accuracy(outputDir, realFile):
    total_word_lev = 0.0
    total_word_count = 0.0
    fns = os.listdir(outputDir)
    with open(realFile, 'r') as file:
        reader = csv.reader(file)
        true_list = [row for row in reader]
        saveFimeName = "./output_npu" + ".txt"
        with open(saveFimeName, 'w', encoding='utf-8') as f:
            for k, i in enumerate(fns):
                outFile_split_list = i.split("_")
                wav_name = outFile_split_list[-4]

                val = np.fromfile(outputDir+i, np.float32).reshape((1, -1, 29))
                decoded_val = ctc(val, val.shape[1])
                denseDecoded_val = sparseToDense(decoded_val[0])
                predic_text = toString(denseDecoded_val[0])
                saveText = "dev-clean-wav/" + wav_name + "," + predic_text
                f.write(saveText)
                f.write("\r\n")

                true_text = ""
                for row in true_list :
                    if wav_name in row[0]:
                        true_text = row[2]
                        print("==== index:", k, "wav_name:", wav_name, "true_text:", true_text)
                        print("==== index:", k, "wav_name:", wav_name, "predic_text:", predic_text)
                        break
                total_word_lev += levenshtein(true_text.split(), predic_text.split())
                total_word_count += len(true_text.split())
    print("total_word_count:", total_word_count, "total_word_lev:", total_word_lev, "WER:", total_word_lev/total_word_count)
    with open('../output/predict_accuracy.txt', 'w', encoding='utf-8') as retFile:
        writeStr = "total_word_count:" + str(total_word_count) + "total_word_lev:" + str(total_word_lev) + "WER:" + str(total_word_lev/total_word_count)
        retFile.write(writeStr)
        retFile.write("\r\n")

if __name__ == "__main__":
    outputDir = sys.argv[1]
    realFile = sys.argv[2]
    calc_jasper_infer_accuracy(outputDir, realFile)

