# transformer post process
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
# WITHOOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import codecs
import argparse
import evaluation_utils
import numpy as np


parser = argparse.ArgumentParser(description='Calculate the BLEU score')
parser.add_argument('--realFile', type=str, default='../datasets/newstest2014.tok.de', help='real file for calculate accuracy')
parser.add_argument('--source_dir', type=str,  default='../result_Files', help='infer result folder')
parser.add_argument('--vocab', type=str, default='../datasets/vocab.share', help='vocable file')


def cal_score(file_bleu, real_file):
    score = evaluation_utils.evaluate(
            real_file,
            file_bleu,
            'bleu',
            subword_option=None)
    print("The bleu score is {}".format(score))


def mapIndex_to_string(index_array, vocab):

    def get_vocab_dict():
        filename = vocab
        with open(filename, 'r', encoding='UTF-8') as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        dic = {}
        for k, word in enumerate(content):
            dic[k] = word
        return dic

    def map_line(line, dict):
        str_line = ""
        # strList = []
        for i in line:
            if i == 2:
                break
            str_line += dict[i] + " "
            # strList.append(dict[i])
        # print(strList)
        return str_line

    res = []
    dic = get_vocab_dict()
    #for line in index_array:
    res.append(map_line(index_array, dic))
    return res


def convert_infer_out_to_str(source_dir, vocab, file_bleu):
    res = []
    print("====len listdir:", len(os.listdir(source_dir)))
    file_seg_str = "_".join(os.listdir(source_dir)[0].split("_")[:-2])
    print("====file_seg_str:", file_seg_str)
    for i in range(len(os.listdir(source_dir))):
        file = file_seg_str + "_" + str(i) + "_output0.bin"
        file_full_name = os.path.join(source_dir, file)
        val = np.fromfile(file_full_name, np.int32)
        #val = np.transpose(val, (1,0))
        print("===fileName:", file_full_name)
        print("===val:", val)
        strVal = mapIndex_to_string(val, vocab)
        str_val_tmp = "".join(strVal)
        #print("===strVal:", str_val_tmp.encode("utf-8").decode("latin1"))
        res += strVal
        # print(val.shape)
        #if i==0:
        #   print(val)
        #if i>-1:
        #   break
    out_file = file_bleu
    with codecs.open(out_file, "w", "utf-8-sig") as temp:
        for i in res:
            temp.write(i+"\n")
    print("Program hit the end successfully")


def post_process(source_dir, real_file, vocab, file_bleu):
    print("======start convert source file======")
    convert_infer_out_to_str(source_dir, vocab, file_bleu)
    print("======start calculate bleu======")
    cal_score(file_bleu, real_file)
    print("======End post process======")

if __name__ == "__main__":
    args = parser.parse_args()
    source_dir = args.source_dir
    vocab = args.vocab
    real_file = args.realFile
    file_bleu = "./infer_out_for_bleu"
    post_process(source_dir, real_file, vocab, file_bleu)
    


