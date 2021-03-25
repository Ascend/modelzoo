# -*-coding:utf8-*-

import os
import pickle
import random
import numpy as np


def create_generator(data_path):
    # load all the data into memory
    flist = [os.path.join(data_path, fi) for fi in os.listdir(data_path) if fi.endswith('.pkl')]
    samples = {}
    keys = []
    # the data is post processed
    for fi in flist:
        with open(fi, 'rb') as fr:
            sample = pickle.load(fr)
        base_name = os.path.basename(fi)
        samples[base_name] = sample
        keys.append(base_name)

    # prepare generation
    def gen_fn():
        while True:
            si = random.choice(keys)
            sp = samples[si]
            mel = sp['padded_mel']
            text = sp['padded_text']
            mask = sp['text_mask']  # boolean type
            gate = sp['mel_mask'].astype(np.int32)  # int type
            yield text, mask, mel, gate

    return gen_fn
