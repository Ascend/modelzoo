# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
import os
import random
import threading
import codecs
import queue

import sys


import librosa
import numpy as np
from params import hparams

from multiprocessing import Process,Queue,Pool
import multiprocessing
import time

def read_binary_lc(file_path, dimension):
    f = open(file_path, 'rb')
    features = np.fromfile(f, dtype=np.float32)
    f.close()
    assert features.size % float(dimension) == 0.0,\
        'specified dimension %s not compatible with data' % (dimension,)
    features = features.reshape((-1, dimension))
    return features


def read_wave_and_lc_features(filelist_scpfile, wave_dir, lc_dir):
    filelist = []
    with codecs.open(filelist_scpfile, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            file_id = line
            filelist.append(file_id)

    random.shuffle(filelist)
    for file_id in filelist:
        wave_path = os.path.join(wave_dir, file_id + '.wav')
        lc_path = os.path.join(lc_dir, file_id + '.mel')

        # read wave
        audio, _ = librosa.load(wave_path, sr=hparams.sample_rate, mono=True)
        audio = audio.reshape(-1, 1)

        # read local condition
        lc_features = read_binary_lc(lc_path, hparams.num_mels)

        yield audio, lc_features, file_id


class DataReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 coord,
                 filelist,
                 wave_dir,
                 lc_dir,
                 queue_size=512,
                 prefetch_size=30*5):
        self.coord = coord
        self.filelist = filelist
        self.wave_dir = wave_dir
        self.lc_dir = lc_dir
        self.lc_dim = hparams.num_mels
        self.lc_frames = hparams.sample_size // hparams.upsampling_rate
        # recompute a sample size
        self.sample_size = self.lc_frames * hparams.upsampling_rate
        self.upsample_rate = hparams.upsampling_rate
        self.threads = []
        # self.queue = queue.Queue(maxsize=queue_size)
        self.audios=np.empty([0, self.sample_size, 1])
        self.lcs=np.empty([0, self.lc_frames, self.lc_dim])
        self.prefetch_size=prefetch_size

        self.manager = multiprocessing.Manager()
        # 父进程创建Queue，并传给各个子进程：
        self.queue  = self.manager.Queue(maxsize=queue_size)
        allparams=[]
        # allparams.append(self.coord)
        allparams.append(self.filelist)
        allparams.append(self.wave_dir)
        allparams.append(self.lc_dir)
        allparams.append(self.lc_dim)
        allparams.append(self.lc_frames)
        allparams.append(self.sample_size)
        allparams.append(self.upsample_rate)
        allparams.append(self.audios)
        allparams.append(self.lcs)
        allparams.append(self.prefetch_size)

        self.paramlist = self.manager.list(allparams)

    def prefetch(self,queue,lock,paramlist):
        sub_lc_dim = paramlist[4]
        sub_lc_frames = paramlist[5]
        sub_sample_size = paramlist[6]
        sub_upsample_rate = paramlist[7]
        sub_audios = paramlist[8]
        sub_lcs = paramlist[9]
        sub_prefetch_size = paramlist[10]

        while True:
            if len(sub_audios)<sub_prefetch_size:
                audio, lc = queue.get(block=True)
                audio = np.reshape(audio, [1, sub_sample_size, 1])
                lc = np.reshape(lc, [1, sub_lc_frames, sub_lc_dim])
                sub_audios = np.concatenate([sub_audios, audio], axis=0)
                sub_lcs = np.concatenate([sub_lcs, lc], axis=0)
            else:
                time.sleep(0.1)

    def dequeue(self, num_elements):
        # batch_audio = np.empty([0, self.sample_size, 1])
        # batch_lc = np.empty([0, self.lc_frames, self.lc_dim])
        # for i in range(num_elements):
        #     audio, lc = self.queue.get(block=True)
        #     audio = np.reshape(audio, [1, self.sample_size, 1])
        #     lc = np.reshape(lc, [1, self.lc_frames, self.lc_dim])
        #     batch_audio = np.concatenate([batch_audio, audio], axis=0)
        #     batch_lc = np.concatenate([batch_lc, lc], axis=0)
        #
        # return batch_audio, batch_lc
        while True:
            if len(self.audios)>=num_elements:
                batch_audio=self.audios[:num_elements]
                batch_lc=self.lcs[:num_elements]
                self.audios=self.audios[num_elements:]
                self.lcs = self.lcs[num_elements:]

                return batch_audio, batch_lc
            else:
                print("data not enough ,sleep..")
                time.sleep(0.01)





    def thread_main(self,queue,lock, paramlist ,pid):
        # sub_coord=paramlist[0]
        sub_filelist=paramlist[1]
        sub_wave_dir=paramlist[2]
        sub_lc_dir = paramlist[3]
        sub_lc_dim = paramlist[4]
        sub_lc_frames = paramlist[5]
        sub_sample_size = paramlist[6]
        sub_upsample_rate = paramlist[7]


        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = read_wave_and_lc_features(sub_filelist,
                                                 sub_wave_dir,
                                                 sub_lc_dir)

            for audio, lc_features, file_id in iterator:
                print("pid:%d iter a audio in file %s..."%(pid,file_id))
                if sub_coord.should_stop():
                    stop = True
                    break

                # force align wave & local condition
                if len(audio) > len(lc_features) * sub_upsample_rate:
                    # clip audio
                    audio = audio[:len(lc_features) * sub_upsample_rate, :]
                elif len(audio) < len(lc_features) * sub_upsample_rate:
                    # clip local condition and audio
                    audio_frames = len(audio) // sub_upsample_rate
                    frames = min(audio_frames, len(lc_features))
                    audio = audio[:frames*sub_upsample_rate, :]
                    lc_features = lc_features[:frames, :]
                else:
                    pass

                # add randomness for the data-generator
                frames = len(lc_features)
                if frames > sub_lc_frames:
                    max_frame_start = frames - sub_lc_frames
                    lc_start = random.randint(0, max_frame_start)

                    audio = audio[lc_start*sub_upsample_rate:, :]
                    lc_features = lc_features[lc_start:, :]

                while len(audio) >= sub_sample_size and len(lc_features) >= sub_lc_frames:
                    audio_piece = audio[:sub_sample_size, :]
                    lc_piece = lc_features[:sub_lc_frames, :]
                    lock.acquire()  # 加上锁
                    queue.put([audio_piece, lc_piece])
                    lock.release()  # 释放锁

                    audio = audio[sub_sample_size:, :]
                    lc_features = lc_features[sub_lc_frames:, :]

    def start_threads(self, n_threads=5):

        lock = self.manager.Lock()  # 初始化一把锁
        pool = Pool()
        for nid in range(n_threads):
            pool.apply_async(self.thread_main, args=(self.queue,lock,self.paramlist,nid))
        pool.apply_async(self.prefetch, args=(self.queue,lock,self.paramlist))
        pool.close()
        pool.join()

    # for nid in range(n_threads):
    #         thread = threading.Thread(target=self.thread_main, args=(nid,))
    #         thread.daemon = True  # Thread will close when parent quits.
    #         thread.start()
    #         self.threads.append(thread)
    #
    #     return self.threads
