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

import random
import numpy as np
import tensorflow as tf

import pathlib
import os
import sys
import math
import glob
import multiprocessing as mp
from multiprocessing import Pool
import librosa
from audio_utils import melspectrogram
import argparse
from params import hparams
import random
import codecs


def load_and_preprocess_wav_file(sound_path, hparams):
    # read wave
    audio, sample_rate = librosa.load(sound_path, sr=None, mono=True)

    if sample_rate != hparams.sample_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sample_rate, hparams.sample_rate))

    audio_and_mel_spec_list=[]
    # Take segment

    if audio.size >= hparams.sample_size:
        for tid in range(math.ceil(audio.size/hparams.sample_size)+1):
            max_audio_start = audio.size -hparams.sample_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start + hparams.sample_size]
            # compute mel spectrogram
            mel_spec = melspectrogram(audio)

            mel_spec = np.array(mel_spec, 'float32')
            assert mel_spec.size % float(hparams.num_mels) == 0.0, \
                'specified dimension %s not compatible with data' % (hparams.num_mels,)
            mel_spec = mel_spec.reshape((-1, hparams.num_mels))

            audio_and_mel_spec_list.append((audio,mel_spec))
    else:
        audio = tf.pad(audio, (0, hparams.sample_size - audio.size), 'CONSTANT')
        # compute mel spectrogram
        mel_spec = melspectrogram(audio)

        mel_spec = np.array(mel_spec, 'float32')
        assert mel_spec.size % float(hparams.num_mels) == 0.0, \
            'specified dimension %s not compatible with data' % (hparams.num_mels,)
        mel_spec = mel_spec.reshape((-1, hparams.num_mels))
        print("mel_spec shape:",mel_spec.shape)
        audio_and_mel_spec_list.append((audio, mel_spec))
    return audio_and_mel_spec_list


# ### Serialize function and proto tf.Example
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def sound_example(sound_path, hparams):
  '''
  Creates a tf.Example message from wav, mel
  '''

  audio_and_mel_spec_list = load_and_preprocess_wav_file(sound_path, hparams)
  tf_example_list=[]
  for (audio,mel) in audio_and_mel_spec_list:
      features = {
          "wav": _float_feature(np.reshape(audio,[-1]).tolist()),
          "mel": _float_feature(np.reshape(mel,[-1]).tolist())
      }
      tf_example=tf.train.Example(features=tf.train.Features(feature=features))
      tf_example_list.append(tf_example)
  return tf_example_list


def single_tfrecords_writer(path_ds, record_file, n_samples, hparams):
    t=0
    with tf.io.TFRecordWriter(record_file) as writer:
        for path, sample in zip(path_ds, range(n_samples)):
            tf_example_list = sound_example(path, hparams)

            t+=len(tf_example_list)
            for ex in tf_example_list:
                writer.write(ex.SerializeToString())
    
    print(t)
def get_tfrecords_dataset(tfrecord_filepath_list,hparams,num_cpu_threads=5,
                          batch_size=16,is_training=True,
                          drop_remainder=True):
    """从tfrecords读取并解析数据"""
    lc_frames = hparams.sample_size // (hparams.transposed_conv_layer1_stride**2)+1
    name_to_features={
        "wav": tf.FixedLenFeature([hparams.sample_size,1], tf.float32),
        "mel": tf.FixedLenFeature([lc_frames,hparams.num_mels], tf.float32)
    }

    #解析一个样本文件
    def _parse_sound_function(record,name_to_features):
        x = tf.parse_single_example(record, name_to_features)
        return x

    def _decode_record(record, name_to_features):
        """解析一个record 样本数据"""
        example = _parse_sound_function(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example


    def input_fn():
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecord_filepath_list,dtype=tf.string))

            d = d.repeat()
            d = d.shuffle(buffer_size=len(tfrecord_filepath_list))

            # `cycle_length` is the number of parallel files that get read.

            cycle_length = min(num_cpu_threads, len(tfrecord_filepath_list))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=5000)
        else:
            # default test record has just one file
            d = tf.data.TFRecordDataset(tfrecord_filepath_list[0])
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=batch_size*30)
        return d

    return input_fn


def main(args):
    wave_files = glob.glob(os.path.join(args.wave_dir, '*.wav'))
    wave_files = sorted(wave_files, reverse=False)
    p = Pool(mp.cpu_count())

    results = []
    filelist = []
    for f in wave_files:
        filelist.append(f)

    # random select 200 ids as test
    test_set = filelist[:10]
    train_set = filelist[10:]
    random.shuffle(train_set)

    test_record_file = os.path.join(args.tfrecords_dir, hparams.test_file)
    single_tfrecords_writer(test_set, test_record_file, len(test_set), hparams)

    # ## Split Training Dataset in TFRecords Shards
    n_shards=20
    train_sample_num = len(train_set)
    sample_per_shard = math.ceil(train_sample_num / n_shards)
    print(train_sample_num, sample_per_shard)


    for idx_shard in range(n_shards):
        print("Currently saving {} samples in : ".format(sample_per_shard))
        fname = hparams.train_files + '_{}_of_{}.tfrecords'.format(idx_shard, n_shards - 1)
        current_path = os.path.join(args.tfrecords_dir, fname)
        print(current_path)
        if idx_shard!=n_shards-1:
            part_file_list=train_set[idx_shard*sample_per_shard:(idx_shard+1)*sample_per_shard]
        else:
            part_file_list = train_set[idx_shard * sample_per_shard:]
        p.apply_async(single_tfrecords_writer,args=(part_file_list,current_path,len(part_file_list),hparams))
    p.close()
    p.join()
    #print("train_totoal_sample_num:",train_totoal_sample_num)
    print("job done!")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave_dir', type=str,default="./LJSpeech-1.1/wavs/", help='wave directory')

    parser.add_argument('--tfrecords_dir', type=str, default="./data/tfrecords/",
                        help='tfrecords folder of the data')

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = get_arguments()

    if not os.path.exists(args.tfrecords_dir):
        os.makedirs(args.tfrecords_dir)
    random.seed(a=1234)
    main(args)


