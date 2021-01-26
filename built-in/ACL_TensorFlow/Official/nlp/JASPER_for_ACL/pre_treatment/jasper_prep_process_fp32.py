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

import os
import pandas as pd
import math
import resampy as rs
import sys
import h5py
import numpy as np
import scipy.io.wavfile as wave
import tensorflow as tf
import csv

# import librosa
filename = "dev-clean-wav/1988-147956-0000.wav"

### INPUT FEATURES CONFIG ####
# input_type = "spectrogram"
# num_audio_features = 96
BACKENDS = []
try:
  import python_speech_features as psf
  BACKENDS.append('psf')
except ImportError:
  pass
try:
  import librosa
  BACKENDS.append('librosa')
except ImportError:
  pass


### PREPROCESSING CACHING CONFIG ###
train_cache_features = False
eval_cache_features = True
cache_format = 'hdf5'
cache_regenerate = False

WINDOWS_FNS = {"hanning": np.hanning, "hamming": np.hamming, "none": None}
params = {
    "cache_features": train_cache_features,  
    "cache_format": cache_format,
    "cache_regenerate": cache_regenerate,
    "backend":"librosa",
    'input_type':"logfbank",
    'num_audio_features':64,
    'window_size':0.02,
    'window_stride':0.01,
    'norm_per_feature':True,
    'sample_freq':16000,
    "dither": 1e-5,
    "mode":"in",
    'max_duration':-1.0,
    'min_duration':-1.0,
    'batch_size':1,
    'num_audio_features':64,
    'dtype':tf.float16,
    'dataset_files':['./datasets/librivox-dev-clean.csv']
    #'dataset_files':['/data/librispeech/a.csv']


    }

#====================================
class PreprocessOnTheFlyException(Exception):
  """ Exception that is thrown to not load preprocessed features from disk;
  recompute on-the-fly.
  This saves disk space (if you're experimenting with data input
  formats/preprocessing) but can be slower.
  The slowdown is especially apparent for small, fast NNs."""
  pass


class RegenerateCacheException(Exception):
  """ Exception that is thrown to force recomputation of (preprocessed) features
  """
  pass
def _parse_audio_transcript_element(element):
  audio_filename, transcript = element
  if not six.PY2:
    transcript = str(transcript, 'utf-8')
    audio_filename = str(audio_filename, 'utf-8')


  if self.params.get("syn_enable", False):
    audio_filename = audio_filename.format(np.random.choice(self.params["syn_subdirs"]))

  source, audio_duration = get_speech_features_from_file(
      audio_filename,
      params
  )
  return source.astype(params['dtype'].as_numpy_dtype()), \
      np.int32([len(source)]), \
      np.float32([audio_duration])


def split_data(data):
  num_workers = 1
  worker_id = 0
  if params['mode'] != 'train' and num_workers is not None:
    size = len(data)
    start = size // num_workers * worker_id
    if worker_id == num_workers - 1:
      end = size
    else:
      end = size // num_workers * (worker_id + 1)
    return data[start:end]
  else:
    return data

def get_preprocessed_data_path(filename, params):

  if isinstance(filename, bytes):  # convert binary string to normal string
    filename = filename.decode('ascii')

  filename = os.path.realpath(filename)  # decode symbolic links

  ## filter relevant parameters # TODO is there a cleaner way of doing this?
  print(list(params.keys()))
  ignored_params = ["cache_features", "cache_format", "cache_regenerate",
                    "vocab_file", "dataset_files", "shuffle", "batch_size",
                    "max_duration",
                    "mode", "interactive", "autoregressive", "char2idx",
                    "tgt_vocab_size", "idx2char", "dtype"]
  def fix_kv(text):
    """ Helper function to shorten length of filenames to get around
    filesystem path length limitations"""
    text = str(text)
    text = text.replace("speed_perturbation_ratio", "sp") \
        .replace("noise_level_min", "nlmin", ) \
        .replace("noise_level_max", "nlmax") \
        .replace("add_derivatives", "d") \
        .replace("add_second_derivatives", "dd")
    return text

  # generate the identifier by simply concatenating preprocessing key-value
  # pairs as strings.
  preprocess_id = "-".join(
      [fix_kv(k) + "_" + fix_kv(v) for k, v in params.items() if
       k not in ignored_params])

  preprocessed_dir = os.path.dirname(filename).replace("wav",
                                                       "preprocessed-" +
                                                       preprocess_id)
  preprocessed_path = os.path.join(preprocessed_dir,
                                   os.path.basename(filename).replace(".wav",
                                                                      ""))

  # create dir if it doesn't exist yet
  if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)
  print(preprocessed_path)
  return preprocessed_path

def load_features(path, data_format):
  """ Function to load (preprocessed) features from disk

  Args:
      :param path:    the path where the features are stored
      :param data_format:  the format in which the features are stored
      :return:        tuple of (features, duration)
      """
  print("=========",(path, data_format))
  if data_format == 'hdf5':
    with h5py.File(path + '.hdf5', "r") as hf5_file:
      features = hf5_file["features"][:]
      duration = hf5_file["features"].attrs["duration"]
  elif data_format == 'npy':
    features, duration = np.load(path + '.npy')
  elif data_format == 'npz':
    data = np.load(path + '.npz')
    features = data['features']
    duration = data['duration']
  else:
    raise ValueError("Invalid data format for caching: ", data_format, "!\n",
                     "options: hdf5, npy, npz")
  print("========",(features, duration))
  return features, duration

def get_speech_features(signal, sample_freq, params):

  backend = params.get('backend', 'psf')

  features_type = params.get('input_type', 'spectrogram')
  num_features = params['num_audio_features']
  window_size = params.get('window_size', 20e-3)
  window_stride = params.get('window_stride', 10e-3)
  augmentation = params.get('augmentation', None)

  if backend == 'librosa':
    print("backend is ",backend)
    window_fn = WINDOWS_FNS[params.get('window', "hanning")]
    dither = params.get('dither', 0.0)
    num_fft = params.get('num_fft', None)
    norm_per_feature = params.get('norm_per_feature', False)
    mel_basis = params.get('mel_basis', None)
    gain = params.get('gain')
    mean = params.get('features_mean')
    std_dev = params.get('features_std_dev')
    mel_basis = params.get('mel_basis', None)
    features, duration = get_speech_features_librosa(
        signal, sample_freq, num_features, features_type,
        window_size, window_stride, augmentation, window_fn=window_fn,
        dither=dither, norm_per_feature=norm_per_feature, num_fft=num_fft,
        mel_basis=mel_basis, gain=gain, mean=mean, std_dev=std_dev
    )
    print("$$",features_type,num_features,window_size,window_stride,augmentation,window_fn,dither,num_fft,norm_per_feature,mel_basis,gain,mean,std_dev)
  else:
    pad_to = params.get('pad_to', 8)
    features, duration = get_speech_features_psf(
        signal, sample_freq, num_features, pad_to, features_type,
        window_size, window_stride, augmentation
    )

  return features, duration

def augment_audio_signal(signal_float, sample_freq, augmentation):

  if 'speed_perturbation_ratio' in augmentation:
    stretch_amount = -1
    if isinstance(augmentation['speed_perturbation_ratio'], list):
      stretch_amount = np.random.choice(augmentation['speed_perturbation_ratio'])
    elif augmentation['speed_perturbation_ratio'] > 0:
      # time stretch (might be slow)
      stretch_amount = 1.0 + (2.0 * np.random.rand() - 1.0) * \
                       augmentation['speed_perturbation_ratio']
    if stretch_amount > 0:
      signal_float = rs.resample(
          signal_float,
          sample_freq,
          int(sample_freq * stretch_amount),
          filter='kaiser_best',
      )

  # noise
  if 'noise_level_min' in augmentation and 'noise_level_max' in augmentation:
    noise_level_db = np.random.randint(low=augmentation['noise_level_min'],
                                       high=augmentation['noise_level_max'])
    signal_float += np.random.randn(signal_float.shape[0]) * \
                    10.0 ** (noise_level_db / 20.0)

  return signal_float

def get_speech_features_librosa(signal, sample_freq, num_features,
                                features_type='spectrogram',
                                window_size=20e-3,
                                window_stride=10e-3,
                                augmentation=None,
                                window_fn=np.hanning,
                                num_fft=None,
                                dither=0.0,
                                norm_per_feature=False,
                                mel_basis=None,
                                gain=None,
                                mean=None,
                                std_dev=None):
  
  signal = normalize_signal(signal.astype(np.float32), gain)
  print("signal is ",signal)
  if augmentation:
    signal = augment_audio_signal(signal, sample_freq, augmentation)

  audio_duration = len(signal) * 1.0 / sample_freq
  print("augmentation is ",augmentation)
  print("audio_duration is ",audio_duration)
  n_window_size = int(sample_freq * window_size)
  n_window_stride = int(sample_freq * window_stride)
  num_fft = num_fft or 2**math.ceil(math.log2(window_size*sample_freq))
  print(n_window_size,n_window_stride,n_window_stride)
  print("dither is ",dither)
  # if dither > 0:
  #   signal += dither*np.random.randn(*signal.shape)
  print("new signal is ",signal)
  if features_type == 'spectrogram':    
    # ignore 1/n_fft multiplier, since there is a post-normalization
    powspec = np.square(np.abs(librosa.core.stft(
        signal, n_fft=n_window_size,
        hop_length=n_window_stride, win_length=n_window_size, center=True,
        window=window_fn)))
    # remove small bins
    powspec[powspec <= 1e-30] = 1e-30
    features = 10 * np.log10(powspec.T)

    assert num_features <= n_window_size // 2 + 1, \
      "num_features for spectrogram should be <= (sample_freq * window_size // 2 + 1)"

    # cut high frequency part
    features = features[:, :num_features]
    #print("features is ", features)
  elif features_type == 'mfcc':
    signal = preemphasis(signal, coeff=0.97)
    S = np.square(
            np.abs(
                librosa.core.stft(signal, n_fft=num_fft,
                                  hop_length=int(window_stride * sample_freq),
                                  win_length=int(window_size * sample_freq),
                                  center=True, window=window_fn
                )
            )
        )
    features = librosa.feature.mfcc(sr=sample_freq, S=S,
        n_mfcc=num_features, n_mels=2*num_features).T
  elif features_type == 'logfbank':
    signal = preemphasis(signal,coeff=0.97)
    print("new single is ",signal)
    S = np.abs(librosa.core.stft(signal, n_fft=num_fft,
                                 hop_length=int(window_stride * sample_freq),
                                 win_length=int(window_size * sample_freq),
                                 center=True, window=window_fn))**2.0
    if mel_basis is None:
      mel_basis = librosa.filters.mel(sample_freq, num_fft, n_mels=num_features,
                                      fmin=0, fmax=int(sample_freq/2))
    features = np.log(np.dot(mel_basis, S) + 1e-20).T
  else:
    raise ValueError('Unknown features type: {}'.format(features_type))

  norm_axis = 0 if norm_per_feature else None
  if mean is None:
    mean = np.mean(features, axis=norm_axis)
  if std_dev is None:
    std_dev = np.std(features, axis=norm_axis)

  features = (features - mean) / std_dev

  if augmentation:
    n_freq_mask = augmentation.get('n_freq_mask', 0)
    n_time_mask = augmentation.get('n_time_mask', 0)
    width_freq_mask = augmentation.get('width_freq_mask', 10)
    width_time_mask = augmentation.get('width_time_mask', 50)

    for idx in range(n_freq_mask):
      freq_band = np.random.randint(width_freq_mask + 1)
      freq_base = np.random.randint(0, features.shape[1] - freq_band)
      features[:, freq_base:freq_base+freq_band] = 0
    for idx in range(n_time_mask):
      time_band = np.random.randint(width_time_mask + 1)
      if features.shape[0] - time_band > 0:
        time_base = np.random.randint(features.shape[0] - time_band)
        features[time_base:time_base+time_band, :] = 0

  return features, audio_duration

def normalize_signal(signal, gain=None):
  """
  Normalize float32 signal to [-1, 1] range
  """
  if gain is None:
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
  return signal * gain

def preemphasis(signal, coeff=0.97):
  return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def save_features(features, duration, path, data_format, verbose=False):
  """ Function to save (preprocessed) features to disk

  Args:
      :param features:            features
      :param duration:            metadata: duration in seconds of audio file
      :param path:                path to store the data
      :param data_format:              format to store the data in ('npy',
      'npz',
      'hdf5')
  """
  if verbose: 
    print("Saving to: ", path)

  if data_format == 'hdf5':
    with h5py.File(path + '.hdf5', "w") as hf5_file:
      dset = hf5_file.create_dataset("features", data=features)
      dset.attrs["duration"] = duration
  elif data_format == 'npy':
    np.save(path + '.npy', [features, duration])
  elif data_format == 'npz':
    np.savez(path + '.npz', features=features, duration=duration)
  else:
    raise ValueError("Invalid data format for caching: ", data_format, "!\n",
                     "options: hdf5, npy, npz")

def get_speech_features_from_file(wavname, params, data_file, out_dir):
    #read wav
    print(wavname,params)
    cache_features = params.get('cache_features', False)
    cache_format = params.get('cache_format', 'hdf5')
    cache_regenerate = params.get('cache_regenerate', False)
    print("==",cache_features,cache_format,cache_regenerate)
    try:
        if not cache_features:
          raise PreprocessOnTheFlyException(
            "on-the-fly preprocessing enforced with 'cache_features'==True")

        if cache_regenerate:
          raise RegenerateCacheException("regenerating cache...")
        print("==========")
        preprocessed_data_path = get_preprocessed_data_path(wavname, params)
        features, duration = load_features(preprocessed_data_path,
                                        data_format=cache_format)
        print("========%%%%%%%%%",features, duration)
    except PreprocessOnTheFlyException:
        if isinstance(wavname, str):
            filename = os.path.join(data_file,wavname)
        else:
            filename = os.path.join(data_file,wavname.decode("utf-8"))
        
        sample_freq, signal = wave.read(filename)
        print("====",sample_freq, signal)

        if sample_freq != params['sample_freq']:
            raise ValueError(
           ("The sampling frequency set in params {} does not match the "
               "frequency {} read from file {}").format(params['sample_freq'],
                                                    sample_freq, filename))
        features, duration = get_speech_features(signal, sample_freq, params)
  
    except (OSError, FileNotFoundError, RegenerateCacheException):
        if isinstance(wavname, str):
            filename = os.path.join(data_file,wavname)
        else:
            filename = os.path.join(data_file,wavname.decode("utf-8"))
        sample_freq, signal = wave.read(filename)
        print("&====",sample_freq, signal)
        # check sample rate
        if sample_freq != params['sample_freq']:
            raise ValueError(
            ("The sampling frequency set in params {} does not match the "
            "frequency {} read from file {}").format(params['sample_freq'],
                                                        sample_freq, filename)
        )
        features, duration = get_speech_features(signal, sample_freq, params)

        preprocessed_data_path = get_preprocessed_data_path(filename, params)
        save_features(features, duration, preprocessed_data_path,
                    data_format=cache_format)
    
    #save_bin_file 
    name = wavname.split('/')[1]
    print("====",features, features[0].shape, features[0].dtype)
    print("****file %s length %d", (name,len(features)) )
    if len(features) > 2336:
      outsize = len(features) - 2336
      zero_length = 0
      fact_len = 2336
      features_new = features[:2336]

    else:
      zero_length = 2336 - len(features)
      fact_len = len(features)
    
      zero_tensor = np.zeros((zero_length,64))
      print("++++++",features.shape, zero_tensor.shape)
      features_new = np.insert(features, fact_len, zero_tensor, axis=0)
      print("++++input",features_new.shape)
    input_tensor = features_new[np.newaxis,np.newaxis,:,:]
    print("====",input_tensor)

    print("=======features",input_tensor,type(input_tensor),input_tensor.shape,input_tensor.dtype,len(input_tensor[0][0]),type(len(input_tensor[0][0])))
    for i in range(1,params['batch_size']):
      input_tensor = np.insert(input_tensor, 0, features_new, axis=0)
    input_tensor = np.array(input_tensor,dtype=np.float32)

    print("==========",input_tensor[0][0].dtype, input_tensor[0][0].shape)  
    input_tensor = np.array(input_tensor,dtype=np.float32)
    print("==========",input_tensor[0][0].dtype, input_tensor[0][0].shape)  
    print("=======features",input_tensor.shape,input_tensor.dtype,input_tensor[0][0].shape,input_tensor[0][0].dtype)

    input_length = [1,]
    input_array = np.array(input_length,dtype=np.float32)
    insert_array = np.array(input_length,dtype=np.float32)
    reshape_fact_len = math.ceil(fact_len / 2)
    for i in range(1,reshape_fact_len):
      input_array = np.insert(input_array, 0, insert_array, axis=0)
    #有效长度不足1168的，后面补0
    input_0 = [0,]
    input_array0 = np.array(input_0, dtype=np.float32)
    for i in range(reshape_fact_len,1168):
      input_array = np.append(input_array, input_array0, axis=0)
    print("=======input_array",input_array.shape,input_array.dtype)

    input_reshape = input_array[np.newaxis,:]
    for i in range(1,params['batch_size']):
      input_reshape = np.insert(input_reshape, 0, input_array, axis=0)
    print("=======input_length",input_reshape,type(input_reshape),input_reshape.shape,input_array.dtype)

    input_dir = out_dir + "/input_0"
    if not os.path.exists(input_dir):
      os.makedirs(input_dir)

    input_reshape_dir = out_dir + "/input_reshape"
    if not os.path.exists(input_reshape_dir):
      os.makedirs(input_reshape_dir)

    input_tensor.tofile(input_dir + "/" + name + "_input_data.bin")
    input_reshape.tofile(input_reshape_dir + "/" + name + "_input_data.bin")

def pre_process(dataset_files, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    csv_file = dataset_files + "/librivox-dev-clean.csv"
    with open(csv_file,'r') as file:
      reader = csv.reader(file)
      col = [row[0] for row in reader]
      for wavname in col[1:]:
        get_speech_features_from_file(wavname, params, dataset_files, out_dir)

if __name__ == "__main__":
    path = sys.argv[1]
    out_dir = sys.argv[2]
    pre_process(path, out_dir)
