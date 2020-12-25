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
# -*- coding: utf-8 -*-

import librosa
import librosa.filters
import numpy as np
from scipy import signal
from params import hparams


def preemphasis(x):
    return signal.lfilter([1, -hparams.preemphasis], [1], x)


def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)


def melspectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    mel_spec = _normalize(S)
    return mel_spec.T


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
    n_fft = hparams.n_fft
    hop_length = hparams.hop_length
    win_length = hparams.win_length
    return n_fft, hop_length, win_length


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    n_fft = hparams.n_fft
    return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels,fmin=0.0,fmax=8000.0)


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0.0, 1.0)
