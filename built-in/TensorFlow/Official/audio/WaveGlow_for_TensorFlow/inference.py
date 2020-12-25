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
import tensorflow as tf
import numpy as np
from scipy.io import wavfile

import argparse
import os
import librosa
from audio_utils import melspectrogram
from params import hparams
from glow import WaveGlow

from ljspeech_to_tfrecords import get_tfrecords_dataset

from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveGlow Network')
    parser.add_argument('--raw_audio_file', type=str, default="./LJSpeech-1.1/wavs/LJ001-0001.wav", required=False,
                        help='local condition file')
    parser.add_argument('--wave_name', type=str, default='waveglow.wav')
    parser.add_argument('--restore_from', type=str, default="./logdir/waveglow/model.ckpt-180000",
                        help='restore model from checkpoint')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='sigma value for inference')
    return parser.parse_args()


def write_wav(waveform, sample_rate, filename):
    """

    :param waveform: [-1,1]
    :param sample_rate:
    :param filename:
    :return:
    """
    # TODO: write wave to 16bit PCM, don't use librosa to write wave
    y = np.array(waveform, dtype=np.float32)
    y *= 32767
    wavfile.write(filename, sample_rate, y.astype(np.int16))
    print('Updated wav file at {}'.format(filename))


def main():
    try:
        args = get_arguments()
        # read wave
        audio, sample_rate = librosa.load(args.raw_audio_file, sr=None, mono=True)

        if sample_rate != hparams.sample_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sample_rate, hparams.sample_rate))
        # compute mel spectrogram
        mel_spec = melspectrogram(audio)

        mel_spec = np.array(mel_spec, 'float32')
        assert mel_spec.size % float(hparams.num_mels) == 0.0, \
            'specified dimension %s not compatible with data' % (hparams.num_mels,)
        mel_spec = mel_spec.reshape((1,-1, hparams.num_mels))

        print(mel_spec)
        glow = WaveGlow(lc_dim=hparams.num_mels,
                        n_flows=hparams.n_flows,
                        n_group=hparams.n_group,
                        n_early_every=hparams.n_early_every,
                        n_early_size=hparams.n_early_size,
                        traindata_iterator=None,
                        testdata_iterator=None
                        )
        lc_placeholder = tf.placeholder(tf.float32, shape=[1, None, hparams.num_mels], name='lc')
        model_audio_ops = glow.infer(lc_placeholder, sigma=args.sigma)

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["mix_compile_mode"].b = True
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关

        sess = tf.Session(config=config)

        print("restore model")
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(sess, args.restore_from)
        print('restore model successfully!')

        audio_output = sess.run(model_audio_ops, feed_dict={lc_placeholder: mel_spec})
        audio_output = audio_output[0].flatten()
        print(audio_output)
        write_wav(audio_output, hparams.sample_rate, args.wave_name)
    except Exception:
        raise


if __name__ == '__main__':
    main()
