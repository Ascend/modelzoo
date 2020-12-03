import argparse

from mindspore import Tensor
from mindspore.train.serialization import export
import numpy as np

from src.config import eval_config
from src.ds_cnn import ds_cnn
from src.models import load_ckpt

parser = argparse.ArgumentParser()

args, model_settings = eval_config(parser)
network = ds_cnn(model_settings, args.model_size_info)
load_ckpt(network, args.model_dir, False)
input = np.random.uniform(0.0, 1.0, size=[1, 1, model_settings['spectrogram_length'],
                                   model_settings['dct_coefficient_count']]).astype(np.float32)
export(network, Tensor(input), file_name=args.model_dir.replace('.ckpt', '.air'), file_format='AIR')