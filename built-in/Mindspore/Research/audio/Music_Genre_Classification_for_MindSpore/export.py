import numpy as np
from mindspore.train.serialization import export
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.musictagger import MusicTaggerCNN
from src.config import music_cfg as cfg

if __name__ == "__main__":
    network = MusicTaggerCNN()
    param_dict = load_checkpoint(cfg.checkpoint_path + "/" + cfg.model_name)
    load_param_into_net(network, param_dict)
    input = np.random.uniform(0.0, 1.0, size=[1, 1, 96,
                                              1366]).astype(np.float32)
    export(network,
           Tensor(input),
           file_name="{}/{}.air".format(cfg.checkpoint_path,
                                       cfg.model_name[:-5]),
           file_format="AIR")
