# Copyright 2021 Huawei Technologies Co., Ltd
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
import argparse
import sys

sys.path.append('.')

from ascendcv.runner.sess_config import get_sess_config
from ascendcv.utils.misc import set_global_random_seed
from ascendvsr.config import cfg
from ascendvsr.models import build_model


def get_args():
    parser = argparse.ArgumentParser(description="Ascend VSR Toolkit")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    return args


def dump_cfg(_cfg):
    cfg_str = _cfg.dump()
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)
    dump_file = os.path.join(cfg.output_dir, f"configure_{cfg.mode}.yaml")
    with open(dump_file, 'w') as f:
        f.write(cfg_str)
    print(_cfg)

if __name__ == '__main__':
    args = get_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cfg.device_ids))
    elif cfg.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    elif cfg.device == 'npu':
        import ascendcv.runner.npu_pkgs
    else:
        raise KeyError

    set_global_random_seed(cfg.random_seed)
    model = build_model(cfg)

    sess_cfg = get_sess_config(cfg.device, cfg.solver.xla, cfg.solver.mix_precision, cfg.rank_size>1)

    assert cfg.mode in ['train', 'eval', 'inference', 'freeze']
    dump_cfg(cfg)
    if cfg.mode == 'train':
        model.train(sess_cfg)
    elif cfg.mode == 'eval':
        model.evaluate(sess_cfg)
    elif cfg.mode == 'inference':
        model.inference(sess_cfg)
    elif cfg.mode == 'freeze':
        model.freeze(sess_cfg)
    else:
        raise KeyError
