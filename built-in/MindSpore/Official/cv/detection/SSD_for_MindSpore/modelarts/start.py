# coding=utf-8
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import glob
import logging
import os
from unittest import mock

import modelarts_utils as ma
import train as model_train
from export_wrapper import export


@mock.patch.object(model_train, 'get_args', ma.parse_args)
def train():
    model_train.main()


def get_last_ckpt():
    ckpt_pattern = os.path.join(ma.CACHE_TRAIN_OUT_URL, 'ckpt_*', "*.ckpt")
    ckpts = glob.glob(ckpt_pattern)
    if not ckpts:
        print(f"Cant't found ckpt in {ma.CACHE_TRAIN_OUT_URL}")
        return
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]


def export_air(args):
    ckpt = get_last_ckpt()
    if not ckpt:
        return
    export(args.device_id, ckpt, "AIR")


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = ma.parse_args()
    print("Training setting args:", args)

    try:
        import moxing as mox
        print('import moxing success.')

        # 改变工作目录，用于模型保存
        os.makedirs(ma.CACHE_TRAIN_OUT_URL, exist_ok=True)
        os.chdir(ma.CACHE_TRAIN_OUT_URL)

        train()
        export_air(args)
        mox.file.copy_parallel(ma.CACHE_TRAIN_OUT_URL, args.train_url)
    except ModuleNotFoundError:
        print('import moxing failed')
        train()


if __name__ == '__main__':
    main()
