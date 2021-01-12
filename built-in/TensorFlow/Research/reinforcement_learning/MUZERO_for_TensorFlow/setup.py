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
# ============================================================================

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import re
import codecs
from setuptools import find_packages, setup
from subprocess import Popen, PIPE
from glob import glob

here = os.path.abspath(os.path.dirname(__file__))
# print(here)

com_smi = Popen(['command -v nvidia-smi'], stdout=PIPE, shell=True)
com_out = com_smi.communicate()[0].decode("UTF-8")
allow_gpu = com_out != ""

install_requires = list()

with codecs.open(os.path.join(here, 'requirements.txt'), 'r') as rf:
    for line in rf:
        package = line.strip()
        install_requires.append(package)
if allow_gpu:
    install_requires.append("tensorflow-gpu==1.15.0")
else:
    install_requires.append("tensorflow==1.15.0")

with open(os.path.join(here, 'xt', '__init__.py')) as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

setup(
    name="xingtian",
    version=version,
    python_requires=">=3.5.*",
    install_requires=install_requires,
    include_package_data=True,
    packages=find_packages(),
    data_files=[
        ("xt", glob("xt/environment/sumo/*/sumo*")),
    ],
    description=" Reinforcement learning platform xingtian enables "
                "easy usage on the art Reinforcement Learning algorithms.",
    author="XingTian development team",
    url=" http://gitlab.huawei.com/ee/train/rl",
    entry_points={
        'console_scripts': [
            'xt_main=xt.main:main',
            # 'xt_train=xt.train:main',
            # 'xt_eval=xt.evaluate:main',
            # 'xt_launch=xt.act_launch:main'
            # 'xt_benchmark=xt.benchmarking:main'
        ],
    }
)
