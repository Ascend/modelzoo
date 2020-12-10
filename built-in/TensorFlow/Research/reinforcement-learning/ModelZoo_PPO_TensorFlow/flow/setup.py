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
#!/usr/bin/env python3
# flake8: noqa
"""Setup script for the Flow repository."""
from os.path import dirname, realpath
from setuptools import find_packages, setup, Distribution
import setuptools.command.build_ext as _build_ext
import subprocess
from flow.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


class build_ext(_build_ext.build_ext):
    """External buid commands."""

    def run(self):
        """Install traci wheels."""
        subprocess.check_call(
            ['pip', 'install',
             'https://akreidieh.s3.amazonaws.com/sumo/flow-0.4.0/'
             'sumotools-0.4.0-py3-none-any.whl'])


class BinaryDistribution(Distribution):
    """See parent class."""

    def has_ext_modules(self):
        """Return True for external modules."""
        return True


setup(
    name='flow',
    version=__version__,
    distclass=BinaryDistribution,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    description=("A system for applying deep reinforcement learning and "
                 "control to autonomous vehicles and traffic infrastructure"),
    long_description=open("README.md").read(),
    url="https://github.com/flow-project/flow",
    keywords=("autonomous vehicles intelligent-traffic-control"
              "reinforcement-learning deep-learning python"),
    install_requires=_read_requirements_file(),
    zip_safe=False,
)
