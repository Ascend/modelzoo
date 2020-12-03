import os
import re
import codecs
from setuptools import find_packages, setup
from subprocess import Popen, PIPE

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
