# -*- coding:utf-8 -*-
"""Setuptools of roma."""

import setuptools

setuptools.setup(
    name='roma',
    version="0.9.1",
    packages=["roma"],
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        "pyyaml>=5.1.2",
        "vega"
    ],
    author='Noah Enabling Engineering Dept',
    author_email='',
    description='AutoML',
    license='MIT',
    url='http://gitlab.huawei.com/ee/train/automl',
)
