#!/usr/bin/env bash

cd ${HOME}/workspace/
source venv/bin/activate
pip install -r requirements.txt --upgrade
python setup.py build
py.test tests --cov=resnet3d --cov-report term-missing
