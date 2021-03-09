#!/usr/bin/env bash

cd utils/pyvotkit
python3.7 setup.py build_ext --inplace
cd ../../

cd utils/pysot/utils/
python3.7 setup.py build_ext --inplace
cd ../../../
