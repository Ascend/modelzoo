#!/bin/bash

wget --no-check-certificate https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
tar zxf dac.tar.gz
rm -f dac.tar.gz

mkdir test_raw_data/
mv ./*.txt test_raw_data/