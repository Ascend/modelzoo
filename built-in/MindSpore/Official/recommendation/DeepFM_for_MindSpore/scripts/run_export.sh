#!/bin/sh

nohup python export.py --ckpt_file './checkpoint/deepfm-15_2582.ckpt' > ./ms_log/export_output.log 2>&1 &