# -*- coding: utf-8 -*-


bash   ./run_npu_env.sh
source ./run_npu_env.sh

export JOB_ID=80000
export  DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1
#export RANK_TABLE_FILE=./1p.json

python3 inference.py --raw_audio_file ./LJSpeech-1.1/wavs/LJ001-0001.wav  --restore_from ./logdir/waveglow/model.ckpt-180000