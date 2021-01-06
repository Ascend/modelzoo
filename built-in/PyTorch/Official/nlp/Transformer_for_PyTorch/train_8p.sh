#! /bin/bash
DATA_DIR=./data/dataset/wmt14_en_de_joined_dict/
MODELDIR="./checkpoints_8p/"
mkdir -p "$MODELDIR"
LOGFILE="$MODELDIR/log"
STAT_FILE="log.txt"

export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
export GLOBAL_LOG_LEVEL=3


python3 train_8p.py $DATA_DIR \
  --arch transformer_wmt_en_de \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --addr 'XX.XXX.XXX.XXX' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-sentences 128\
  --max-tokens 102400 \
  --seed 1 \
  --save-dir $MODELDIR \
  --stat-file $STAT_FILE\
  --log-interval 1\
  --amp\
  --amp-level O2
