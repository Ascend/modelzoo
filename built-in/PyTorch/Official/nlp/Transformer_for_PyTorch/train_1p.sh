#! /bin/bash
DATA_DIR=./data/dataset/wmt14_en_de_joined_dict/
MODELDIR="./checkpoints/"
mkdir -p "$MODELDIR"
LOGFILE="$MODELDIR/log"
STAT_FILE="log.txt"

python3 -u train_1p.py \
  ./data/dataset/wmt14_en_de_joined_dict/ \
  --device-id 7\
  --arch transformer_wmt_en_de \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0003 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-sentences 128\
  --max-tokens 102400\
  --seed 1 \
  --save-dir $MODELDIR \
  --save-interval 1\
  --online-eval\
  --update-freq 1\
  --log-interval 1\
  --stat-file $STAT_FILE\
  --distributed-world-size 1\
  --amp\
  --amp-level O2