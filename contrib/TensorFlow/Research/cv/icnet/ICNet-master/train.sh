export EXPERIMENTAL_DYNAMIC_PARTITION=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
python3.7 train.py --train-beta-gamma \
      --random-scale --random-mirror --dataset ade20k --filter-scale 2
