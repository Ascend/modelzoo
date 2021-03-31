#export PYTHONPATH=$PYTHONPATH:/opt/npu/yangyang/NCF/
dir=`pwd`
currentDir=$(cd "$(dirname "$0")"; pwd)
export PYTHONPATH=/usr/local/Ascend/runtime/ops/op_impl/built-in/ai_core/tbe/:../../
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/x86_64-linux-gnu
PATH=$PATH:$HOME/bin
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:$PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export WHICH_OP=GEOP
export DDK_VERSION_FLAG=1.60.T17.B830
export SOC_VERSION=Ascend910
export NEW_GE_FE_ID=1
#export GE_AICPU_FLAG=1
#export DUMP_GRAPH_LEVEL=1
#export DUMP_GE_GRAPH=1
#export DUMP_OP=1
#export DISABLE_REUSE_MEMORY=1
#export TE_PARALLEL_COMPILER=0
#export SLOG_PRINT_TO_STDOUT=1
export JOB_ID=10086
export DEVICE_ID=0
export DEVICE_INDEX=0
#export PRINT_MODEL=1
export RANK_ID=0
export RANK_SIZE=1
#export RANK_TABLE_FILE=./new_rank_table_1p.json
export FUSION_TENSOR_SIZE=1000000000
rm *txt
rm -rf kernel_meta/*
rm -rf model_ckpt/*
rm -f core.*
rm -f /var/log/npu/slog/host-0/*
rm -f /var/log/npu/slog/device-0/*
rm -f /var/log/npu/slog/device-os-0/*
#rm -rf dump*
rm -f slog/*
ulimit -c 0
python3.7 -u ncf_estimator_main.py \
      --model_dir './model_ckpt' \
      --data_dir './movielens_data' \
      --dataset ml-20m \
      --train_epochs 1 \
      --batch_size 1048576 \
      --eval_batch_size 160000 \
      --learning_rate 0.00382059 \
      --num_factors 16 \
      --hr_threshold 1.0 \
      --accelerator '1980' \
      --print_freq 10 \
      --debug_steps 100 \
      --debug_mod \
      #--seed 1 \
      #--use_synthetic_data \
sleep 2
cp /var/log/npu/slog/host-0/* ./slog
cp /var/log/npu/slog/device-0/* ./slog
cp /var/log/npu/slog/device-os-0/* ./slog
#the train result is HR = 0.6876, NDCG = 0.4081
#the result in paper with same parameter is HR = 0.684, NDCG = 0.410
#19418 steps per epcoh
