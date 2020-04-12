dir=`pwd`
currentDir=$(cd "$(dirname "$0")"; pwd)
export PYTHONPATH=/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe/:../
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/HiAI/fwkacllib/lib64/:/usr/local/HiAI/driver/lib64/common/:/usr/local/HiAI/driver/lib64/driver/:/usr/local/HiAI/add-ons/:/usr/lib/x86_64-linux-gnu
PATH=$PATH:$HOME/bin
export PATH=$PATH:/usr/local/HiAI/fwkacllib/ccec_compiler/bin:$PATH
export HiAI_OPP_PATH=/usr/local/HiAI/opp
export DDK_VERSION_FLAG=1.60.T17.B830
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910
export DUMP_GE_GRAPH=1
export DUMP_GRAPH_LEVEL=3
#export DUMP_OP=1
#export DISABLE_REUSE_MEMORY=1
#export PRINT_NODE=1
export JOB_ID=10086
export DEVICE_ID=0
export DEVICE_INDEX=0
#export PRINT_MODEL=1
export RANK_ID=0
export RANK_SIZE=1
#export RANK_TABLE_FILE=./new_rank_table_1p.json
export FUSION_TENSOR_SIZE=1000000000
rm *txt
rm -f slog/*
rm -rf kernel_meta/*
rm -rf model
rm -f core.*
rm -f /var/log/npu/slog/host-0/*
rm -f /var/log/npu/slog/device-0/*
rm -f /var/log/npu/slog/device-os-0/*
rm -rf dump*
ulimit -c 0
cd ../criteo

python3.7 -u widedeep_main_record_multigpu_fp16_huifeng.py 10000
cp /var/log/npu/slog/host-0/* ./slog
cp /var/log/npu/slog/device-0/* ./slog
cp /var/log/npu/slog/device-os-0/* ./slog
