#!/bin/bash
current_exec_path=$(pwd)
export DEVICE_ID=$2
export RANK_ID=$1
export RANK_SIZE=$3

ulimit -n 65535
 
cd ${current_exec_path}
rm -rf device_$RANK_ID
mkdir device_$RANK_ID
cd device_$RANK_ID
env  >log_$RANK_ID.log
# pytest -s /opt/npu/cgf/mindspore/faceReidToMe/testfile/test_distributed_cross_entropy_np.py::test_exec_forward -k DistributedCrossEntropy >>log_$RANK_ID.log  2>&1 &
# pytest -s /opt/npu/cgf/mindspore/faceReidToMe/testfile/test_reid_stage1_1024node.py::test_trains >>log_$RANK_ID.log  2>&1 &

python -u ${current_exec_path}/tools/train_multinpu.py  >test_deep$i.log 2>&1 &

