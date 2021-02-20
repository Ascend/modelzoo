# !/bin/bash

DEVICE_ID=$1
CKPT=$2
DEVICE_RANK=1

# set env
export MOX_USE_NPU=1
export FUSION_TENSOR_SIZE=2000000000
export MOX_USE_TF_ESTIMATOR=0
export MOX_USE_TDT=1

export HEARTBEAT=1
export CONITNUE_TRAIN=true
export LOG_DIR=./log

export ASCEND_GLOBAL_EVENT_LEVEL=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TF_CPP_MIN_LOG_LEVEL=3

# Turn profiling on
export JOB_ID=123456789
export DEVICE_ID=${DEVICE_ID}
export DEVICE_INDEX=${DEVICE_ID}
export RANK_ID=${DEVICE_ID}
export RANK_SIZE=${DEVICE_RANK}
if [ ${DEVICE_RANK} -gt 1 ]; then
    export RANK_TABLE_FILE=scripts/${DEVICE_RANK}p.json
fi

echo "[INFO] test training. For the given configuration, the step time should be about 200 ms (the 1st print should not count)."
rm -rf kernel_meta

python3 tools/main.py \
    --config-file configs/edvr.yaml \
    solver.checkpoint_interval 5000 \
    solver.print_interval 20 \
    solver.lr_schedule.total_steps [100] \
    rank_size ${DEVICE_RANK}

echo "[INFO] test training done"

echo "[INFO] test evaluation. For the given configuration, the inference time should be about 110 ms (maybe less). The PSNR should be around 30.24dB."
rm -rf kernel_meta

python3 tools/main.py \
    --config-file configs/edvr.yaml \
    mode eval \
    data.data_dir 'data/reds' \
    data.eval_in_size 180,320 \
    checkpoint ${CKPT}
echo "[INFO] test evaluation done"

echo "[INFO] test freeze. "
echo "[INFO] (1/2) batchsize = 1, input dim = 5"
rm -rf kernel_meta

python3 tools/main.py \
    --config-file configs/edvr.yaml \
    mode freeze \
    model.input_format_dimension 5 \
    model.convert_output_to_uint8 False \
    data.eval_batch_size 1 \
    checkpoint ${CKPT}

echo "[INFO] (2/2) batchsize = None, input dim = 4"
rm -rf kernel_meta

python3 tools/main.py \
    --config-file configs/edvr.yaml \
    mode freeze \
    model.input_format_dimension 4 \
    model.convert_output_to_uint8 True \
    data.eval_batch_size -1 \
    checkpoint ${CKPT}

echo "[INFO] test freeze done."

echo "[INFO] test inference. The inference time is about 110ms (maybe less)."
echo "[INFO] (1/2) input dimension = 5, output dtype = uint8"
python3 tools/main.py \
    --config-file configs/edvr.yaml \
    mode inference \
    data.data_dir 'data/reds' \
    data.eval_in_size 180,320 \
    model.convert_output_to_uint8 True \
    checkpoint ${CKPT}


echo "[INFO] (2/2) input dimension = 4, output dtype = float32"
python3 tools/main.py \
    --config-file configs/edvr.yaml \
    mode inference \
    data.data_dir 'data/reds' \
    data.eval_in_size 180,320 \
    model.input_format_dimension 4 \
    checkpoint ${CKPT}
