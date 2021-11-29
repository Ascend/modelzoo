export DDK_VERSION_FLAG=1.76.T1.0.B010
export WHICH_OP=GEOP
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export GE_USE_STATIC_MEMORY=1
export OPTION_EXEC_HCCL_FLAG=1
export HCCL_CONNECT_TIMEOUT=600
export SOC_VERSION=Ascend910

export ASCEND_HOME=/usr/local/Ascend


# Autotune
export FLAG_AUTOTUNE="" #"RL,GA"
export TUNE_BANK_PATH=/home/HwHiAiUser/custom_tune_bank
export ASCEND_DEVICE_ID=0
#export TUNE_OPS_NAME=
#export REPEAT_TUNE=True
#export ENABLE_TUNE_BANK=True
mkdir -p $TUNE_BANK_PATH
chown -R HwHiAiUser:HwHiAiUser $TUNE_BANK_PATH
