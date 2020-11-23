
rm -rf /var/log/npu/slog/host-0/*



currentDir=$(cd "$(dirname "$0")"; pwd)


export JOB_ID=9999001
export RANK_SIZE=1
export RANK_ID=npu1p

export SLOG_PRINT_TO_STDOUT=0


#device id of NPU
device_phy_id=0

${currentDir}/train_1p_s8_beta.sh ${device_phy_id}  
