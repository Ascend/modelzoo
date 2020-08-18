
rm -rf *.log
rm -rf /var/log/npu/slog/host-0/*


CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
source ${CURRENT_DIR}/env.sh

# user env
export JOB_ID=9999001
export RANK_TABLE_FILE=${CURRENT_DIR}/8p.json
export RANK_SIZE=8
#export RANK_INDEX=0
export RANK_ID=npu8p

export SLOG_PRINT_TO_STDOUT=0


# run one script for 8p training
device_group="0 1 2 3 4 5 6 7"
for device_phy_id in ${device_group}
do
       echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> main.log
   	 ${CURRENT_DIR}/train_s16_r1.sh ${device_phy_id}  &
	done

wait


#echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> ${CURRENT_DIR}/main.log
echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> main.log



# run one script for 8p training
device_group="0 1 2 3 4 5 6 7"

for device_phy_id in ${device_group}
do
        echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> main.log
         ${CURRENT_DIR}/train_s16_r2.sh ${device_phy_id}  &
done

wait



#echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> ${CURRENT_DIR}/main.log
echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> main.log


