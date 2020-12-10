#!/bin/bash
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

source ${currentDir}/npu_set_env_8p.sh

rm -rf ${currentDir}/log
mkdir -p ${currentDir}/log

device_group='0 1 2 3 4 5 6 7'
device_num=8

touch ${currentDir}/main.log

corenum=`cat /proc/cpuinfo |grep "processor"|wc -l`
echo "cpu has ${corenum} cores"

echo "divice group is ${device_group}"
for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> ${currentDir}/main.log
    echo "taskset -c $((${device_phy_id}*${corenum}/8))-$(((${device_phy_id}+1)*${corenum}/8-1)) ${currentDir}/train.sh ${device_phy_id}" >> ${currentDir}/main.log
    taskset -c $((device_phy_id*${corenum}/8))-$(((device_phy_id+1)*${corenum}/8-1)) ${currentDir}/train_sample.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> ${currentDir}/main.log
