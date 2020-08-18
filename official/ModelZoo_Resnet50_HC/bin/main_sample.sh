#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

device_group=$@
device_num=$#

touch ${currentDir}/main.log

corenum=`cat /proc/cpuinfo |grep "processor"|wc -l`
echo "cpu has ${corenum} cores"

for device_phy_id in ${device_group}
do
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] start: train.sh ${device_phy_id} & " >> ${currentDir}/main.log
    echo "taskset -c $((${device_phy_id}*${corenum}/8))-$(((${device_phy_id}+1)*${corenum}/8-1)) ${currentDir}/train.sh ${device_phy_id}" >> ${currentDir}/main.log
    taskset -c $((device_phy_id*${corenum}/8))-$(((device_phy_id+1)*${corenum}/8-1)) ${currentDir}/train.sh ${device_phy_id}  &
done

wait

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] all train.sh exit " >> ${currentDir}/main.log
