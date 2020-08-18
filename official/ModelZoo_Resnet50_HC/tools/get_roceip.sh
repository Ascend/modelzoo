#!/bin/bash

device_id=$1

currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

ostype=`uname -m`
if [ x"${ostype}" = xaarch64 ];
then
    GET_DEVICE_IP=`${currentDir}/turing_dsmi_arm func_dsmi_get_device_ip_address 1 ${device_id} 1 ${device_id}`
else
    GET_DEVICE_IP=`${currentDir}/turing_dsmi_x86 func_dsmi_get_device_ip_address 1 ${device_id} 1 ${device_id}`
fi

DEVICE_IP=`echo ${GET_DEVICE_IP} | grep "device_ip =" | awk -F " " '{print $(NF-3)}'`

echo "ROCE_IP=${DEVICE_IP}"
