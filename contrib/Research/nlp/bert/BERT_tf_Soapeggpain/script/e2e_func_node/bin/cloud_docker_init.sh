#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

DEVICE_LIST=$@

export exec_type={MODE}

prog_exit()
{
    if [ x"${exec_type}" = xdocker ];
    then
        # stop slogd progress
        bash /usr/local/Ascend/driver/tools/docker_stop_post_sys.sh
    fi
}

# register prog_exit
trap "prog_exit" SIGTERM

if [ x"${exec_type}" = xdocker ];
then
    #set env
    . ${currentDir}/npu_set_env.sh

    # start slogd progress
    mkdir -p /var/log/npu/slog/slogd
    /usr/local/Ascend/driver/tools/docker/slogd &

    # start main.sh
    ${currentDir}/main.sh ${DEVICE_LIST} &

    # wait slogd stop
    flag=1
    while [ $flag -ne 0 ];
    do
        sleep 5;
        flag=`ps -ef | grep train.sh | grep -v grep | wc -l`
        ps -ef >> ${currentDir}/ps.log
        echo "" >> ${currentDir}/ps.log
    done
else
    RANK_ID=`cat ${currentDir}/npu_set_env.sh | grep "RANK_ID=" | awk -F"=" '{print $2}'`
    # start main.sh
    . ${currentDir}/npu_set_env.sh;export PROFILING_DIR=/var/log/npu/profiling/container/${RANK_ID};${currentDir}/main.sh ${DEVICE_LIST} &
    wait
fi

