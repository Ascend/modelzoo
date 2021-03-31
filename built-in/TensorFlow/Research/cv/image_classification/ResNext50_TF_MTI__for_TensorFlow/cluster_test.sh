#!/bin/bash
#set -x
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] current dir: ${currentDir}"

chown -R HwHiAiUser:HwHiAiUser ${currentDir}
chmod -R 770 ${currentDir}

version=`cat /usr/local/Ascend/version.info | grep "Version" | cut -d '.' -f 4`
current_path=`pwd`

function test_exec()
{   
    inputcsv=$1
    execnum=$2
    mode=$3
    rank_id=$4
    rank_table_file=$5
    host_device=$6
    for eachcsv in ${inputcsv}
    do
        i=1
        while read line
        do
            #echo "====================Next Case========================="
            if [ ${i} -eq 1 ] ;#skip the first line
            then
                #echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] skip the first line."
                let i++
                continue
            fi
                
            execflag=`echo ${line:0:1}`
            if [ x"${execflag}" = x"#" ] ; 
            then
                #echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] no need exec line[${i}], skip."
                let i++
                continue
            fi

            CASENUM=`echo ${line} | awk -F"," '{print $1}'`
            if [ x"${execnum}" != x"${CASENUM}" -a x"${execnum}" != x"all" ] ;
            then
                #echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] no need exec line[${i}], skip."
                let i++
                continue
            fi

            currtime=`date +%Y%m%d%H%M%S`
            exec_path=${current_path}/result/${rank_id}
            mkdir -p ${exec_path}/config

            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] host mount dir: ${exec_path}"
            CASENAME=`echo ${line} | awk -F"," '{print $2}'`
            ENV_SET=`echo ${line} | awk -F"," '{print $3}'`
            JOB_ID=`echo ${line} | awk -F"," '{print $4}'`
            PROFILING_MODE=`echo ${line} | awk -F"," '{print $5}'`
            PROFILING_OPTIONS=`echo ${line} | awk -F"," '{print $6}'`
            FP_POINT=`echo ${line} | awk -F"," '{print $7}'`
            BP_POINT=`echo ${line} | awk -F"," '{print $8}'`
            AICPU_PROFILING_MODE=`echo ${line} | awk -F"," '{print $9}'`

            RANK_ID=`echo ${rank_id}`
            DEVICE_ID=`echo ${host_device} | sed 's#\+#'\ '#g'`
            DEVICE_NUM=`echo ${DEVICE_ID} | awk '{print NF}'`
            RUN_ALGORITHM_CMD=`echo ${line} | awk -F"," '{print $12}'`
            CHECKPOINT_DIR=`echo ${line} | awk -F"," '{print $13}'`

            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] CASENUM               = ${CASENUM}"
            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] CASENAME              = ${CASENAME}"
            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] JOB_ID                = ${JOB_ID}"
            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] ENV_SET               = ${ENV_SET}"
            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] PROFILING_MODE        = ${PROFILING_MODE}"
	    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] AICPU_PROFILING_MODE  = ${AICPU_PROFILING_MODE}"
            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] RANK_ID               = ${RANK_ID}"
            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] DEVICE_ID             = ${DEVICE_ID}"
            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] DEVICE_NUM            = ${DEVICE_NUM}"

            if [ x"${mode}" = xdocker ];
            then
                RANK_TABLE_FILE=/d_solution/config/${rank_table_file}
            else
                RANK_TABLE_FILE=${exec_path}/config/${rank_table_file}
                RUN_ALGORITHM_CMD="${current_path}${RUN_ALGORITHM_CMD}"
            fi

            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] RANK_TABLE_FILE       = ${RANK_TABLE_FILE}"

            RANK_INDEX=`echo ${RANK_ID} | awk -F"-" '{print $NF}'`
            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] RANK_INDEX            = ${RANK_INDEX}"

            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] RUN_ALGORITHM_CMD     = ${RUN_ALGORITHM_CMD}"
            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] CHECKPOINT_BASE       = ${CHECKPOINT_DIR}"

            #modify cloud_docker_init.sh
            cp ${current_path}/bin/cloud_docker_init.sh          ${exec_path}/cloud_docker_init.sh
            sed -i 's#{MODE}#'${mode}'#g'                           ${exec_path}/cloud_docker_init.sh

            #modify npu_set_env.sh
            cp ${current_path}/bin/${ENV_SET}                           ${exec_path}/npu_set_env.sh
            sed -i 's#{JOB_ID}#'${JOB_ID}'#g'                              ${exec_path}/npu_set_env.sh
            sed -i 's#{PROFILING_MODE}#'${PROFILING_MODE}'#g'              ${exec_path}/npu_set_env.sh
            sed -i 's#{AICPU_PROFILING_MODE}#'${AICPU_PROFILING_MODE}'#g'       ${exec_path}/npu_set_env.sh
            sed -i 's#{RANK_TABLE_FILE}#'${RANK_TABLE_FILE}'#g'            ${exec_path}/npu_set_env.sh
            sed -i 's#{RANK_ID}#'${RANK_ID}'#g'                            ${exec_path}/npu_set_env.sh
            sed -i 's#{RANK_INDEX}#'${RANK_INDEX}'#g'                      ${exec_path}/npu_set_env.sh

            sed -i 's#{PROFILING_OPTIONS}#'${PROFILING_OPTIONS}'#g'        ${exec_path}/npu_set_env.sh
            sed -i 's#{FP_POINT}#'${FP_POINT}'#g'                          ${exec_path}/npu_set_env.sh
            sed -i 's#{BP_POINT}#'${BP_POINT}'#g'                          ${exec_path}/npu_set_env.sh

            #modify main.sh
            cp ${current_path}/bin/main_sample.sh                ${exec_path}/main.sh

            #modify train.sh
            cp ${current_path}/bin/train_sample.sh                 ${exec_path}/train.sh
            sed -i 's#{RUN_ALGORITHM_CMD}#'"${RUN_ALGORITHM_CMD}"'#g' ${exec_path}/train.sh
            sed -i 's#{CHECKPOINT_DIR}#'"${CHECKPOINT_DIR}"'#g'       ${exec_path}/train.sh

            #modify hccl.json
            cp ${current_path}/config/${rank_table_file}    ${exec_path}/config/${rank_table_file}

            RANK_SIZE=`grep "device_count" ${exec_path}/config/${rank_table_file} | awk -F"\"" '{print $4}'`
            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] RANK_SIZE             = ${RANK_SIZE}"
            sed -i 's#{RANK_SIZE}#'${RANK_SIZE}'#g'                ${exec_path}/npu_set_env.sh

            if [ x"${mode}" = xdocker ];
            then
                docker_run
            elif [ x"${mode}" = xhost ];
            then
                host_run
            else
                echo "[`date +%Y%m%d-%H:%M:%S`] [ERROR] not support mode[${mode}] "
                exit 1
            fi

            result_check

            let i++
        done < ${eachcsv}	
    done
}

function docker_run()
{
    docker stop $(docker ps -a | awk 'NR>20{print $1}')  1>/dev/null 2>&1
    docker rm -f $(docker ps -a | awk 'NR>20{print $1}') 1>/dev/null 2>&1

    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] docker process start "

    #pre-smoke-need-rm
    rm -rf /var/log/npu/profiling/container/${RANK_ID}
    rm -rf /var/log/npu/profiling/JOB*

    container_list=""
    mkdir -p /var/log/npu/slog/container/${RANK_ID}
    su - HwHiAiUser -c "mkdir -p /var/log/npu/profiling/container/${RANK_ID}" 1>/dev/null 2>&1
    mkdir -p /var/log/npu/dump/container/${RANK_ID}

    rm -rf /var/log/npu/docker_slog_${RANK_ID}/
    mkdir -p /var/log/npu/docker_slog_${RANK_ID}

    chown -R HwHiAiUser:HwHiAiUser /var/log/npu
    chmod 775 /var/log/npu/slog/container/${RANK_ID}
    chmod 775 /var/log/npu/profiling/container/${RANK_ID}
    chmod 775 /var/log/npu/dump/container/${RANK_ID}
    chmod 775 /var/log/npu/docker_slog_${RANK_ID}

    chown -R HwHiAiUser:HwHiAiUser ${exec_path}
    chmod -R 777 ${exec_path}

    DEVICE_DEV=""
    for each_id in ${DEVICE_ID}
    do
        DEVICE_DEV=`echo "${DEVICE_DEV}" --device=/dev/davinci${each_id}`
    done

    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] Docker CMD:"
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] docker run -ti --user=HwHiAiUser:HwHiAiUser ${DEVICE_DEV} --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v ${exec_path}:/d_solution -v ${current_path}/data:/data -v ${current_path}/code:/code -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/container/${RANK_ID}:/var/log/npu/slog -v /var/log/npu/profiling/container/${RANK_ID}:/var/log/npu/profiling -v /var/log/npu/dump/container/${RANK_ID}:/var/log/npu/dump -v /var/log/npu/docker_slog_${RANK_ID}:/usr/slog ${dockerimage} /bin/bash -c "/d_solution/cloud_docker_init.sh ${DEVICE_ID}""

    container_id=`docker run -d --user=HwHiAiUser:HwHiAiUser ${DEVICE_DEV} --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v ${exec_path}:/d_solution -v ${current_path}/data:/data -v ${current_path}/code:/code -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/container/${RANK_ID}:/var/log/npu/slog -v /var/log/npu/profiling/container/${RANK_ID}:/var/log/npu/profiling -v /var/log/npu/dump/container/${RANK_ID}:/var/log/npu/dump -v /var/log/npu/docker_slog_${RANK_ID}:/usr/slog ${dockerimage} /bin/bash -c "/d_solution/cloud_docker_init.sh ${DEVICE_ID}"`

    container_list="${container_list} `echo ${container_id} | cut -c1-12`"
    timeused=0
    while true
    do
        res=""
        for c_id in ${container_list}
        do
            c_status=`docker ps | grep ${c_id}`
            if [ x"${c_status}" != x ] ;
            then
                echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] train job is working, wait more 5s "
                sleep 5
                let timeused+=5
                break
            fi
        done

        if [ x"${c_status}" = x ] ;
        then
            break
        fi
    done

    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] docker process end "
}

function host_run()
{
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] host process start "

    #pre-smoke-need-rm
    rm -rf /var/log/npu/slog/*
    rm -rf /var/log/npu/profiling/container/*
    rm -rf /var/log/npu/profiling/JOB*
    su - HwHiAiUser -c "mkdir -p /var/log/npu/profiling/container/${RANK_ID}" 1>/dev/null 2>&1
    export PROFILING_DIR=/var/log/npu/profiling/container/${RANK_ID}

    chown -R HwHiAiUser:HwHiAiUser ${exec_path}
    chmod -R 777 ${exec_path}

    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] Host CMD:"
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] ${exec_path}/cloud_docker_init.sh ${DEVICE_ID} & "

    ${exec_path}/cloud_docker_init.sh ${DEVICE_ID} &
    workshell=$!
    while true
    do
        ret=`ps -ef | grep cloud_docker_init.sh | grep ${workshell} | grep -v grep`
        if [ x"${ret}" = x ];
        then
            break
        else
            echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] train job is working, wait more 5s "
            sleep 5
        fi
    done

    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] host process end "
}

function result_check()
{
    failed_device=""
    for each_id in ${DEVICE_ID}
    do
        if [ ! -f ${exec_path}/train_${each_id}.log ];
        then
            echo "[`date +%Y%m%d-%H:%M:%S`] [ERROR] ${exec_path}/train_${each_id}.log not exist "
            failed_device="echo ${failed_device} ${each_id}"
        else
            check_step=`grep "turing train success" ${exec_path}/train_${each_id}.log | wc -l`
            if [ ${check_step} -ne 1 ];
            then
                echo "[`date +%Y%m%d-%H:%M:%S`] [ERROR] device:${each_id}'s training job failed "
                failed_device="echo ${failed_device} ${each_id}"
            fi
        fi
    done

    if [ "${failed_device}"x = x ] ;
    then
        echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] exec train success "
    else
        echo "[`date +%Y%m%d-%H:%M:%S`] [ERROR] exec train fail "
    fi

    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] see log: ${exec_path}"
}

if [ $# -lt 4 ] ;
then
    echo "==================================================================="
    echo "usage:"
    echo "Docker: sh e2e_test.sh csvfile casenum mode dockerimage rank_id rank_table_file host_device"
    echo "Host  : sh e2e_test.sh csvfile casenum mode rank_id rank_table_file host_device"
    echo "==================================================================="
    exit
fi

inputcsv=$1
dos2unix ${inputcsv} 1>/dev/null 2>&1

execnum=$2
mode=$3
if [ x$mode == xdocker ];then
    dockerimage=$4
    rank_id=$5
    rank_table_file=$6
    host_device=$7
else
    rank_id=$4
    rank_table_file=$5
    host_device=$6
    dockerimage=""
fi

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] Input param: ${inputcsv} ${execnum} ${mode} ${dockerimage} ${rank_id} ${rank_table_file} ${host_device}"

echo "/var/log/npu/dump/core.%e.%p" > /proc/sys/kernel/core_pattern

uid=`id -u HwHiAiUser`
gid=`id -g HwHiAiUser`
if [[ ${uid} -eq "1000" && ${gid} -eq "1000" ]];
then
    test_exec ${inputcsv} ${execnum} ${mode} ${rank_id} ${rank_table_file} ${host_device}
else
    echo "[`date +%Y%m%d-%H:%M:%S`] [ERROR] HwHiAiUser's UID&GID must be 1000"
    exit 1
fi

exit 0


