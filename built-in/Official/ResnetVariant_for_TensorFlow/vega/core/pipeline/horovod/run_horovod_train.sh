#!/usr/bin/env bash

# thx for KungFu Team and modify by ModelArts Team
# ps: do not share for other persons, it has the security problem

# This script runs the Horovod training job on the modelarts platform.
#set -x

#cd /home/work/user-job-dir/automl

# Modify script path to point to training script (please set aboslute path)
# for example
SCRIPT_PATH=/root/miniconda3/lib/python3.6/site-packages/vega/core/pipeline/horovod/horovod_train.py
# Modify RSH agent path to point to kube-plm-rsh-agent file (please set aboslute path)
RSH_AGENT_PATH=/root/miniconda3/lib/python3.6/site-packages/vega/core/pipeline/horovod/kube_plm_rsh_agent
# Modify hostfile indicating where pod characteristics are located (please set aboslute path)
HOST_FILE_PATH=/home/work/hostfile

chmod +x $RSH_AGENT_PATH

gen_hostfile() {
    pods=$(kubectl get pods -o name | grep job | grep ${BATCH_JOB_ID} | awk -F '/' '{print $2}')
    for pod in $pods; do
        echo "$pod slots=8" # modify slots to the real slogs num (generally, it equals to the number of gpus)
    done
}

gen_hostfile >$HOST_FILE_PATH

MPI_HOME=$HOME/local/openmpi

run_experiment() {
    local np=$1
    shift

    mpirun --allow-run-as-root -np ${np} \
        -mca plm_rsh_agent $RSH_AGENT_PATH \
        --hostfile $HOST_FILE_PATH \
        --bind-to socket \
        -x LD_LIBRARY_PATH \
        -x PYTHONPATH  -x VEGA_INIT_ENV \
        -x S3_REGION -x AWS_SECRET_ACCESS_KEY -x S3_ENDPOINT -x S3_VERIFY_SSL -x AWS_ACCESS_KEY_ID -x S3_USE_HTTPS \
        -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=ib0,bond0,eth0 -x NCCL_SOCKET_FAMILY=AF_INET -x NCCL_IB_DISABLE=0 \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -mca pml ob1 -mca btl ^openib \
        -mca plm_rsh_no_tree_spawn true \
        -mca btl_tcp_if_include 192.168.0.0/16 \
        $@
}

export TF_CPP_MIN_LOG_LEVEL=1

if [ "$DLS_TASK_NUMBER" = "1" ] || [ "$DLS_TASK_INDEX" = "0" ]; then
    # modify this value to the real np num (generally, it equals to the number of node number * gpu number)
    nps=$1
    run_experiment $nps python3 $SCRIPT_PATH --cf_file $2
else
    sleep 5d
fi
