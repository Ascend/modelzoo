#!/bin/bash

# Initialize environment

DLS_USER_HOME_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DLS_USER_HOME_DIR"
DLS_USER_JOB_DIR="$DLS_USER_HOME_DIR/user-job-dir"
export PYTHONPATH="$DLS_USER_JOB_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Get job/task-related information from environmental variables

# The following variables can be used:
#   - DLS_JOB_ID : Job ID
#   - DLS_TASK_INDEX : Current task index
#   - DLS_TASK_NUMBER : Total task number
#   - DLS_APP_URL : Application (code) URL
#   - DLS_DATA_URL : Data (input) URL
#   - DLS_TRAIN_URL : Train (output) URL
# E.g.
#   TENSORFLOW_JOB_NAME=`if [ $(( $DLS_TASK_INDEX % 2)) -eq 0 ]; then echo "ps"; else echo "worker"; fi`
#   TENSORFLOW_TASK_INDEX=$(( DLS_TASK_INDEX / 2))
#   "$@" --job_name=$TENSORFLOW_JOB_NAME --task_index=$TENSORFLOW_TASK_INDEX

source utils.sh
dls_fix_dns
unset_job_env_except_self "$DLS_JOB_ID"
decrypt_dls_aes_env

app_url="$DLS_APP_URL"
user_url="${app_url%/*}"
user_url_f="${user_url##*/}"

VEGA_USER_DIR="$DLS_USER_JOB_DIR/$user_url_f"
export PYTHONPATH="$VEGA_USER_DIR:$PYTHONPATH"
log_url="/tmp/dls-task-$DLS_TASK_INDEX.log"

echo "user: `id`"
echo "pwd: $PWD"
echo "app_url: $app_url"
echo "log_url: $log_url"
echo "command:" "$@"



# Launch process (task)

mkdir -p "$DLS_USER_JOB_DIR" && cd "$DLS_USER_JOB_DIR"

dls_create_log "$log_url"
tail -F "$log_url" &
TAIL_PID=$!
DLS_DOWNLOADER_LOG_FILE=/tmp/dls-downloader.log
dls_get_app "$app_url" 2>&1 | tee "$DLS_DOWNLOADER_LOG_FILE"
if [ "${PIPESTATUS[0]}" = "0" ]
then
    cd "$VEGA_USER_DIR"
    echo $(ls)
    vega_user_url="$@"
    vega_main_url_1="${vega_user_url%/*}"
    vega_all_url="$VEGA_USER_DIR$vega_main_url_1"
    cd "$vega_all_url"
    vega_start="${vega_user_url##*/}"
    echo "$PWD"
    echo "$vega_start"
    echo $(ls)
    vega_python="python3"
    echo "$vega_python $vega_start"
    export PYTHONPATH="$vega_all_url:$PYTHONPATH"
    # horovod env
    RSH_AGENT_PATH=/home/work/kube_plm_rsh_agent
    HOST_FILE_PATH=/home/work/hostfile
    chmod +x $RSH_AGENT_PATH
    KUBE_SA_CONFIG=/home/work
    if [ -d $KUBE_SA_CONFIG ]; then
        NAMESPACE=default
        TOKEN=$(cat $KUBE_SA_CONFIG/token)
    fi

    kubectl config set-cluster this --server https://kubernetes/ --certificate-authority=$KUBE_SA_CONFIG/ca.crt
    kubectl config set-credentials me --token "$TOKEN"
    kubectl config set-context me@this --cluster=this --user=me --namespace "$NAMESPACE"
    kubectl config use me@this
    kubectl config view
        
   # stdbuf -oL -eL "$vega_python $vega_all_url/$vega_start" 2>&1 | dls_logger "$log_url"
    echo "$vega_python $vega_all_url/$vega_start"
    eval "$vega_python $vega_start" 2>&1 | dls_logger "$log_url"
    RET_CODE=${PIPESTATUS[0]}

else
    (echo "App download error: "; cat "$DLS_DOWNLOADER_LOG_FILE") | dls_logger "$log_url"
    RET_CODE=127
fi

if [ ! -z "$DLS_USE_UPLOADER" ] && [ "$DLS_USE_UPLOADER" != "0" ]
then
    dls_upload_log "$log_url" "$DLS_UPLOAD_LOG_OBS_DIR" 2>&1 | tee -a "$DLS_DOWNLOADER_LOG_FILE"
_USER_JOB_DIR
    then
        (echo "Log upload error: "; cat "$DLS_DOWNLOADER_LOG_FILE") | dls_logger "$log_url" "append"
fi

sleep 3
kill $TAIL_PID
exit $RET_CODE
