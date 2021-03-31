#!/bin/bash
current_exec_path=$(pwd)
echo ${current_exec_path}

# SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
# echo ${SCRIPTPATH}
source $current_exec_path/tools/common.sh

cluster_config_path="${current_exec_path}/tools/cluster_16p.json"
RANK_SIZE=$(get_rank_size ${cluster_config_path})
RANK_START=0
node_list=$(get_cluster_list ${cluster_config_path})

for node in ${node_list}
do
  echo $node
  user=$(get_node_user ${cluster_config_path} ${node})
  passwd=$(get_node_passwd ${cluster_config_path} ${node})
  echo "---------------------------------${user}@${node}--------------------------------"
  ssh_pass ${node} ${user} ${passwd} "mkdir -p ${current_exec_path}; cd ${current_exec_path}; bash ${current_exec_path}/scripts_16p/run.sh ${RANK_SIZE} ${RANK_START}"
  echo "finish"
  RANK_START=$[RANK_START+8]
  if [[ $RANK_START -ge $RANK_SIZE ]]; then
    break;
  fi
done

