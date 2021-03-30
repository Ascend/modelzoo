#!/bin/bash
SCRIPTPATH=$(pwd)

source $SCRIPTPATH/tools/common.sh
cluster_config_path="${SCRIPTPATH}/tools/cluster_16p.json"
RANK_SIZE=$(get_rank_size ${cluster_config_path})
RANK_START=0

node_list=$(get_cluster_list ${cluster_config_path})

for node in ${node_list}
do
  user=$(get_node_user ${cluster_config_path} ${node})
  passwd=$(get_node_passwd ${cluster_config_path} ${node})
  echo "---------------------------------${user}@${node}--------------------------------"
  ssh_pass ${node} ${user} ${passwd} "mkdir -p /opt/npu/cgf/mindspore"
  scp_pass ${node} ${user} ${passwd} /opt/npu/cgf/mindspore/faceReidToMe /opt/npu/cgf/mindspore/
  RANK_START=$[RANK_START+8]
  if [[ $RANK_START -ge $RANK_SIZE ]]; then
    break;
  fi
done
