SSH="ssh -o StrictHostKeyChecking=no"
SCP="scp -o StrictHostKeyChecking=no"

error_msg()
{
	local msg="$@"
	echo "[ERROR]: $msg" 1>&2
	exit 1
}

ssh_pass()
{
	local node="$1"
	local user="$2"
	local passwd="$3"
	shift 3
	local cmd="$@"
	sshpass -p "${passwd}" ${SSH} "${user}"@"${node}" ${cmd}
}

scp_pass()
{
	local node="$1"
	local user="$2"
	local passwd="$3"
	local src="$4"
	local target="$5"
	sshpass -p "${passwd}" ${SCP} -r "${src}" "${user}"@"${node}":"${target}"
}

rscp_pass()
{
	local node="$1"
	local user="$2"
	local passwd="$3"
	local src="$4"
	local target="$5"
	sshpass -p "${passwd}" ${SCP} -r "${user}"@"${node}":"${src}" "${target}"
}

get_rank_size()
{
	local cluster_config=$1
	cat ${cluster_config} | python3 -c 'import sys,json;print(json.load(sys.stdin)["rank_size"])'
}

get_train_dataset()
{
	local cluster_config=$1
	cat ${cluster_config} | python3 -c 'import sys,json;print(json.load(sys.stdin)["train_dataset"])'
}

get_cluster_list()
{
	local cluster_config=$1
	cat ${cluster_config} | python3 -c 'import sys,json;[print(node) for node in json.load(sys.stdin)["cluster"].keys()]' | sort
}


get_node_user()
{
	local cluster_config=$1
	local node=$2
	cat ${cluster_config} | python3 -c 'import sys,json;print(json.load(sys.stdin)["cluster"]['\"${node}\"']["user"])'
}

get_node_passwd()
{
	local cluster_config=$1
	local node=$2
	cat ${cluster_config} | python3 -c 'import sys,json;print(json.load(sys.stdin)["cluster"]['\"${node}\"']["passwd"])'
}

rsync_sshpass()
{
	local node=$1
	local user="$2"
	local passwd="$3"
	scp_pass "${node}" "${user}" "${passwd}" /usr/local/bin/sshpass /usr/local/bin/sshpass
}

get_device_eth_ip()
{
	local node=$1
	local user=$2
	local passwd=$3
	local eth=$4
	local device_user="HwHiAiUser"
	local device_passwd="Huawei2012#"

	if [ "${eth}" = "eth0" -o "${eth}" = "eth1" -o "${eth}" = "eth2" -o "${eth}" = "eth3" ]
	then
		local device_ip=[network_id].[host_id]
		local ifconfig_cmd="${SSH} ${device_user}@${device_ip} /usr/sbin/ifconfig ${eth}"
		local out=$(ssh_pass "${node}" "${user}" "${passwd}" sshpass -p "${device_passwd}" ${ifconfig_cmd})
		echo $out | grep 'inet addr' | awk -F' |:' '{print $14}'
	fi
	if [ "${eth}" = "eth4" -o "${eth}" = "eth5" -o "${eth}" = "eth6" -o "${eth}" = "eth7" ]
	then
		local device_ip=[network_id].[host_id]
		local ifconfig_cmd="${SSH} ${device_user}@${device_ip} /usr/sbin/ifconfig ${eth}"
		local out=$(ssh_pass "${node}" "${user}" "${passwd}" sshpass -p "${device_passwd}" ${ifconfig_cmd})
		echo $out | grep 'inet addr' | awk -F' |:' '{print $14}'
	fi
}

