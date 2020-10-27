# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The main part of the cluster framework.

The DaskEnv Class which used in Master to init and set basic dask-distributed
environment.
"""
import os
import json
import psutil
import subprocess
import traceback
import logging
import fcntl
from distributed.diagnostics.plugin import WorkerPlugin


class WorkerEnv(WorkerPlugin):
    """WorkerEnv for add plugin in each worker in dask cluster.

    :param int workers_each_node: worker count on each slave node.
    :param int device_quota: device num for each worker to use.
    :param str master_host_name: the dask cluster master host name.
    :param str master_pid: the process id of the master process.

    """

    def __init__(self, workers_each_node, device_quota, master_host_name, master_pid, temp_path):
        """Init the WorkerEnv."""
        self.workers_each_node = workers_each_node
        self.device_quota = device_quota
        self.master_host_name = master_host_name
        self.local_host_name = None
        self.master_pid = master_pid
        self.device_list = []
        self.device_category = os.environ['DEVICE_CATEGORY']
        self.__worker_number_file__ = os.path.join(temp_path, '.vega_worker_env_gpu')
        self.__worker_null_file__ = os.path.join(temp_path, '.vega_null')
        self.__worker_device_folder__ = os.path.join(temp_path, '.vega_device')
        print('worker number file: {}'.format(self.__worker_number_file__))
        return

    def _set_device_env(self):
        """Use a local file to save a label to mark gpu id used for different workers on a same slave node."""
        if not os.path.isfile(self.__worker_number_file__):
            os.makedirs(os.path.dirname(self.__worker_number_file__), exist_ok=True)
            fp = open(self.__worker_number_file__, 'w')
            fcntl.flock(fp, fcntl.LOCK_EX)
            fp.write('{}'.format(0))
            fcntl.flock(fp, fcntl.LOCK_UN)
            fp.close()
        return

    def _get_device_list(self):
        """Get the cuda devices id list that are visible to current workers.

        :return: the current worker visible gpu id list.
        :rtype: list

        """
        current_count = 0
        with open(self.__worker_number_file__, 'r+') as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            f_str = fp.readline()
            try:
                current_count = int(f_str.strip()) % self.workers_each_node
            except Exception:
                pass
            with open(self.__worker_number_file__, 'w') as fn:
                fn.write('{}'.format(current_count + 1))
            fcntl.flock(fp, fcntl.LOCK_UN)
        device_list = []
        for i in range(current_count * self.device_quota, (current_count + 1) * self.device_quota):
            device_list.append('{}'.format(i))
        return device_list

    def _set_visible_devices(self):
        """Set visible devices to each worker env."""
        if self.device_category == 'GPU':
            cuda_device_list_str = ",".join(self.device_list)
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device_list_str
            logging.info("CUDA_VISIBLE_DEVICES:" + cuda_device_list_str)
        elif self.device_category == 'NPU':
            origin_rank_file = os.environ.get('ORIGIN_RANK_TABLE_FILE')
            with open(origin_rank_file, 'r') as f:
                rank_table_json = json.loads(f.read())

            ## server list method
            rank_table_json['server_count'] = 1
            group_info = rank_table_json['server_list']
            devices_info = []
            keep_idx = int(os.environ.get('BATCH_TASK_INDEX'))
            instance_info = group_info[keep_idx]
            for device_id in self.device_list:
                device_id = int(device_id)
                devices_info.append(instance_info['device'][device_id])
            if len(devices_info) == 0:
                raise Exception('No matching devices info.')
            rank_table_json['server_list'] = [instance_info]
            rank_table_json['server_list'][0]['device'] = devices_info

            ## instance list method
            # group_info = rank_table_json['group_list'][0]
            # group_info['device_count'] = str(len(self.device_list))
            # group_info['instance_count'] = '1'
            # devices_info = []
            # keep_idx = 0
            # for idx, instance in enumerate(group_info['instance_list']):
            #     if instance['pod_name'] == local_pod_name:
            #         for device_id in self.device_list:
            #             device_id = int(device_id)
            #             devices_info.append(instance['devices'][device_id])
            #         keep_idx = idx
            #         break
            # if len(devices_info) == 0:
            #     raise Exception('No matching devices info.')
            # group_info['instance_list'] = [group_info['instance_list'][keep_idx]]
            # group_info['instance_list'][0]['devices'] = devices_info

            new_rank_table_file = os.path.join(self.__worker_device_folder__,
                                               'rank_table_{}.json'.format(self.device_list[0]))
            if not os.path.exists(self.__worker_device_folder__):
                os.makedirs(self.__worker_device_folder__, exist_ok=True)
            with open(new_rank_table_file, 'w') as f:
                f.write(json.dumps(rank_table_json))
            print('worker {} rank table json: {}'.format(self.device_list[0], rank_table_json))
            os.environ['RANK_TABLE_FILE'] = new_rank_table_file
            os.environ['RANK_SIZE'] = str(len(self.device_list))
            os.environ['DEVICE_ID'] = self.device_list[0]
            os.environ['RANK_ID'] = rank_table_json['server_list'][0]['device'][0]['rank_id']
            logging.info("RANK_TABLE_FILE:" + new_rank_table_file)
        else:
            raise Exception('device category must be GPU or NPU.')

    def setup(self, worker):
        """Call back function for worker setup.

        here to get worker's local host name, and set worker visible gpu ids in
        CUDA_VISIBLE_DEVICES.

        """
        if "BATCH_CURRENT_HOST" in os.environ:
            self.local_host_name = os.environ["BATCH_CURRENT_HOST"]
        elif "BATCH_CUSTOM0_HOSTS" in os.environ:
            self.local_host_name = os.environ["BATCH_CUSTOM0_HOSTS"]
        self._set_device_env()
        self.device_list = self._get_device_list()
        self._set_visible_devices()
        return

    def teardown(self, worker):
        """Call back function for worker teardown."""
        return

    def transition(self, key, start, finish, *args, **kwargs):
        """Call back function for worker status transition.

        here to clean the gpu memory whe worker status turn to `ready`,
        use `fuser -v` list all pid that use cuda, and filter the master's
        processes, and kill all other processes.

        :param str key: Description of parameter `key`.
        :param str start: Start state of the transition.
            One of waiting, ready, executing, long-running, memory, error.
        :param str finish: Final state of the transition.
        :param type * args: Description of parameter `*args`.
        :param type ** kwargs: Description of parameter `**kwargs`.

        """
        logging.info(" Plugin transition ")
        #
        if finish == 'ready' and len(self.device_list) > 0:
            try:
                current_pid = os.getpid()
                protect_pid_set = set()
                protect_pid_set.add(int(current_pid))
                # if self.master_host_name is not None and self.master_host_name == self.local_host_name:
                protect_pid_set.add(int(self.master_pid))
                try:
                    parent = psutil.Process(self.master_pid)
                    for p in parent.children(recursive=False):
                        protect_pid_set.add(int(p.pid))
                except Exception:
                    logging.debug("In slave node, master pid is not existed, process does not need to protect.")
                if self.device_category == 'GPU':
                    cuda_pid_set = set()
                    for id in self.device_list:
                        device = "/dev/nvidia{}".format(id)
                        fh = open(self.__worker_null_file__, "w")
                        p = subprocess.Popen(["fuser", "-v", device], stdout=subprocess.PIPE, stderr=fh)
                        p.wait()
                        fh.close()
                        sub_pids = p.stdout.read().split()
                        for spid in sub_pids[1:]:
                            cuda_pid_set.add(int(spid))
                    for spid in protect_pid_set:
                        if spid in cuda_pid_set:
                            cuda_pid_set.remove(spid)
                    # for spid in cuda_pid_set:
                    #     subprocess.call(["kill", "-9", "{}".format(spid)])
                    if cuda_pid_set:
                        logging.info("Non-Vega process is using GPU, pids={}".format(cuda_pid_set))
            except Exception:
                logging.error("Worker Plugin Error.")
                logging.error(traceback.format_exc())
        logging.info("cleaned the cuda memory...")
        return
