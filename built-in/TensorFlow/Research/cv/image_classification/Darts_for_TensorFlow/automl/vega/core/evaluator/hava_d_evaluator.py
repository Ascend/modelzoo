# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""D-Loop Evaluator.

D-Loop Evaluator Class used to download the model and config file
from worker path of s3, write and upload the json file to s3 and
wait for result from Monitor and Hava service, modify the result
message and upload to remote worker path.
"""
import os
import json
import time
import logging
import pickle
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.trainer.distributed_worker import DistributedWorker
from vega.core.trainer.utils import WorkerTypes
from vega.core.common.file_ops import FileOps


@ClassFactory.register(ClassType.HAVA_D_EVALUATOR)
class HavaDEvaluator(DistributedWorker):
    """This is a class of D-Loop evaluator used to send a message of models.

    which are needed to convert, and receive a message from monitor.

    :param args: arguments from user and default config file
    :type args: dict or Config, default to None
    :param worker_info: validate dataset
    :type worker_info: torch dataset, default to None
    """

    def __init__(self, worker_info=None):
        """Init HavaDEvaluator."""
        super(HavaDEvaluator, self).__init__(self.cfg)
        # for init ids
        self.worker_type = WorkerTypes.HAVA_D_EVALUATOR
        self.worker_info = worker_info
        if worker_info is not None:
            self.step_name = self.worker_info["step_name"]
            self.worker_id = self.worker_info["worker_id"]
        self.final_model_dir = '.finsished_model'
        self.final_weight_dir = '.finsished_weight'

    def download_model_convertor(self):
        """Download the pytorch trained model from s3 path of worker.

        :return: 'True' if download successfully, 'False' otherwise
        :rtype: bool
        """
        step_name = self.step_name
        worker_id = self.worker_id
        r_base = self.backup_base_path
        r_path = self.cfg.task.backup_worker_subpath
        r_path = r_path.replace("[step_name]", str(step_name)).replace("[worker_id]", str(worker_id))
        self.evaluate_remote_path = FileOps.join_path(r_base, r_path)
        logging.info('evaluate remote path: {}'.format(self.evaluate_remote_path))
        l_base = self.local_base_path
        l_path = self.cfg.task.local_worker_subpath
        l_path = l_path.replace("[step_name]", str(step_name)).replace("[worker_id]", str(worker_id))
        self.evaluate_local_path = FileOps.join_path(l_base, l_path)
        FileOps.make_dir(self.evaluate_local_path)
        if not os.path.exists(self.evaluate_remote_path):
            return False
        return self._download_model_before_convert(
            FileOps.join_path(self.evaluate_remote_path, self.final_model_dir),
            FileOps.join_path(self.evaluate_remote_path, self.final_weight_dir),
            FileOps.join_path(self.evaluate_local_path, self.final_model_dir))

    def _get_torch_name(self, remote_model_path, remote_weight_path):
        """Get the model name and weight name of torch model in s3 path of worker.

        :param remote_path: remote s3 path
        :type remote_path: str
        :return: model name and config name
        :rtype: tuple
        """
        model_name = None
        weight_name = None
        f_model_list = os.listdir(remote_model_path)
        f_weight_list = os.listdir(remote_weight_path)
        for f in f_model_list:
            if f.endswith('.py'):
                model_name = f
            if model_name:
                break
        for f in f_weight_list:
            if f.endswith('.pth'):
                weight_name = f
            if weight_name:
                break
        if model_name is None:
            logging.warn("{} has no model or config".format(remote_model_path))
        if weight_name is None:
            logging.warn("{} has no model or config".format(remote_weight_path))
        return model_name, weight_name

    def write_upload_configfile(self):
        """Write a json config file.

        include framework type, model and config path,
        and validate directory in s3.
        """
        try:
            self.base_evaluate_json = "{}{}.json".format(
                os.environ['DLS_JOB_ID'],
                time.strftime("_%Y%m%d_%H%M%S"))
            evaluate_json = FileOps.join_path(self.evaluate_local_path, self.base_evaluate_json)
            json_msg = {}
            json_msg['framework_type'] = self.cfg.task.framework_type
            json_msg['model_file'] = FileOps.join_path(self.evaluate_remote_path,
                                                       self.final_model_dir,
                                                       self.model_name)
            json_msg['weight_file'] = FileOps.join_path(self.evaluate_remote_path,
                                                        self.final_weight_dir,
                                                        self.weight_name)
            json_msg['validate_dir'] = self.cfg.task.remote_validate_path
            with open(evaluate_json, 'w') as f:
                json.dump(json_msg, f)
            FileOps.copy_file(FileOps.join_path(self.evaluate_local_path, self.base_evaluate_json),
                              FileOps.join_path(self.cfg.task.remote_pending_path, self.base_evaluate_json))
        except Exception as e:
            logging.error("write and upload dloop config error:{}".format(str(e)))

    def waiting_result(self):
        """Wait for the D-Loop inference result.

        if waiting time is larger than timeout, stop waiting and exit.

        :return: 'True' if D-Loop finished file exists, 'False' otherwise
        :rtype: bool
        """
        remote_finished_file = FileOps.join_path(self.cfg.task.remote_finished_path, self.base_evaluate_json)
        local_finished_file = FileOps.join_path(self.evaluate_local_path, self.base_evaluate_json)
        start_wait_time = time.time()
        logging.info('dloop waiting result')
        while True:
            if os.path.exists(remote_finished_file):
                logging.info('dloop finished file exists')
                FileOps.copy_file(remote_finished_file, local_finished_file)
                return True
            if time.time() - start_wait_time > (self.cfg.worker.evaluate_timeout * 60 * 60):
                return False
            time.sleep(1)

    def update_result(self):
        """Delete some useless message in finished json file, and upload to worker path of s3."""
        try:
            local_finished_file = FileOps.join_path(self.evaluate_local_path, self.base_evaluate_json)
            with open(local_finished_file, 'r') as f_load:
                json_msg = json.load(f_load)
            json_msg.pop('model_file')
            json_msg.pop('weight_file')
            json_msg.pop('validate_dir')
            new_base_file = 'dloop_result.json'
            new_finished_file = FileOps.join_path(self.evaluate_local_path, new_base_file)
            #
            new_pkl_file = FileOps.join_path(self.evaluate_local_path, 'dloop_result.pkl')
            with open(new_pkl_file, 'wb') as f_dump:
                pickle.dump(json_msg, f_dump)
            #
            with open(new_finished_file, 'w') as f_dump:
                json.dump(json_msg, f_dump)
            logging.info('modify json file')
            FileOps.copy_file(new_finished_file, FileOps.join_path(self.evaluate_remote_path,
                                                                   new_base_file))
            FileOps.copy_file(new_pkl_file, FileOps.join_path(self.evaluate_remote_path,
                                                              'dloop_result.pkl'))
        except Exception as e:
            logging.error('Dloop Update Result Error:{}'.format(str(e)))

    def _download_model_before_convert(self, remote_model_path, remote_weight_path, local_path):
        """Download pytorch model and config file from s3 to local path.

        :param remote_path: remote s3 path
        :type remote_path: str
        :param local_path: local path
        :type local_path: str
        :return: 'True' if downloading done, 'False' otherwise
        :rtype: bool
        """
        logging.info('getting torch model config name')
        self.model_name, self.weight_name = self._get_torch_name(remote_model_path, remote_weight_path)
        if self.model_name is None or self.weight_name is None:
            logging.error('dloop torch model config name error')
            return False
        logging.info('downloading model which need to be converted')
        try:
            FileOps.copy_file(FileOps.join_path(remote_model_path, self.model_name),
                              FileOps.join_path(local_path, self.model_name))
            FileOps.copy_file(FileOps.join_path(remote_weight_path, self.weight_name),
                              FileOps.join_path(local_path, self.weight_name))
        except Exception as e:
            logging.error('dloop downloading model error:{}'.format(str(e)))
            return False
        logging.info('downloading model done')
        return True

    def __call__(self, *args, **kwargs):
        """Call function of D-Loop evaluator.

        :param args: positional argument
        :type args: tuple
        :param kwargs: keyword arguments
        :type kwargs: dict
        """
        has_model = self.download_model_convertor()
        if not has_model:
            return
        self.write_upload_configfile()
        finished = self.waiting_result()
        if finished:
            self.update_result()
            logging.info('dloop evaluating finished')
        else:
            logging.warn('dloop evaluating timeout!')
        return
