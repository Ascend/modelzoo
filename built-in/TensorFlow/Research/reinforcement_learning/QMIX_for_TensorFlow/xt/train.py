# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#!/usr/bin/env python
"""
xingtian train entrance.
"""
import os
import signal
import sys
import time
from subprocess import Popen
import pprint

from absl import logging
import yaml
import zmq

from xt.evaluate import setup_evaluate_adapter
from xt.framework.broker_launcher import launch_broker
from xt.framework.learner import setup_learner, patch_alg_within_config
from xt.framework.explorer import setup_explorer
from xt.util.common import get_config_file
from xt.benchmark.tools.get_config import parse_xt_multi_case_paras

TRAIN_PROCESS_LIST = list()


def _makeup_learner(config_info, data_url, verbosity):
    """make up a learner instance, and build the relation with broker"""

    config_info = patch_alg_within_config(config_info.copy())

    _exp_params = pprint.pformat(config_info, indent=0, width=1,)
    logging.info("init learner with:\n{}\n".format(_exp_params))

    broker_master = launch_broker(config_info, start_port=21000)  # 20000

    eval_adapter = setup_evaluate_adapter(config_info, broker_master, verbosity)

    # fixme: split the relation between learner and tester
    learner = setup_learner(config_info, eval_adapter, data_url)

    learner.send_predict = broker_master.register("predict")
    learner.send_train = broker_master.register("train")
    learner.stats_deliver = broker_master.register("stats_msg")
    learner.send_broker = broker_master.recv_local_q
    learner.start()

    broker_master.main_task = learner

    env_num = config_info.get("env_num")
    for i in range(env_num):
        setup_explorer(broker_master, config_info, i)
    return broker_master


def start_train(config_info, data_url=None, try_times=5, verbosity="info"):
    """ start train"""
    for _ in range(try_times):
        try:
            return _makeup_learner(config_info, data_url, verbosity)

        except zmq.error.ZMQError as err:
            logging.error("catch: {}, \n try with times-{}".format(err, _))
            continue
        except BaseException as ex:
            logging.exception(ex)
            os.system("pkill -9 fab")
            sys.exit(3)


def handle_multi_case(sig, frame):
    """ Catch <ctrl+c> signal for clean stop """

    global TRAIN_PROCESS_LIST
    for p in TRAIN_PROCESS_LIST:
        p.send_signal(signal.SIGINT)

    time.sleep(1)
    os._exit(0)


def main(config_info, s3_path=None, verbosity="info"):
    """do train task with single case """
    broker_master = start_train(config_info, data_url=s3_path, verbosity=verbosity)
    loop_is_end = False
    try:
        broker_master.main_loop()
        loop_is_end = True
    except (KeyboardInterrupt, EOFError) as err:
        logging.warning("Get a KeyboardInterrupt, Stop early.")

    # handle close signal, with cleaning works.
    broker_master.main_task.train_worker.logger.save_to_json()
    broker_master.stop()

    # fixme: make close harmonious between broker master & slave
    time.sleep(2)
    if loop_is_end:
        logging.info("Finished train job normally.")

    os._exit(0)


# train with multi case
def write_conf_file(config_folder, config):
    """ write config to file """
    with open(config_folder, "w") as f:
        yaml.dump(config, f)


def makeup_multi_case(config_file, s3_path):
    """ run multi case """
    signal.signal(signal.SIGINT, handle_multi_case)
    # fixme: setup with archive path
    if os.path.isdir("log") is False:
        os.makedirs("log")
    if os.path.isdir("tmp_config") is False:
        os.makedirs("tmp_config")

    ret_para = parse_xt_multi_case_paras(config_file)
    config_file_base_name = os.path.split(config_file)[-1]

    for i, para in enumerate(ret_para):
        if i > 9:
            logging.fatal("only support 10 parallel case")
            break

        tmp_config_file = "{}_{}".format(config_file_base_name, i)
        config_file = os.path.join("tmp_config", tmp_config_file)
        write_conf_file(config_file, para)

        abs_config_file = os.path.abspath(config_file)

        log_file = os.path.join("log", "log_{}.log".format(tmp_config_file))

        TRAIN_PROCESS_LIST.append(
            launch_train_with_shell(
                abs_config_file, s3_path=s3_path, stdout2file=log_file
            )
        )

    while True:
        time.sleep(100)


def launch_train_with_shell(abs_config_file, s3_path=None, stdout2file="./xt.log"):
    """ run train process """
    cmd = "import xt; from xt.train import main; main('{}', {})".format(
        abs_config_file, s3_path
    )
    logging.info("start launching train with cmd: \n{}".format(cmd))
    file_out = open(stdout2file, "w")

    process_instance = Popen(
        ["python3", "-c", cmd],
        stdout=file_out,
        # stderr=subprocess.PIPE,
        # shell=True,
    )
    time.sleep(1)
    return process_instance


if __name__ == "__main__":
    main(get_config_file())
