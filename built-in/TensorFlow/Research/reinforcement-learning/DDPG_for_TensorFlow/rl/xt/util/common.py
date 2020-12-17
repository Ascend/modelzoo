# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
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
"""
common is a set of functions to find module
"""
from __future__ import division, print_function

import importlib
import os
import argparse
import sys
import ast
import socket
import yaml


def import_config(global_para, config):
    """
    :param global_para
    :param config
    :return: None
    """
    if not config:
        return
    for key in config.keys():
        if key in global_para:
            global_para[key] = config[key]


def node_para(args):
    """
    :param args: args
    :return: node config or test node config list
    """
    node_list = []
    i = 0
    if args.find("//") >= 0:
        for node in args.split("//"):
            node_list.append([])
            ip, name, passwd = node.split(",")
            node_list[i].append(ip)
            node_list[i].append(name)
            node_list[i].append(passwd)
            i += 1
    else:
        node_list.append([])
        ip, name, passwd = args.split(",")
        node_list[i].append(ip)
        node_list[i].append(name)
        node_list[i].append(passwd)
    return node_list


def analysis_para(args):
    """
    :param args:
    :return: Dictionary of args
    """
    dict_args = {}
    for kv in args.split(","):
        key, value = kv.split("=")
        if key == "action_dim":
            value = int(value)
            dict_args[key] = value
        elif key == "state_dim":
            value = ast.literal_eval(value)
            dict_args[key] = value
        elif key == "vision":
            if value == "True":
                value = True
            else:
                value = False
            dict_args[key] = value
        else:
            dict_args[key] = value
    return dict_args


def get_config_file():
    """
    :return: config file for training or testing
    """
    parser = argparse.ArgumentParser(
        description="parse key pairs into a dictionary for xt training or testing",
        usage="python train.py --config_file YAML_FILE OR "
        "python train.py --alg_para KEY=VALUE "
        "--env_para KEY=VALUE --env_info KEY=VALUE --agent_para KEY=VALUE "
        "--actor KEY=VALUE",
    )

    parser.add_argument("-f", "--config_file")
    parser.add_argument(
        "-s3", "--save_to_s3", default=None, help="save model into s3 bucket."
    )

    parser.add_argument("--alg_para", type=analysis_para)
    parser.add_argument("--alg_config", type=analysis_para)

    parser.add_argument("--env_para", type=analysis_para)
    parser.add_argument("--env_info", type=analysis_para)

    parser.add_argument("--agent_para", type=analysis_para)
    parser.add_argument("--agent_config", type=analysis_para)

    parser.add_argument("--actor", type=analysis_para)
    parser.add_argument("--critic", type=analysis_para)

    parser.add_argument("--model_name", default="model_name")
    parser.add_argument("--env_num", type=int, default=1)

    parser.add_argument(
        "--node_config", type=node_para, default=[["127.0.0.1", "username", "passwd"],]
    )
    parser.add_argument("--test_node_config", type=node_para)

    # parser.add_argument(
    #     "--model_path", default="../xt_train_data/test_model/" + str(os.getpid())
    # )
    parser.add_argument(
        "--test_model_path", default="../xt_train_data/train_model/" + str(os.getpid())
    )
    parser.add_argument(
        "--result_path",
        default="../xt_train_data/test_res/" + str(os.getpid()) + ".csv",
    )

    args = parser.parse_args(sys.argv[1:])
    if len(sys.argv) < 2:
        print(parser.print_help())
        exit(1)
    if args.config_file is not None:
        return args.config_file, args.save_to_s3
    args_dict = vars(args)

    model_para = {}
    model_para["actor"] = args_dict["actor"]
    model_para["critic"] = args_dict["critic"]
    args_dict["model_para"] = model_para

    args_dict.pop("actor")
    args_dict.pop("critic")

    args_dict["env_para"]["env_info"] = args_dict["env_info"]
    args_dict.pop("env_info")

    if args_dict["agent_config"] is not None:
        args_dict["agent_para"]["agent_config"] = args_dict["agent_config"]
    args_dict.pop("agent_config")

    if args_dict["alg_config"] is not None:
        args_dict["alg_para"]["alg_config"] = args_dict["alg_config"]
    args_dict.pop("alg_config")

    if args_dict["test_node_config"] is None:
        args_dict.pop("test_node_config")

    yaml_file = "./xt_{}_{}.yaml".format(
        args_dict["alg_para"]["alg_name"], args_dict["env_para"]["env_info"]["name"]
    )
    with open(yaml_file, "w") as f:
        f.write(yaml.dump(args_dict))

    return yaml_file, args.save_to_s3


def check_port(ip, port):
    """ check if port  is in use """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        print("port is used", int(port))
        return True
    except BaseException:
        return False

def bytes_to_str(data):
    """bytes to string, used after data transform by internet."""
    if isinstance(data, bytes):
        return data if sys.version_info.major == 2 else data.decode("ascii")

    if isinstance(data, dict):
        return dict(map(bytes_to_str, data.items()))

    if isinstance(data, tuple):
        return map(bytes_to_str, data)

    return data
