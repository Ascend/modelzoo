# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Main entrance for xingtian library

Usage:
    python main.py -f examples/default_cases/cartpole_ppo.yaml -t train

"""

import argparse
import pprint
import yaml
from absl import logging
from xt.train import main as xt_train
from xt.train import makeup_multi_case
from xt.evaluate import main as xt_eval
from xt.benchmarking import main as xt_benchmarking
from xt.benchmark.tools.get_config import parse_xt_multi_case_paras
from xt.util.logger import VERBOSITY_MAP

# from xt.framework.register import import_all_modules_for_register
# import_all_modules_for_register()

from xt.framework.remoter import distribute_xt_if_need, start_remote_node

logging.set_verbosity(logging.INFO)


def main():
    """
    :return: config file for training or testing
    """
    parser = argparse.ArgumentParser(description="Usage xingtian with sourcecode.")

    parser.add_argument(
        "-f", "--config_file", required=True, help="""config file with yaml""",
    )
    # fixme: split local and hw_cloud,
    #  source path could read from yaml startswith s3
    parser.add_argument(
        "-s3", "--save_to_s3", default=None, help="save model/records into s3 bucket."
    )
    parser.add_argument(
        "-t",
        "--task",
        # required=True,
        default="train",
        choices=["train", "evaluate", "train_with_evaluate", "benchmark"],
        help="task choice to run xingtian.",
    )
    parser.add_argument(
        "-v", "--verbosity", default="info", help="logging.set_verbosity"
    )

    args, _ = parser.parse_known_args()
    if _:
        logging.warning("get unknown args: {}".format(_))

    if args.verbosity in VERBOSITY_MAP.keys():
        logging.set_verbosity(VERBOSITY_MAP[args.verbosity])
    else:
        logging.warning("un-known logging level-{}".format(args.verbosity))

    _exp_params = pprint.pformat(args, indent=0, width=1,)
    logging.info(
        "\n{}\n XT start work...\n{}\n{}".format("*" * 50, _exp_params, "*" * 50)
    )

    with open(args.config_file, "r") as conf_file:
        config_info = yaml.safe_load(conf_file)

    config_info = start_remote_node(config_info)
    print(config_info)
    distribute_xt_if_need(config=config_info, remote_env=config_info.get("remote_env"))

    if args.task in ("train", "train_with_evaluate"):
        ret_para = parse_xt_multi_case_paras(args.config_file)
        if len(ret_para) > 1:
            makeup_multi_case(args.config_file, args.save_to_s3)
        else:
            xt_train(config_info, args.save_to_s3, args.verbosity)

    elif args.task == "evaluate":
        xt_eval(config_info, args.save_to_s3)

    elif args.task == "benchmark":
        # fixme: with benchmark usage in code.
        # xt_benchmark(args.config_file)
        xt_benchmarking()
    else:
        logging.fatal("Get invalid task: {}".format(args.task))


if __name__ == "__main__":
    main()
