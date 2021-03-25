# Copyright 2021 Huawei Technologies Co., Ltd
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
# =============================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-18 下午4:11
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : log_utils.py
# @IDE: PyCharm Community Edition
"""
Set the log config
"""
import logging
from logging import handlers
import os
import os.path as ops


def init_logger(level=logging.DEBUG, when="D", backup=7,
                _format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s",
                datefmt="%m-%d %H:%M:%S"):
    """
    init_log - initialize log module
    :param level: msg above the level will be displayed DEBUG < INFO < WARNING < ERROR < CRITICAL
                  the default value is logging.INFO
    :param when:  how to split the log file by time interval
                  'S' : Seconds
                  'M' : Minutes
                  'H' : Hours
                  'D' : Days
                  'W' : Week day
                  default value: 'D'
    :param backup: how many backup file to keep default value: 7
    :param _format: format of the log default format:
                   %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
                   INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
    :param datefmt:
    :return:
    """
    formatter = logging.Formatter(_format, datefmt)
    logger = logging.getLogger()
    logger.setLevel(level)

    log_path = ops.join(os.getcwd(), 'logs/shadownet.log')
    _dir = os.path.dirname(log_path)
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

    handler = handlers.TimedRotatingFileHandler(log_path, when=when, backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = handlers.TimedRotatingFileHandler(log_path + ".log.wf", when=when, backupCount=backup)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
