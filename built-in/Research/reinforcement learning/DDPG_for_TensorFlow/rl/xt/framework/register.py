#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Register factory.
"""

import importlib
import os
from os.path import dirname
import sys
import glob
from absl import logging
# logging.set_verbosity(logging.DEBUG)
try:
    import xt
except ModuleNotFoundError as err:
    xt = None


class Register(object):
    """Register module"""

    def __init__(self, name):
        self._dict = {}
        self._name = name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be callable.")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key:{} already in registry {}.".format(key, self._name))
        self._dict[key] = value

    def register(self, param):
        """Decorator to register a function or class."""

        def decorator(key, value):
            self[key] = value
            return value

        if callable(param):
            # @reg.register
            return decorator(None, param)
        # @reg.register('alias')
        return lambda x: decorator(param, x)

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as e:
            logging.error("module {} not found: {}".format(key, e))
            raise e

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """return all the keys contained"""
        return self._dict.keys()


class Registers(object):  # pylint: disable=invalid-name, too-few-public-methods
    """All module registers within it."""

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    agent = Register("agent")
    model = Register("model")
    algorithm = Register("algorithm")
    env = Register("env")
    comm = Register("comm")


def path_to_module_format(py_path):
    """Transform a python/file/path to module format match to the importlib."""
    # return py_path.replace("/", ".").rstrip(".py")
    return os.path.splitext(py_path)[0].replace("/", ".")


def _keep_patten(file_name):
    return False if "__init__" in file_name or "_conf" in file_name else True


def _catch_subdir_modules(module_path):
    # print("pwd: ", os.getcwd())
    if xt:
        work_path = os.path.join(dirname(dirname(xt.__file__)), module_path)
    else:
        work_path = module_path
    target_file = glob.glob("{}/*/*.py".format(work_path))
    model_path_len = len(work_path)
    target_file_clip = [item[model_path_len:] for item in target_file]
    used_file = [item for item in target_file_clip if _keep_patten(item)]

    return [path_to_module_format(_item) for _item in used_file]


def _catch_peer_modules(module_path):
    if xt:
        work_path = os.path.join(dirname(dirname(xt.__file__)), module_path)
    else:
        work_path = module_path

    target_file = glob.glob("{}/*.py".format(work_path))
    model_path_len = len(work_path)
    target_file_clip = [item[model_path_len:] for item in target_file]

    used_file = [item for item in target_file_clip if _keep_patten(item)]

    return [path_to_module_format(_item) for _item in used_file]


MODEL_MODULES = _catch_subdir_modules("xt/model")
ALG_MODULES = _catch_subdir_modules("xt/algorithm")
AGENT_MODULES = _catch_subdir_modules("xt/agent")
ENV_MODULES = _catch_subdir_modules("xt/environment")

COMM_MODULES = _catch_peer_modules("xt/framework/comm")

ALL_MODULES = [
    ("xt.model", MODEL_MODULES),
    ("xt.algorithm", ALG_MODULES),
    ("xt.agent", AGENT_MODULES),
    ("xt.environment", ENV_MODULES),
    ("xt.framework.comm", COMM_MODULES),
]


def _handle_errors(errors):
    """Log out and possibly re-raise errors during import."""
    if not errors:
        return
    for name, err in errors:
        logging.warning("Module {} import failed: {}".format(name, err))
    # logging.fatal("Please check these modules.")  # could shutdown


def get_custom_modules(config=None):
    """Add custom modules to system_modules
    # import xt.agent.custom_agent/*.py, the config will be follows
    {"custom_modules": [xt.agent.custom_agent]}
    """
    custom_module_path = list()

    if config is not None and "custom_modules" in config:
        custom_modules = config["custom_modules"]
        if not isinstance(custom_modules, list):
            custom_modules = [custom_modules]
        custom_module_path.extend(
            [("", [path_to_module_format(module)]) for module in custom_modules]
        )
    return custom_module_path


def import_all_modules_for_register(custom_module_config=None):
    """Import all modules for register."""
    # add `pwd` into path.
    current_work_dir = os.getcwd()
    if current_work_dir not in sys.path:
        sys.path.append(current_work_dir)

    all_modules = ALL_MODULES
    all_modules.extend(get_custom_modules(custom_module_config))

    logging.debug("import all_modules: {}".format(all_modules))
    errors = []

    for base_dir, modules in all_modules:
        for name in modules:
            try:
                if base_dir != "":
                    full_name = base_dir + name
                else:
                    full_name = name
                importlib.import_module(full_name)
                logging.debug("{} loaded.".format(full_name))
            except ImportError as error:
                errors.append((name, error))
    _handle_errors(errors)
