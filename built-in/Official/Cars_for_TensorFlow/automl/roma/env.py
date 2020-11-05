# -*- coding:utf-8 -*-
"""This is the founction to initialize env."""
import os
from vega.core.common import Config
from .file_ops import replace_file_ops
from .register_cluster_args import register_cluster_args


def init_env(regin="hn1_y"):
    """Init env."""
    os.environ['VEGA_INIT_ENV'] = "from roma.env import init_env;init_env();"
    _set_default_config(regin)
    replace_file_ops()
    register_cluster_args()


def _set_default_config(regin):
    # current_path = os.path.abspath(os.path.dirname(__file__))
    # full_path = os.path.join(current_path, "config", "{}.yml".format(regin))
    # cfg = Config(full_path)
    # DefaultConfig().data = cfg
    pass
