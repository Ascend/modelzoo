# -*- coding:utf-8 -*-
"""This is an exmple to use BO HPO."""

import vega


if __name__ == '__main__':
    vega.init_local_cluster_args('./init_cluster_config.yml')
    vega.run('./bo.yml')
