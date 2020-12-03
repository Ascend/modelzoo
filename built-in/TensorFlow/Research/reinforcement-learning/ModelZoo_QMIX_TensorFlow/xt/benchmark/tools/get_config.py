"""
Utils for parse the xt config files
"""
from copy import deepcopy
from itertools import product

import numpy as np
import yaml

from xt.benchmark.configs.default_xt import XtBenchmarkConf as bm_conf


def get_xt_benchmark_config(yaml_obj, default_bm_id=bm_conf.default_id):
    """
    Get xt benchmark information from config files
    Args:
        yaml_obj:
        default_bm_id:

    Returns: benchmark config dict of xt

    """
    benchmark_id = yaml_obj.get("benchmark", dict()).get("id", default_bm_id)

    # alg
    alg_name = yaml_obj.get("alg_para", dict()).get("alg_name")
    if not alg_name:
        raise KeyError("config: {} invalid, can't get 'alg_name'! ".format(yaml_obj))

    # env
    _env_para = yaml_obj.get("env_para", dict())
    if not _env_para:
        raise KeyError("config : {} invalid, can't get 'env_para'!".format(yaml_obj))

    env_name = _env_para.get("env_name")
    if not env_name:
        raise KeyError("config: {} invalid, can't get 'env_name'! ".format(yaml_obj))

    env_info_name = _env_para.get("env_info", dict()).get("name")
    if not env_info_name:
        raise KeyError("invalid config file, without 'env_info_name' para!")

    return benchmark_id, alg_name, env_name, env_info_name


def _get_product_value(dict_val):
    """
    for multi-case config file parse
    Args:
        dict_val:

    Returns:

    """

    def _combine_dict(dict_list):
        _dict = dict()
        [_dict.update(i) for i in dict_list]
        return _dict

    dict_list = list()
    # hypothesis: is order in dict.keys()
    for sub_key, sub_val in dict_val.items():
        for _key, para_val in sub_val.items():
            dict_list.append([{sub_key: {_key: para}} for para in para_val])
    return [_combine_dict(list(_i)) for _i in product(*dict_list)]


def finditem(obj, key):
    """ find key in dict """
    if not isinstance(obj, dict):
        return None
    elif key in obj:
        return obj[key]

    for k, v in obj.items():
        ret_obj = finditem(v, key)
        if ret_obj is not None:
            return ret_obj


def _get_combination_info(yaml_obj, key_fields):
    """
    for multi-case config file parse
    Args:
        yaml_obj:
        key_fields:

    Returns:

    """
    parse_candidate = {_k: dict() for _k in key_fields}
    combination_len_info = list()

    for key in key_fields:
        sub_val = finditem(yaml_obj, key)
        if sub_val is None:
            continue
        # print(sub_val)
        for para_key, para_val in sub_val.items():
            if isinstance(para_val, list):
                parse_candidate[key].update({para_key: para_val})
                combination_len_info.append(len(para_val))

    return parse_candidate, np.prod(combination_len_info)


def parse_xt_multi_case_paras(
        config_file, key_fields=("alg_config", "agent_config")  # pylint: disable=R0914
):
    """
    The multi-case config file parse entrance for benchmark
    Args:
        config_file:
        key_fields:

    Returns:
    """
    with open(config_file) as file_hander:
        yaml_obj = yaml.safe_load(file_hander)

    parse_candidate, combination_count = _get_combination_info(yaml_obj, key_fields)
    # print(parse_candidate, combination_count)

    para_prod_val = _get_product_value(parse_candidate)
    assert len(para_prod_val) == combination_count, "product error"
    # print(para_prod_val)

    ret_para = [deepcopy(yaml_obj) for _i in range(int(combination_count))]
    for i, obj in enumerate(ret_para):
        for key, _ in para_prod_val[i].items():
            sub_val = finditem(obj, key)
            sub_val.update(para_prod_val[i][key])

    return ret_para
