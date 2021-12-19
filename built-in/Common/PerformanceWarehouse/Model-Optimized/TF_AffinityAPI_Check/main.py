#! -*- coding:utf-8 -*-
# ============================================================================
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
# ============================================================================


import os
import logging
from absl import app, flags
import pandas as pd

import config as cfg
from common import TFGraphCheck, TrainingCodeCheck, ProfilingCheck, NpuLogCheck


flags.DEFINE_string(name="tfadpater_graph_dir", default='', help="a dir with tfadpater_graph")
flags.DEFINE_string(name="profiling_data_dir", default='', help="a dir with profiling_data")
flags.DEFINE_string(name="training_code_dir", default='', help="a dir with training_code")
flags.DEFINE_string(name="npu_log_dir", default='', help="a dir with npu_log")
flags.DEFINE_string(name="net_name", default=None, help="network name")

logging.basicConfig(level=logging.DEBUG,
                    filename='suggestions.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

def write_suggestions_to_excel(name, tf_sug, pro_sug, code_sug, log_sug):

    df = pd.DataFrame()
    df_cur = pd.DataFrame()
    all_sug = tf_sug + pro_sug + code_sug + log_sug
    df_name = pd.DataFrame([name] * len(all_sug), columns=['NetWork Name'])
    df_type = ['TFAdpater Suggestions'] * len(tf_sug) + ['Profiling Suggestions'] * len(pro_sug) + \
              ['TrainingCode Suggestions'] * len(code_sug) + ['NpuLog Suggestions'] * len(log_sug)

    df_type = pd.DataFrame(df_type, columns=['Suggestions Type'])
    df_all_sug = pd.DataFrame(all_sug, columns=['All Suggestions'])

    df_cur = pd.concat([df_cur, df_name.join(df_type, how='outer').join(df_all_sug, how='outer')])

    df = pd.concat([df, df_cur], axis=0)
    df.fillna(method='pad')

    writer = pd.ExcelWriter('analysis.xlsx')

    df.to_excel(writer, sheet_name='Sheet1', index=False)

    for column in df:
        column_length = max(df[column].astype(str).map(len).max(), len(column))
        col_idx = df.columns.get_loc(column)
        writer.sheets['Sheet1'].set_column(col_idx, col_idx, column_length)
    writer.save()

def _check_path_is_exists(path):
    return path and os.path.exists(path)

def start_check_performance_optimization(sub_path: dict):
    """Check performance optimization items

    :param sub_path: a dict containing the data corresponding to the check item
    :return:
    """
    if _check_path_is_exists(sub_path['tfadpater_graph']):
        tfadpatercheck = TFGraphCheck(sub_path['tfadpater_graph'])
        TFAdpater_Suggestions = []

    if _check_path_is_exists(sub_path['profiling_data']):
        profiling = ProfilingCheck(sub_path['profiling_data'])
        Profiling_Suggestions = []

    if _check_path_is_exists(sub_path['training_code']):
        codecheck = TrainingCodeCheck(sub_path['training_code'])
        TrainingCode_Suggestions = []

    if _check_path_is_exists(sub_path['npu_log']):
        npulogcheck = NpuLogCheck(sub_path['npu_log'])
        Npu_Log_Suggestions = []

    if tfadpatercheck:
        precision_flag, mix_compile_flag, op_debug_flag, precision_mode_type, loss_scale_suggestion \
            = tfadpatercheck.check_session_config_with_graph()

        if precision_flag:
            TFAdpater_Suggestions.append(cfg.PRECISION_MODE_SUGGESTIONS.format(precision_mode_type))
        else:
            TFAdpater_Suggestions.append(loss_scale_suggestion)

        if mix_compile_flag:
            TFAdpater_Suggestions.append(cfg.MIX_COMPILE_MODE_SUGGESTIONS)

        if op_debug_flag:
            TFAdpater_Suggestions.append(cfg.OP_DEBUG_LEVEL)


        if profiling.check_data_aug_with_step_trace():
            op_flag, enable_data_flag = tfadpatercheck.check_dataset_getnext_with_graph()

            if not op_flag:
                TFAdpater_Suggestions.append(cfg.DATASET_INTERFACE_SUGGESTIONS)

            if op_flag and not enable_data_flag:
                match_getnext_api, describe_api = codecheck.search_target_api(apis=cfg.GETNEXT_API)

                if match_getnext_api:
                    TFAdpater_Suggestions.append([cfg.GETNEXT_SUGGESTIONS, match_getnext_api])
                    logging.warning("4.3 The following api was found in the network: {}".format(match_getnext_api))
                    for describe in describe_api:
                        logging.warning("file:{}, line:{}, api:{}".format(describe[0], describe[1], describe[2]))
        else:
            logging.info("4.1. Data aug time is not long without checking whether it is sinking")

    if profiling:

        if profiling.check_dropout_with_summary():
            Profiling_Suggestions.append(cfg.DROPOUT_SUGGESTIONS)

        if profiling.check_dynamic_rnn_with_summary():
            match_rnn_api, describe_api = codecheck.search_target_api(apis=cfg.DYNAMIC_RNN_APIS)

            if match_rnn_api:
                Profiling_Suggestions.append([cfg.DYNAMIC_RNN_SUGGESTIONS, match_rnn_api])
                logging.warning("6.1 The following api was found in the network: {}".format(match_rnn_api))
                for describe in describe_api:
                    logging.warning("file:{}, line:{}, api:{}".format(describe[0], describe[1], describe[2]))

        profiling.check_top_task_duration_with_summary()

        if profiling.check_aicpu_dtype_with_summary():
           Profiling_Suggestions.append(cfg.AICPU_SUGGESTIONS)

        if profiling.check_TransposeD_with_summary():
            Profiling_Suggestions.append(cfg.TRANSPOSED_SUGGESTIONS)

    if codecheck:

        # 检查LSTM接口
        match_lstm_api, describe_api = codecheck.search_target_api(apis=cfg.LSTM_APIS)

        if match_lstm_api:
             TrainingCode_Suggestions.append('{}; api:{}'.format(cfg.LSTM_SUGGESTIONS, match_lstm_api))
             logging.warning("10. The following api was found in the network: {}".format(match_lstm_api))

             for describe in describe_api:
                 logging.warning("file:{}, line:{}, api:{}".format(describe[0], describe[1], describe[2]))
        else:
            logging.info("10. Lstm interface not found.")

        # 检查溢出检测接口
        match_finite_api, describe_api = codecheck.search_target_api(apis=cfg.FINITE_APIS)

        if match_finite_api and profiling.check_finite_with_summary():
            TrainingCode_Suggestions.append([cfg.FINITE_SUGGESTIONS, match_finite_api])
            logging.warning("11. The following api was found in the network: {}".format(match_finite_api))
            for describe in describe_api:
                logging.warning("file:{}, line:{}, api:{}".format(describe[0], describe[1], describe[2]))
        else:
            logging.info("11. Finite interface not found.")

    if npulogcheck:

        if npulogcheck.check_dataset_bottleneck_with_log():
            Npu_Log_Suggestions.append(cfg.DATASET_LOG_SUGGESTIONS)

    return TFAdpater_Suggestions, Profiling_Suggestions, TrainingCode_Suggestions, Npu_Log_Suggestions

def log_title(name):
    """

    :param name: network_name
    :return:
    """
    sentence = "Start to check {}".format(name)
    screen_width = 80
    text_width = len(sentence)
    box_width = text_width + 6
    left_width = (screen_width - box_width) // 2
    logging.info(' ' * left_width + "+" + '-' * (box_width - 2) + '+')
    logging.info(' ' * left_width + "|  " + ' ' * text_width + '  |')
    logging.info(' ' * left_width + "|  " + sentence + '  |')
    logging.info(' ' * left_width + "|  " + ' ' * text_width + '  |')
    logging.info(' ' * left_width + "+" + '-' * (box_width - 2) + '+')

def _get_dir_with_command(dir):
    return dir if dir else None

def use_offline_dir(flags_obj):
    """

    :param flags_obj:
    :return:
    """
    tfadpater_graph = _get_dir_with_command(flags_obj.tfadpater_graph_dir)

    profiling_data = _get_dir_with_command(flags_obj.profiling_data_dir)

    training_code = _get_dir_with_command(flags_obj.training_code_dir)

    npu_log = _get_dir_with_command(flags_obj.npu_log_dir)

    if not flags_obj.net_name:
        raise ValueError("network name is must")

    network_name = flags_obj.net_name

    sub_path = {
        'tfadpater_graph': tfadpater_graph,
        'profiling_data': profiling_data,
        'training_code': training_code,
        'npu_log': npu_log
    }

    return sub_path, network_name


def main(_):
    flags_obj = flags.FLAGS

    sub_path, network_name = use_offline_dir(flags_obj)
    log_title(network_name)

    tf_sug, pro_sug, code_sug, log_sug = \
        start_check_performance_optimization(sub_path)

    logging.info("End to check {}".format(network_name))

    write_suggestions_to_excel(name=network_name,
                               tf_sug=tf_sug,
                               pro_sug=pro_sug,
                               code_sug=code_sug,
                               log_sug=log_sug)


if __name__ == '__main__':
    app.run(main)

