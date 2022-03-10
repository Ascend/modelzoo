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
import re
import logging
import pandas as pd

import config as cfg

logging.basicConfig(level=logging.DEBUG,
                    filename='suggestions.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )


class TFGraphCheck(object):
    """check GE graph here."""

    def __init__(self, sub_dir):
        """Init function of GEGraphCheck

        :param sub_dir: a dir contains ge txt or pbtxt
        """

        self.ge_check_list = []
        self.sub_dir = sub_dir
        self.os_system1 = "grep -rnwi -A2 "
        self.os_system2 = self.sub_dir + "|head -n 3 | grep 's: '|awk -F '\"' '{print$2}'"
        self.os_system3 = self.sub_dir + "| grep 'op: '|awk -F '\"' '{print $2}' |head - n 1"

    def check_session_config_with_graph(self):
        precision_flag = False
        mix_compile_flag = False
        op_debug_flag = False

        precision_mode = os.popen(self.os_system1 + "_precision_mode" + " " + self.os_system2)
        precision_mode_type = precision_mode.read().strip()

        if precision_mode_type != "allow_mix_precision":
            precision_flag = True
            logging.warning(
                "1.The precision_mode is {}, suggest to set allow_mix_precision.".format(precision_mode_type))
        else:
            logging.info("1.The precision_mode is allow_mix_precision")
            _, loss_scale_suggestion = self.check_loss_scale_with_graph()

        mix_compile_mode = os.popen(self.os_system1 + "_mix_compile_mode" + " " + self.os_system2)
        mix_compile_mode_type = mix_compile_mode.read().strip()

        if mix_compile_mode_type == 1:
            mix_compile_flag = True
            logging.warning("2.The mix_compile_mode is True, suggest to set False")
        else:
            logging.info("2.The mix_compile_mode is False")

        op_debug_level = os.popen(self.os_system1 + "_op_debug_level" + " " + self.os_system2)
        op_debug_level_type = op_debug_level.read().strip()

        if op_debug_level_type == 1:
            op_debug_flag = True
            logging.warning("3.The op_debug_level is 1, suggest to set 0")
        else:
            logging.info("3.The op_debug_level is 0")

        return precision_flag, mix_compile_flag, op_debug_flag, precision_mode_type, loss_scale_suggestion

    def check_loss_scale_with_graph(self):
        isexist_flag = False
        isrepeat_flag = True

        pattern = re.compile('[\s]+name: \"NpuGetFloatStatus\"')

        for parent, dirnames, filenames in os.walk(self.sub_dir, followlinks=True):
            for file in filenames:
                file_path = os.path.join(parent, file)
                if os.path.isfile(file_path):

                    count = 0
                    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                        filelines = f.readlines()
                        for line in filelines:
                            if re.match(pattern, line):
                                count += 1

                        if count == 1:
                            isexist_flag = True
                        elif count > 1:
                            return isrepeat_flag, "网络代码中重复开启LossScale, 请检查去重."

        if isexist_flag:
            return isexist_flag, "网络脚本中已开启LossScale."
        else:
            return isexist_flag, "网络使用的是allow_mix_precision精度模式, 但未开启LossScale, 请开启."

    def check_dataset_getnext_with_graph(self):
        match_op = []
        op_flag, enable_data_flag = False, False

        for operator in cfg.DATASET_OP:
            if os.popen(self.os_system1 + operator + " " + self.os_system3).read():
                match_op.append(operator)

        if match_op:
            op_flag = True
            logging.info("4.1 The network uses the Dataset, and is checking whether the enable_data_pre_proc is used.")
            if os.popen(self.os_system1 + "_enable_data_pre_proc" + " " + self.os_system2).read():
                enable_data_flag = True
                logging.info("4.2 The network uses enable_data_pre_proc.")

            else:
                logging.warning("4.2 The Getnext operator does not sink, check the script, "
                                "replace make_one_shot_iterator with make_initializable_iterator")

        else:
            logging.warning("4.1 The network does not use the Dataset, try to change to the Dataset interface mode")

        return op_flag, enable_data_flag


class ProfilingCheck(object):
    """check profiling data here."""

    def __init__(self, sub_dir):
        """

        :param sub_dir: a dir contains profiling data
        """
        self.profiling_check_list = []
        self.profiling_dir = sub_dir
        self.job_sub_dir = ['summary', 'timeline']
        if self.profiling_dir:
            self.job_sub_dir_profile = self.get_job_dir_profile()

    def get_job_dir_profile(self) -> dict:
        """Find the target aicpu、op_summary、op_statistic、step_trace profile


        :return:
        """
        data_job_dir, summary_profile = [], []
        aicpu, op_summary, op_statistic, step_trace = [], [], [], []

        for dir in os.listdir(self.profiling_dir):
            if os.path.isdir(os.path.join(self.profiling_dir, dir)):
                job_dir = os.path.join(self.profiling_dir, dir)

        data_job_dir += [os.path.join(job_dir, x) for x in self.job_sub_dir
                         if os.path.isdir(os.path.join(job_dir, x))]

        for folder in data_job_dir:
            summary_profile += [os.path.relpath(os.path.join(folder, x))
                                for x in os.listdir(folder)
                                if os.path.isfile(os.path.join(folder, x)) and 'summary' in folder]

        for file in summary_profile:
            if 'aicpu' in file:
                aicpu.append(file)
            if 'op_summary' in file:
                op_summary.append(file)
            if 'op_statistic' in file:
                op_statistic.append(file)
            if 'step_trace' in file:
                step_trace.append(file)

        return {
            'aicpu': aicpu,
            'op_summary': op_summary,
            'op_statistic': op_statistic,
            'step_trace': step_trace
        }

    def search_target_operator(self, flag: bool, nodes: str):
        """Find the target operator in the profiling file

        :param flag:
        :param nodes: Target operator
        :return:
        """
        total_time = 0

        for aicpu_file in self.job_sub_dir_profile['aicpu']:
            logging.info('the aicpu csv is {}.'.format(aicpu_file))
            aicpu_csv = pd.read_csv(aicpu_file)
            for i in aicpu_csv.index:
                if isinstance(nodes, list):
                    for node in nodes:
                        if node in aicpu_csv.iloc[i]["Node"]:
                            flag = True
                            total_time += aicpu_csv.iloc[i]["Total_time(ms)"]

                if isinstance(nodes, str):
                    if nodes in aicpu_csv.iloc[i]["Node"]:
                        flag = True
                        total_time += aicpu_csv.iloc[i]["Total_time(ms)"]

        return flag, total_time

    def check_dropout_with_summary(self):
        """Whether RandomUniform appears and it is an AICPU operator

        :return:
        """
        dropout_flag = False

        dropout_flag, total_time = self.search_target_operator(dropout_flag, cfg.DROPOUT_NODE)

        if dropout_flag == True:
            logging.warning("5. RandomUniform operator is AICPU operator, consume time:{}(ms)"
                            ", can use npu_ops.dropout to replace.".format(total_time))
            return True
        else:
            logging.info("5. No dropout operators found.")
            return False

    def check_dynamic_rnn_with_summary(self):
        dynamic_rnn_flag = False
        dynamic_rnn_flag, total_time = self.search_target_operator(dynamic_rnn_flag, cfg.DYNAMIC_RNN_NODE)

        if dynamic_rnn_flag == True:
            logging.warning("6. Find dynamic_rnn, consume time:{}(ms).".format(total_time))
            return True

        else:
            logging.info("6. No dynamic_rnn operators found.")
            return False

    def check_top_task_duration_with_summary(self):
        for op_summary_file in self.job_sub_dir_profile['op_summary']:
            logging.info("7. Start to analysis op_summary_file: {}".format(op_summary_file))
            op_summary_csv = pd.read_csv(op_summary_file)
            op_summary_csv = op_summary_csv.sort_values(by="Task Duration(us)", ascending=False)
            if op_summary_csv.shape[0] > 10:
                for i in range(10):
                    logging.info(
                        "op_type:{}; Block Dim:{}; Task Duration(us):{}".format(op_summary_csv.iloc[i]["OP Type"],
                                                                                op_summary_csv.iloc[i]["Block Dim"],
                                                                                op_summary_csv.iloc[i][
                                                                                        "Task Duration(us)"]))
            else:
                for i in range(op_summary_csv.shape[0]):
                    logging.info(
                        "op_type:{}; Block Dim:{}; Task Duration(us):{}".format(op_summary_csv.iloc[i]["OP Type"],
                                                                                op_summary_csv.iloc[i]["Block Dim"],
                                                                                op_summary_csv.iloc[i][
                                                                                    "Task Duration(us)"]))

    def check_aicpu_dtype_with_summary(self):
        """ check AI_CPU operator dtype whether int64.

        :return:
        """
        aicpu_flag = False

        for op_summary_file in self.job_sub_dir_profile['op_summary']:
            all_aicpu_time = 0
            op_summary_csv = pd.read_csv(op_summary_file)
            all_op_time = op_summary_csv["Task Duration(us)"].sum()
            for i in op_summary_csv.index:
                task_type = op_summary_csv.iloc[i]["Task Type"]
                input_data_type = op_summary_csv.iloc[i]["Input Data Types"]

                if task_type == "AI_CPU" and "DT_INT64" in input_data_type:
                    # aicpu_flag = True
                    all_aicpu_time = op_summary_csv.iloc[i]["Task Duration(us)"]
                    logging.warning("AI_CPU int64 operator: {}".format(op_summary_csv.iloc[i]["Op Name"]))


            if all_aicpu_time / all_op_time > 0.1:
                aicpu_flag = True

        if aicpu_flag:
            logging.info("8. Here has some aicpu operator with dtype int64, please pay attention")
            return True
        else:
            logging.info("8. Not find some long time aicpu operators with dtype int64")
            return False

    def check_TransposeD_with_summary(self):
        """whether has some long time-consuming TransposeD operator.

        :return:
        """
        TransposeD_flag = False

        for op_summary_file in self.job_sub_dir_profile['op_summary']:
            op_summary_csv = pd.read_csv(op_summary_file)
            all_op_time = op_summary_csv["Task Duration(us)"].sum()
            for i in op_summary_csv.index:
                op_type = op_summary_csv.iloc[i]["OP Type"]
                task_duration = op_summary_csv.iloc[i]["Task Duration(us)"]
                if op_type == "TransposeD" and (task_duration / all_op_time > 0.01 or task_duration > 1000):
                    logging.warning("{} operator Task Duration {}(us), TransposeD Task Duration{}(us)"
                                    .format(op_summary_file, all_op_time, task_duration))
                    TransposeD_flag = True
                    break

        if TransposeD_flag:
            logging.warning("9. Here has some long time-consuming TransposeD operator, please pay attention")
            return True
        else:
            logging.info("9. Not find a long time-consuming TransposeD operator")
            return False

    def check_finite_with_summary(self):
        IsFinite_flag = False

        for op_summary_file in self.job_sub_dir_profile['op_summary']:
            op_summary_csv = pd.read_csv(op_summary_file)
            for i in op_summary_csv.index:
                op_type = op_summary_csv.iloc[i]["OP Type"]
                if op_type == "IsFinite":
                    IsFinite_flag = True
                    break

        return IsFinite_flag


    def check_data_aug_with_step_trace(self):
        data_aug_flag = False

        for step_trace_file in self.job_sub_dir_profile['step_trace']:
            step_trace = pd.read_csv(step_trace_file)
            step_trace = step_trace.dropna(axis=0, how='any')
            model_id = set(step_trace["Model ID"])
            average_data_aug_bound = 0
            average_iteration_time = 0

            if model_id:
                for id in model_id:
                    model_id_data = step_trace.loc[(step_trace["Model ID"] == id)]

                    if len(model_id_data) > 3:
                        all_data_aug_bound = model_id_data["Data Aug Bound(us)"][3:].sum()
                        average_data_aug_bound += all_data_aug_bound // (len(model_id_data) - 3)

                        all_iteration_time = model_id_data["Iteration Time(us)"][3:].sum()
                        average_iteration_time += all_iteration_time // (len(model_id_data) - 3)

            if average_data_aug_bound\
                    and average_iteration_time:
                data_aug = average_data_aug_bound / (average_data_aug_bound + average_iteration_time)
                logging.info("4. The data_aug in all time is: {}".format(data_aug))

                if data_aug > 0.1:
                    data_aug_flag = True

            return data_aug_flag


class TrainingCodeCheck(object):
    """check code here."""

    def __init__(self, sub_dir):
        """

        :param sub_dir:
        """
        self.code_check_list = []
        self.sub_dir = sub_dir

    def search_target_api(self, apis):
        """find in the code according to the api list.

        :param apis:
        :return:
        """

        match_api = set()
        describe_api = []

        for api in apis:
            for dirpath, dirnames, filenames in os.walk(self.sub_dir, followlinks=True):
                for file in filenames:
                    file_path = os.path.join(dirpath, file)
                    if os.path.isfile(file_path) and os.path.splitext(file_path)[1] == '.py':
                        with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                            filelines = f.readlines()
                            for i in range(len(filelines)):
                                if not filelines[i].startswith("#") and api in filelines[i]:
                                    describe_api.append([file_path, i, api])
                                    match_api.add(api)

        return match_api, describe_api

class NpuLogCheck(object):
    """check npu log here."""

    def __init__(self, sub_dir):
        """

        :param sub_dir:
        """
        self.sub_dir = sub_dir
        self.training_log = "This training has reached the data-preprocess performance bottleneck"

    def check_zeros(self, nums):
        nums = [int(num) for num in nums]
        for i in range(len(nums) - 2):
            if nums[i] + nums[i + 1] + nums[i + 2] == 0:
                return True
        return False

    def check_data_preprocess_bottleneck(self):
        bottle_cmd = "grep -rn 'performance bottleneck' " + self.sub_dir + "/device-*/* | wc -l"
        dv_tdt_cmd = "grep -rn 'outContent:Queue_Edge_' " + self.sub_dir + "/device-*/*|awk '{print $3}'|awk -F ':' '{print $NF}'"
        host_tdt_cmd = "grep -rn 'has sent a tdtDataElem' " + self.sub_dir + "/plog/* |awk '{print$22}' |awk -F ',' '{print$1}'|sort|uniq"
        host_queue_cmd = "grep -rn 'Host queue' " + self.sub_dir + "/plog/*|awk '{print $NF}'"

        bottel_res = os.popen(bottle_cmd).read()
        if int(bottel_res[0]) > 0:  # DP-queue空
            print(0)
            logging.info("12. This training has reached the data-preprocess performance bottleneck")
            dv_tdt_res = os.popen(dv_tdt_cmd).read()
            if self.check_zeros(dv_tdt_res):  # DEVICE-TDT 空
                host_tdt_res = os.popen(host_tdt_cmd).read()
                if self.check_zeros(host_tdt_res):  # HOST-TDT 空
                    host_queue_res = os.popen(host_queue_cmd).read()
                    if self.check_zeros(host_queue_res):  # HOST-Queue 空
                        logging.info(
                            "12. Some iters found no data in Host Queue, it may have performance bottleneck here!")
                        return True
                    else:
                        logging.info(
                            "12. Have enough data in Host Queue but null in Host TDT Queue, it may have performance bottleneck here!")
                        return True
                else:
                    logging.info(
                        "12. Have enough data in Host TDT Queue but null in Device TDT Queue, it may have performance bottleneck here!")
                    return True
            else:
                logging.info(
                    "12. Have data in Device TDT Queue data but null in DP Queue,it may have performance bottleneck here! ")
                return True
        else:
            print(1)
            logging.info("12. Not found data-preprocess performance bottleneck")
            return False

# class NpuLogCheck(object):
#     """check npu log here."""

#     def __init__(self, sub_dir):
#         """

#         :param sub_dir:
#         """
#         self.sub_dir = sub_dir
#         self.training_log = "This training has reached the data-preprocess performance bottleneck"

#     def check_dataset_bottleneck_with_log(self):
#         bottleneck_flag = False

#         for parent, dirnames, filenames in os.walk(self.sub_dir, followlinks=True):
#             for file in filenames:
#                 file_path = os.path.join(parent, file)
#                 if os.path.isfile(file_path):
#                     with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
#                         filelines = f.readlines()
#                         for line in filelines:
#                             if self.training_log in line:
#                                 bottleneck_flag = True

#         if bottleneck_flag:
#             logging.warning("12. This training has reached the data-preprocess performance bottleneck")
#             return True
#         else:
#             logging.info("12. Not found data-preprocess performance bottleneck")
#             return False
