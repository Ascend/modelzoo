# -*- coding: UTF-8 -*-
import xlrd

"""
calculate time
"""

FIRST_STEP_ROW_NUM = 1
SECOND_STEP_ROW_NUM = 2
FP_START_COL_NUM = 0
RA1_START_COL_NUM = 1
RA1_END_COL_NUM = 2
BP_END_COL_NUM = 3
RA2_START_COL_NUM = 4
RA2_END_COL_NUM = 5
ITER_END_COL_NUM = 6
FREQUENCY_TIME = 10**5


class CalIterTime:
    def __init__(self, input_file_name=None):
        """
        :param input_file_name: excel file name
        """
        self.excel_file_name = input_file_name
        self.workbook = xlrd.open_workbook(filename=self.excel_file_name)
        self.booksheet = self.workbook.sheet_by_index(0)
        if self.excel_file_name is None:
            raise ValueError('Input file or output file is None.')

    def iteration_total_time(self):
        """
        calculate every iteration cost, using current iteration end time minus last iteration end time
        :return: mean cost of each iteration, variance and standard deviation of iteration time
        """
        total_time_list = []
        for i in range(SECOND_STEP_ROW_NUM, self.booksheet.nrows):
            try:
                every_total_cycle = float(self.booksheet.cell(i, ITER_END_COL_NUM).value) - \
                                    float(self.booksheet.cell(i - 1, ITER_END_COL_NUM).value)
            except ValueError:
                return None
            total_time_list.append(every_total_cycle / FREQUENCY_TIME)
        return total_time_list

    def iteration_interval_time(self):
        """
        calculate every iteration interval cost, using current fp start time minus last iteration end time
        :return: mean cost of each iteration interval
        """
        interval_time_list = []
        for i in range(SECOND_STEP_ROW_NUM, self.booksheet.nrows):
            try:
                every_interval_cycle = float(self.booksheet.cell(i, FP_START_COL_NUM).value) - \
                                       float(self.booksheet.cell(i - 1, ITER_END_COL_NUM).value)
            except ValueError:
                return None
            interval_time_list.append(every_interval_cycle / FREQUENCY_TIME)
        return interval_time_list

    def bp_fp_time(self):
        """
        calculate every iteration fp and bp cost, using current bp end time minus fp start time
        :return: mean cost of each iteration fp and bp calculation
        """
        bp_fp_time_list = []
        for i in range(FIRST_STEP_ROW_NUM, self.booksheet.nrows):
            try:
                every_bp_fp_cycle = float(self.booksheet.cell(i, BP_END_COL_NUM).value) - \
                                    float(self.booksheet.cell(i, FP_START_COL_NUM).value)
            except ValueError:
                return None
            bp_fp_time_list.append(every_bp_fp_cycle / FREQUENCY_TIME)
        return bp_fp_time_list

    def bpend_to_iter_time(self):
        """
        calculate every iteration fp and bp cost, using current bp end time minus fp start time
        :return: mean cost of each iteration fp and bp calculation
        """
        bpend_to_iter_time_list = []
        for i in range(FIRST_STEP_ROW_NUM, self.booksheet.nrows):
            try:
                every_bpend_to_iter_cycle = float(self.booksheet.cell(i, ITER_END_COL_NUM).value) - \
                                    float(self.booksheet.cell(i, BP_END_COL_NUM).value)
            except ValueError:
                return None
            bpend_to_iter_time_list.append(every_bpend_to_iter_cycle / FREQUENCY_TIME)
        return bpend_to_iter_time_list

    def allreduce1_start_time(self):
        """
        calculate every iteration fp and bp cost, using current bp end time minus fp start time
        :return: mean cost of each iteration fp and bp calculation
        """
        allreduce1_start_time_list = []
        for i in range(FIRST_STEP_ROW_NUM, self.booksheet.nrows):
            try:
                every_allreduce1_start_cycle = float(self.booksheet.cell(i, RA1_START_COL_NUM).value) - \
                                    float(self.booksheet.cell(i, FP_START_COL_NUM).value)
            except ValueError:
                return None
            allreduce1_start_time_list.append(every_allreduce1_start_cycle / FREQUENCY_TIME)
        return allreduce1_start_time_list

    def reduceadd1_time(self):
        """
        calculate every iteration reduceadd1 cost, using current reduceadd1 end time minus reduceadd1 start time
        :return: mean cost of each iteration reduceadd1 calculation
        """
        reduceadd1_time_list = []
        for i in range(FIRST_STEP_ROW_NUM, self.booksheet.nrows):
            try:
                every_reduceadd1_cycle = float(self.booksheet.cell(i, RA1_END_COL_NUM).value) - \
                                         float(self.booksheet.cell(i, RA1_START_COL_NUM).value)
            except ValueError:
                return None, None
            reduceadd1_time_list.append(every_reduceadd1_cycle / FREQUENCY_TIME)
        return reduceadd1_time_list

    def reduce1end_to_bpend_time(self):
        """
        calculate every iteration fp and bp cost, using current bp end time minus fp start time
        :return: mean cost of each iteration fp and bp calculation
        """
        reduce1end_to_bpend_list = []
        for i in range(FIRST_STEP_ROW_NUM, self.booksheet.nrows):
            try:
                every_reduce1end_to_bpend_cycle = float(self.booksheet.cell(i, BP_END_COL_NUM).value) - \
                                    float(self.booksheet.cell(i, RA1_END_COL_NUM).value)
            except ValueError:
                return None
            reduce1end_to_bpend_list.append(every_reduce1end_to_bpend_cycle / FREQUENCY_TIME)
        return reduce1end_to_bpend_list

    def bpend_to_reduce2start_time(self):
        """
        calculate every iteration fp and bp cost, using current bp end time minus fp start time
        :return: mean cost of each iteration fp and bp calculation
        """
        bpend_to_reduce2start_list = []
        for i in range(FIRST_STEP_ROW_NUM, self.booksheet.nrows):
            try:
                every_bpend_to_reduce2start_cycle = float(self.booksheet.cell(i, RA2_START_COL_NUM).value) - \
                                    float(self.booksheet.cell(i, BP_END_COL_NUM).value)
            except ValueError:
                return None
            bpend_to_reduce2start_list.append(every_bpend_to_reduce2start_cycle / FREQUENCY_TIME)
        return bpend_to_reduce2start_list

    def reduceadd2_time(self):
        """
        calculate every iteration reduceadd1 cost, using current reduceadd1 end time minus reduceadd1 start time
        :return: mean cost of each iteration reduceadd1 calculation
        """
        reduceadd2_time_list = []
        for i in range(FIRST_STEP_ROW_NUM, self.booksheet.nrows):
            try:
                every_reduceadd2_cycle = float(self.booksheet.cell(i, RA2_END_COL_NUM).value) - \
                                         float(self.booksheet.cell(i, RA2_START_COL_NUM).value)
            except ValueError:
                return None, None
            reduceadd2_time_list.append(every_reduceadd2_cycle / FREQUENCY_TIME)
        return reduceadd2_time_list

    def reduce2_to_iter_time(self):
        """
        calculate every iteration reduceadd1 cost, using current reduceadd1 end time minus reduceadd1 start time
        :return: mean cost of each iteration reduceadd1 calculation
        """
        reduce2_to_iter_time_list = []
        for i in range(FIRST_STEP_ROW_NUM, self.booksheet.nrows):
            try:
                every_reduce2_to_iter_cycle = float(self.booksheet.cell(i, ITER_END_COL_NUM).value) - \
                                         float(self.booksheet.cell(i, RA2_END_COL_NUM).value)
            except ValueError:
                return None, None
            reduce2_to_iter_time_list.append(every_reduce2_to_iter_cycle / FREQUENCY_TIME)
        return reduce2_to_iter_time_list