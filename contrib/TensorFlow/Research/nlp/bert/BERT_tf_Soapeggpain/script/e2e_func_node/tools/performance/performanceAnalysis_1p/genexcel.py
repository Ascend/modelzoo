# -*- coding: UTF-8 -*-
import os
import xlsxwriter

"""
generate excel file
"""

DEFAULT_COL_LEN = 12
FIRST_ROW_NUM = 0
DATA_START_ROW_NUM = 1
FP_START_COL_NUM = 0
RA1_START_COL_NUM = 1
RA1_END_COL_NUM = 2
BP_END_COL_NUM = 3
RA2_START_COL_NUM = 4
RA2_END_COL_NUM = 5
ITER_END_COL_NUM = 6


class GenExcel:
    def __init__(self, input_file_name=None, output_file_name=None):
        """
        :param input_file_name: input parsed profiling logs
        :param output_file_name: output excel file
        """
        self.parsed_file_name = input_file_name
        self.excel_file_name = output_file_name
        if self.parsed_file_name is not None and self.excel_file_name is not None:
            self.clear_output_file()
            self.gen_excel_file()
        else:
            raise ValueError('Input file or output file is None.')

    def clear_output_file(self):
        """
        clear old output files
        :return:
        """
        if os.path.exists(self.excel_file_name):
            os.remove(self.excel_file_name)

    def gen_excel_file(self):
        """
        generate excel file from parsed profiling files
        :return:
        """
        excel_file = xlsxwriter.Workbook(self.excel_file_name)
        excel_sheet = excel_file.add_worksheet('device ' + self.parsed_file_name.split('_')[-1])
        first_row = ['FP Start',
                     'Reduceadd1 Start',
                     'Reduceadd1 End',
                     'BP End',
                     'Iteration End']
        first_row_style = excel_file.add_format({
            'font_name': 'Times New Roman',
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#92D050'
        })
        other_row_style = excel_file.add_format({
            'font_name': 'Times New Roman',
            'bold': False
        })
        for i in range(len(first_row)):
            excel_sheet.write(FIRST_ROW_NUM, i, first_row[i], first_row_style)
            # if word length bigger than DEFAULT_COL_LEN, using word length(like 'Reduceadd1 Start') as column length,
            # else(like 'BP End') using DEFAULT_COL_LEN as column length.
            if len(first_row[i]) > DEFAULT_COL_LEN:
                excel_sheet.set_column(i, i, len(first_row[i]))
            else:
                excel_sheet.set_column(i, i, DEFAULT_COL_LEN)
        with open(self.parsed_file_name, 'r') as pf:
            row_num = DATA_START_ROW_NUM
            # distinguish reduceadd1 and reduceadd2, flag == 0 means reduceadd1, flag == 1 means reduceadd2
            ra_start_flag = 0
            ra_end_flag = 0
            for line in pf.readlines():
                if 'FP_start' in line:
                    fp_start_value = line.split('=')[-1].strip()
                    excel_sheet.write(row_num, FP_START_COL_NUM, float(fp_start_value), other_row_style)
                elif 'BP_end' in line:
                    bp_end_value = line.split('=')[-1].strip()
                    excel_sheet.write(row_num, BP_END_COL_NUM, float(bp_end_value), other_row_style)
                elif 'iteration_end' in line:
                    ie_end_value = line.split('=')[-1].strip()
                    excel_sheet.write(row_num, ITER_END_COL_NUM, float(ie_end_value), other_row_style)
                    row_num += 1
        excel_file.close()
