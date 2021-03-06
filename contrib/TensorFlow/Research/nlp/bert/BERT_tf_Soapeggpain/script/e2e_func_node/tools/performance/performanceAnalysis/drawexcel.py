# -*- coding: UTF-8 -*-
import os
import xlsxwriter
import numpy as np

"""
draw excel picture
"""

EXCEL_FILE = None
EXCEL_SHEET = None
DEFAULT_COL_LEN = 12
FIRST_ROW_NUM = 0
DATA_START_ROW_NUM = 1
FIRST_STEP_ROW_NUM = 1
SECOND_STEP_ROW_NUM = 2
SERIAL_LIST_COL_NUM = 0

INTER_LIST_COL_NUM = 1
BP_FP_LIST_COL_NUM = 2
BPEND_TO_ITER_COL_NUM = 3
FPSTART_TO_AR1_COL_NUM = 4
RA1_LIST_COL_NUM = 5
AR1END_TO_BPEND_COL_NUM = 6
BPEND_TO_AR2START_COL_NUM = 7
RA2_LIST_COL_NUM = 8
AR2END_TO_ITEREND_COL_NUM = 9
ITER_LIST_COL_NUM = 10

MEAN_ITER_COL_NUM = 12
MEAN_INTER_COL_NUM = 13
MEAN_BP_FP_COL_NUM = 14
MEAN_RA1_COL_NUM = 15
MEAN_RA2_COL_NUM = 16
LOSS_LIST_COL_NUM = 1
MIOU_LIST_COL_NUM = 2


class DrawExcel:
    def __init__(self,
                 output_file_name=None,
                 iteration_list=None,
                 interval_list=None,
                 bp_fp_list=None,
                 bpend_to_iter_list=None,
                 allreduce1_start_list=None,
                 reduceadd1_list=None,
                 reduce1end_to_bpend_list=None,
                 bpend_to_reduce2start_list=None,
                 reduceadd2_list=None,
                 reduce2_to_iter_list=None):
        """
        :param output_file_name: output excel file
        :param iteration_list: whole iteration cost list
        :param interval_list: whole interval cost list
        :param bp_fp_list: whole bp_fp cost list
        :param reduceadd1_list: whole reduceadd1 cost list
        :param reduceadd2_list: whole reduceadd2 cost list
        """
        self.excel_file_name = output_file_name
        self.iteration_list = iteration_list
        self.interval_list = interval_list
        self.bp_fp_list = bp_fp_list
        self.bpend_to_iter_list = bpend_to_iter_list
        self.allreduce1_start_list = allreduce1_start_list
        self.reduce1end_to_bpend_list = reduce1end_to_bpend_list
        self.bpend_to_reduce2start_list = bpend_to_reduce2start_list
        self.reduce2_to_iter_list = reduce2_to_iter_list
        self.reduceadd1_list = reduceadd1_list
        self.reduceadd2_list = reduceadd2_list
        self.clear_output_file()
        self.draw_excel_file()

    def clear_output_file(self):
        """
        clear old output files
        :return:
        """
        if os.path.exists(self.excel_file_name):
            os.remove(self.excel_file_name)

    def draw_line(self):
        """
        draw excel chart
        :return:
        """
        global EXCEL_FILE
        global EXCEL_SHEET
        data_col_list = ('B', 'C', 'D', 'E', 'F','G')
        mean_col_list = ('L', 'M', 'N', 'O', 'P')
        name_col_list = ( 'Interval', 'BP FP', 'BPend_to_Iter','Reduceadd1', 'Reduceadd2','Iteration',)
        data_color_list = ['red', 'blue', 'green', 'orange', 'magenta']
        mean_color_list = ['brown', 'black', 'pink', 'purple', 'navy']
        for i in range(len(data_col_list)):
            chart_line = EXCEL_FILE.add_chart({'type': 'line'})
            chart_line.add_series({
                'name': '=DASHBOARD!$' + data_col_list[i] + '$1',
                'categories': '=DASHBOARD!$A$2:$A$' + str(len(self.iteration_list)),
                'values': '=DASHBOARD!$' + data_col_list[i] + '$2:$'
                          + data_col_list[i] + '$' + str(len(self.iteration_list)),
                'line': {'color': data_color_list[i], 'width': 1}
            })
            chart_mean = EXCEL_FILE.add_chart({'type': 'line'})
            chart_mean.add_series({
                'name': 'Mean Cost',
                'categories': '=DASHBOARD!$A$2:$A$' + str(len(self.iteration_list)),
                'values': '=DASHBOARD!$' + mean_col_list[i] + '$2:$'
                          + mean_col_list[i] + '$' + str(len(self.iteration_list)),
                'line': {'color': mean_color_list[i], 'width': 1.5}
            })
            chart_line.combine(chart_mean)
            chart_line.show_hidden_data()
            chart_line.set_title({'name': 'Calculate ' + name_col_list[i] + ' Cost Time'})
            chart_line.set_x_axis({'name': "Serial Number"})
            chart_line.set_y_axis({'name': 'Time (ms)'})
            chart_line.set_size({'width': len(self.iteration_list), 'height': 300})
            EXCEL_SHEET.insert_chart('G1', chart_line, {'x_offset': 25, 'y_offset': i * 300})

    def draw_excel_file(self):
        """
        generate excel file
        :return:
        """
        global EXCEL_FILE
        global EXCEL_SHEET
        EXCEL_FILE = xlsxwriter.Workbook(self.excel_file_name)
        EXCEL_SHEET = EXCEL_FILE.add_worksheet('DASHBOARD')
        first_row = ['index',
                     'liter_end_to_next_FP_start',
                     'BP_and_FP',
                     'BP_end_to_iter_end',
                     'FP_start_to_Reduce1_start',
                     'AllReduce1_total',
                     'Reduce1_end_to_BP_end',
                     'BP_end_to_reduce2_start',
                     'AllReduce2_total',
                     'AllReduce2_end_to_iter_end',
                     'total_time']
        first_row_style = EXCEL_FILE.add_format({
            'font_name': 'Times New Roman',
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#92D050'
        })
        other_row_style = EXCEL_FILE.add_format({
            'font_name': 'Times New Roman',
            'bold': False
        })
        error_row_style = EXCEL_FILE.add_format({
            'font_name': 'Times New Roman',
            'font_color': 'red',
            'bold': True
        })
        # write first row
        for i in range(len(first_row)):
            EXCEL_SHEET.write(FIRST_ROW_NUM, i, first_row[i], first_row_style)
            if len(first_row[i]) > DEFAULT_COL_LEN:
                EXCEL_SHEET.set_column(i, i, len(first_row[i]))
            else:
                EXCEL_SHEET.set_column(i, i, DEFAULT_COL_LEN)
        # write every column list
        if self.iteration_list is not None:
            iteration_time_array = np.array(self.iteration_list)
            mean_iteration_time = np.mean(iteration_time_array)
            for i in range(len(self.iteration_list)):
                # write serial number
                EXCEL_SHEET.write(i + DATA_START_ROW_NUM, SERIAL_LIST_COL_NUM, i + DATA_START_ROW_NUM, other_row_style)
                if self.iteration_list[i] > mean_iteration_time:
                    EXCEL_SHEET.write(i + SECOND_STEP_ROW_NUM,
                                      ITER_LIST_COL_NUM,
                                      self.iteration_list[i],
                                      error_row_style)
                else:
                    EXCEL_SHEET.write(i + SECOND_STEP_ROW_NUM,
                                      ITER_LIST_COL_NUM,
                                      self.iteration_list[i],
                                      other_row_style)
                EXCEL_SHEET.write(i + SECOND_STEP_ROW_NUM, MEAN_ITER_COL_NUM, mean_iteration_time, other_row_style)
        if self.interval_list is not None:
            interval_time_array = np.array(self.interval_list)
            mean_interval_time = np.mean(interval_time_array)
            for i in range(len(self.interval_list)):
                EXCEL_SHEET.write(i + DATA_START_ROW_NUM, SERIAL_LIST_COL_NUM, i + DATA_START_ROW_NUM, other_row_style)
                if self.interval_list[i] > mean_interval_time:
                    EXCEL_SHEET.write(i + SECOND_STEP_ROW_NUM,
                                      INTER_LIST_COL_NUM,
                                      self.interval_list[i],
                                      error_row_style)
                else:
                    EXCEL_SHEET.write(i + SECOND_STEP_ROW_NUM,
                                      INTER_LIST_COL_NUM,
                                      self.interval_list[i],
                                      other_row_style)
                EXCEL_SHEET.write(i + SECOND_STEP_ROW_NUM, MEAN_INTER_COL_NUM, mean_interval_time, other_row_style)
        if self.bp_fp_list is not None:
            bp_fp_time_array = np.array(self.bp_fp_list)
            mean_bp_fp_time = np.mean(bp_fp_time_array)
            for i in range(len(self.bp_fp_list)):
                EXCEL_SHEET.write(i + DATA_START_ROW_NUM, SERIAL_LIST_COL_NUM, i + DATA_START_ROW_NUM, other_row_style)
                if self.bp_fp_list[i] > mean_bp_fp_time:
                    EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM,
                                      BP_FP_LIST_COL_NUM,
                                      self.bp_fp_list[i],
                                      error_row_style)
                else:
                    EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM,
                                      BP_FP_LIST_COL_NUM,
                                      self.bp_fp_list[i],
                                      other_row_style)
                EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM, MEAN_BP_FP_COL_NUM, mean_bp_fp_time, other_row_style)
        if self.bpend_to_iter_list is not None:
            for i in range(len(self.bpend_to_iter_list)):
                EXCEL_SHEET.write(i + DATA_START_ROW_NUM, SERIAL_LIST_COL_NUM, i + DATA_START_ROW_NUM, other_row_style)
                EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM,
                                      BPEND_TO_ITER_COL_NUM,
                                      self.bpend_to_iter_list[i],
                                      other_row_style)

        if self.allreduce1_start_list is not None:
            for i in range(len(self.allreduce1_start_list)):
                EXCEL_SHEET.write(i + DATA_START_ROW_NUM, SERIAL_LIST_COL_NUM, i + DATA_START_ROW_NUM, other_row_style)
                EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM,
                                  FPSTART_TO_AR1_COL_NUM,
                                      self.allreduce1_start_list[i],
                                      other_row_style)

        if self.reduceadd1_list is not None:
            reduceadd1_time_array = np.array(self.reduceadd1_list)
            mean_reduceadd1_time = np.mean(reduceadd1_time_array)
            for i in range(len(self.reduceadd1_list)):
                EXCEL_SHEET.write(i + DATA_START_ROW_NUM, SERIAL_LIST_COL_NUM, i + DATA_START_ROW_NUM, other_row_style)
                if self.reduceadd1_list[i] > mean_reduceadd1_time:
                    EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM,
                                      RA1_LIST_COL_NUM,
                                      self.reduceadd1_list[i],
                                      error_row_style)
                else:
                    EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM,
                                      RA1_LIST_COL_NUM,
                                      self.reduceadd1_list[i],
                                      other_row_style)
                EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM, MEAN_RA1_COL_NUM, mean_reduceadd1_time, other_row_style)

        if self.reduce1end_to_bpend_list is not None:
            for i in range(len(self.reduce1end_to_bpend_list)):
                EXCEL_SHEET.write(i + DATA_START_ROW_NUM, SERIAL_LIST_COL_NUM, i + DATA_START_ROW_NUM, other_row_style)
                EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM,
                                  AR1END_TO_BPEND_COL_NUM,
                                      self.reduce1end_to_bpend_list[i],
                                      other_row_style)

        if self.bpend_to_reduce2start_list is not None:
            for i in range(len(self.bpend_to_reduce2start_list)):
                EXCEL_SHEET.write(i + DATA_START_ROW_NUM, SERIAL_LIST_COL_NUM, i + DATA_START_ROW_NUM, other_row_style)
                EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM,
                                  BPEND_TO_AR2START_COL_NUM,
                                      self.bpend_to_reduce2start_list[i],
                                      other_row_style)

        if self.reduceadd2_list is not None:
            reduceadd2_time_array = np.array(self.reduceadd2_list)
            mean_reduceadd2_time = np.mean(reduceadd2_time_array)
            for i in range(len(self.reduceadd2_list)):
                EXCEL_SHEET.write(i + DATA_START_ROW_NUM, SERIAL_LIST_COL_NUM, i + DATA_START_ROW_NUM, other_row_style)
                if self.reduceadd2_list[i] > mean_reduceadd2_time:
                    EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM,
                                      RA2_LIST_COL_NUM,
                                      self.reduceadd2_list[i],
                                      error_row_style)
                else:
                    EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM,
                                      RA2_LIST_COL_NUM,
                                      self.reduceadd2_list[i],
                                      other_row_style)
                EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM, MEAN_RA2_COL_NUM, mean_reduceadd2_time, other_row_style)

        if self.reduce2_to_iter_list is not None:
            for i in range(len(self.reduce2_to_iter_list)):
                EXCEL_SHEET.write(i + DATA_START_ROW_NUM, SERIAL_LIST_COL_NUM, i + DATA_START_ROW_NUM, other_row_style)
                EXCEL_SHEET.write(i + FIRST_STEP_ROW_NUM,
                                  AR2END_TO_ITEREND_COL_NUM,
                                      self.reduce2_to_iter_list[i],
                                      other_row_style)
        #self.draw_line()
        EXCEL_SHEET.set_column(MEAN_ITER_COL_NUM, MEAN_RA2_COL_NUM, None, None, {'hidden': True})
        EXCEL_FILE.close()

    @staticmethod
    def draw_loss_file(input_data1=None, input_data2=None, output_path=None):
        file_name = xlsxwriter.Workbook(output_path)
        file_sheet = file_name.add_worksheet('LossGraph')
        first_row = ['Serial Number',
                     'Loss List',
                     'mIOU List']
        first_row_style = file_name.add_format({
            'font_name': 'Times New Roman',
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#92D050'
        })
        other_row_style = file_name.add_format({
            'font_name': 'Times New Roman',
            'bold': False
        })
        for i in range(len(first_row)):
            file_sheet.write(FIRST_ROW_NUM, i, first_row[i], first_row_style)
            if len(first_row[i]) > DEFAULT_COL_LEN:
                file_sheet.set_column(i, i, len(first_row[i]))
            else:
                file_sheet.set_column(i, i, DEFAULT_COL_LEN)
        for i in range(len(input_data1)):
            file_sheet.write(i + DATA_START_ROW_NUM, SERIAL_LIST_COL_NUM, i + DATA_START_ROW_NUM, other_row_style)
            file_sheet.write(i + FIRST_STEP_ROW_NUM, LOSS_LIST_COL_NUM, float(input_data1[i]), other_row_style)
            file_sheet.write(i + FIRST_STEP_ROW_NUM, MIOU_LIST_COL_NUM, float(input_data2[i]), other_row_style)
        chart_line1 = file_name.add_chart({'type': 'line'})
        chart_line1.add_series({
            'name': '=LossGraph!$B$1',
            'categories': '=LossGraph!$A$2:$A$' + str(len(input_data1)),
            'values': '=LossGraph!$B$2:$B$' + str(len(input_data1)),
            'line': {'color': 'blue', 'width': 1.5}
        })
        chart_line2 = file_name.add_chart({'type': 'line'})
        chart_line2.add_series({
            'name': '=LossGraph!$C$1',
            'categories': '=LossGraph!$A$2:$A$' + str(len(input_data2)+1),
            'values': '=LossGraph!$C$2:$C$' + str(len(input_data2)+1),
            'line': {'color': 'orange', 'width': 1.5}
        })
        chart_line1.combine(chart_line2)
        chart_line1.set_title({'name': 'Loss and mIOU Trend'})
        chart_line1.set_x_axis({'name': "Serial Number"})
        chart_line1.set_y_axis({'name': 'Value'})
        chart_line1.set_size({'width': 1000, 'height': 600})
        file_sheet.insert_chart('D1', chart_line1, {'x_offset': 25, 'y_offset': 0})
        file_name.close()
