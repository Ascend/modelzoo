# -*- coding: UTF-8 -*-
import struct
import os

"""
parse data from profiling files
"""


class ParseData:
    def __init__(self, input_file_name=None, output_file_name=None):
        """
        :param input_file_name: profiling bin file name
        :param output_file_name: parsed profiling log file
        """
        self.profiling_file_name = input_file_name
        self.save_file_name = output_file_name
        if self.profiling_file_name is not None and self.save_file_name is not None:
            self.clear_output_file()
            self.parse_trace_data()
        else:
            raise ValueError('Input file or output file is None.')

    def parse_trace_data(self):
        """
        parse trace file data
        :return:
        """
        iteration_id = 1
        with open(self.profiling_file_name, 'rb') as bin_file:
            while True:
                job_id_bin = bin_file.read(struct.calcsize("=Q"))
                if job_id_bin:
                    job_id = struct.unpack("=Q", job_id_bin)[0]
                    if job_id > 255:
                        _ = struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]
                        _ = struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]
                        _ = struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]
                        check_id_bin = bin_file.read(struct.calcsize("=Q"))
                        self.get_fp_bp(check_id_bin, bin_file)
                        iteration_id += 1
                    else:
                        _ = bin_file.read(struct.calcsize("=HHQ"))
                        continue
                else:
                    break

    def get_fp_bp(self, check_id_bin, bin_file):
        """
        parse fp start, bp end, reduce start, reduce end
        :param check_id_bin: bin file blocks
        :param bin_file: profiling bin file name
        :return:
        """
        with open(self.save_file_name, 'a+') as output_file:
            while check_id_bin:
                check_id = struct.unpack("=Q", check_id_bin)[0]
                if check_id < 255:
                    if check_id == 1:
                        output_file.write("cp_add.FP_stream =" + str(
                            struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                        output_file.write("cp_add.FP_task =" + str(
                            struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                        output_file.write("cp_add.FP_start =" + str(
                            struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                    elif check_id == 2:
                        output_file.write("cp_add.BP_stream =" + str(
                            struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                        output_file.write("cp_add.BP_task =" + str(
                            struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                        output_file.write("cp_add.BP_end =" + str(
                            struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                    elif check_id % 2 == 1:
                        output_file.write("cp_reduceadd = cp_add.all_reduces.add()")
                        output_file.write("cp_reduceadd.start_stream =" + str(
                            struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                        output_file.write("cp_reduceadd.start_task =" + str(
                            struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                        output_file.write("cp_reduceadd.start =" + str(
                            struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                    elif check_id % 2 == 0:
                        output_file.write("cp_reduceadd.end_stream =" + str(
                            struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                        output_file.write("cp_reduceadd.end_task =" + str(
                            struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                        output_file.write("cp_reduceadd.end =" + str(
                            struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                    check_id_bin = bin_file.read(struct.calcsize("=Q"))
                elif check_id == 255:
                    output_file.write("cp_add.iter_stream =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_add.iter_task =" + str(
                        struct.unpack("=H", bin_file.read(struct.calcsize("=H")))[0]) + '\n')
                    output_file.write("cp_add.iteration_end =" + str(
                        struct.unpack("=Q", bin_file.read(struct.calcsize("=Q")))[0]) + '\n')
                    break
                else:
                    break

    def clear_output_file(self):
        """
        clear old output files
        :return:
        """
        if os.path.exists(self.save_file_name):
            os.remove(self.save_file_name)
