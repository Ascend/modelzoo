# -*- coding: UTF-8 -*-
import zipfile
import os
import shutil
import numpy as np
import subprocess as sp
import multiprocessing as mp
import parsedata as pd
import genexcel as ge
import calculatetime as ct
import drawexcel as de
import compareckpt as cc


PROFILING_FILE_PATH = '/var/log/npu/profiling/'
#RESULT_PATH = '/home/deeplabv3/e2e_test_new/result/'
#JOB_PATH = os.popen("ls -lt " + RESULT_PATH + " | grep -v total | head -n1 | awk '{print $NF}'").read().strip('\n')
#CKPT_PATH = os.path.join(RESULT_PATH, JOB_PATH)
DEVICE_NUMBER = 8
BIN_FILE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'extract_file')
PARSED_FILE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'parsed_file')
COMBINE_FILE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'combine_file')
EXCEL_FILE_PATH = os.path.abspath(os.path.dirname(__file__))


def _env_clear():
    """
    clear old extract file, parsed file and combine file.
    :return: null
    """
    if os.path.isdir(BIN_FILE_PATH):
        shutil.rmtree(BIN_FILE_PATH)
    os.mkdir(BIN_FILE_PATH)
    if os.path.isdir(PARSED_FILE_PATH):
        shutil.rmtree(PARSED_FILE_PATH)
    os.mkdir(PARSED_FILE_PATH)
    if os.path.isdir(COMBINE_FILE_PATH):
        shutil.rmtree(COMBINE_FILE_PATH)
    os.mkdir(COMBINE_FILE_PATH)


def _parse_data(input_file):
    """
    parse bin file data and save data into parsed file.
    :param input_file: bin file path
    :return: null
    """
    pd.ParseData(input_file_name=os.path.join(BIN_FILE_PATH, input_file),
                 output_file_name=os.path.join(PARSED_FILE_PATH, input_file))


def _gen_excel(i):
    """
    read all pieces of parsed files and combine into one file named device_i_profiling.log, then draw excel graph.
    :param i: device index
    :return: null
    """
    cmd_i = "ls " + PARSED_FILE_PATH + "/*tag." + str(i) + "* | wc -l"
    result_i = sp.Popen(cmd_i, shell=True, stdout=sp.PIPE, stderr=sp.PIPE).stdout.read().decode('utf-8').strip('\n')
    if int(result_i) != 0:
        combine_file = os.path.join(COMBINE_FILE_PATH, "device_%s_profiling.log" % i)
        with open(combine_file, 'wb') as out_file:
            for j in range(int(result_i)):
                cmd_j = "ls " + PARSED_FILE_PATH + "/*tag." + str(i) + ".slice_" + str(j)
                result_j = os.popen(cmd_j).read().strip('\n')
                with open(result_j, 'rb') as in_file:
                    infile_data = in_file.read()
                    out_file.write(infile_data)
        ge.GenExcel(input_file_name=combine_file,
                    output_file_name=os.path.join(combine_file + '.xlsx'))
        # xxx_list include every step xxx value
        calculate_time = ct.CalIterTime(input_file_name=os.path.join(combine_file + '.xlsx'))
        iteration_list = calculate_time.iteration_total_time()
        interval_list = calculate_time.iteration_interval_time()
        bp_fp_list = calculate_time.bp_fp_time()
        bpend_to_iter_list = calculate_time.bpend_to_iter_time()

        de.DrawExcel(output_file_name=(os.path.join(EXCEL_FILE_PATH, 'analysis_performance_tag.' + str(i) + '.xlsx')),
                     iteration_list=iteration_list,
                     interval_list=interval_list,
                     bp_fp_list=bp_fp_list,
                     bpend_to_iter_list=bpend_to_iter_list)


def _un_zip():
    """
    unzip zip files and return unzip file name list
    :return: unzip file name list
    """
    zip_list = []
    output_list = []
    for root, dirs, files in os.walk(PROFILING_FILE_PATH):
        for file in files:
            if 'training_trace' in file and os.path.splitext(file)[1] == '.zip':
                zip_list.append(os.path.join(root, file))
    if len(zip_list) != 0:
        for zip_package in zip_list:
            with zipfile.ZipFile(zip_package) as zip_file:
                for names in zip_file.namelist():
                    if 'done' in names:
                        pass
                    else:
                        zip_file.extract(names, BIN_FILE_PATH)
                        output_list.append(names)
    else:
        for root, dirs, files in os.walk(PROFILING_FILE_PATH):
            for file in files:
                if 'training_trace' in file and 'done' not in file:
                    open(os.path.join(BIN_FILE_PATH, file), "wb").write(open(os.path.join(root, file), "rb").read())
                    output_list.append(file)
    return sorted(output_list)


def _cosine_similarity(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denorm = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denorm
    sim = 0.5 + 0.5 * cos
    return sim


if __name__ == '__main__':
    _env_clear()
    # performance analysis
    file_list = _un_zip()
    process_list = []
    for file_name in file_list:
        p = mp.Process(target=_parse_data, args=(file_name,))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()
    process_list = []
    for iter_num in range(DEVICE_NUMBER):
        p = mp.Process(target=_gen_excel, args=(iter_num,))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()
    # checkpoint consistency analysis
    # try:
    #     cc.CompareCkpt(CKPT_PATH)
    # except ValueError as e:
    #     print(e)
    # # loss analysis
    #
    # cmd = "ls " + CKPT_PATH + "/train_*.log | wc -l"
    # device_num = os.popen(cmd).read().strip('\n')
    # benchmark_loss = np.load(os.path.join(EXCEL_FILE_PATH, 'benchmark.npy'))
    # benchmark_loss.tolist()
    # for device_index in range(int(device_num)):
    #     loss_path = os.path.join(CKPT_PATH, 'train_' + str(device_index) + '.log')
    #     cmd = "cat " + loss_path + " | grep total_loss | awk '{print $9}'"
    #     loss_result = os.popen(cmd)
    #     loss_list = loss_result.read().strip('\n').split('\n')
    #     cmd = "cat " + loss_path + " | grep total_loss | awk '{print $6}' | awk -F',' '{print $1}'"
    #     miou_result = os.popen(cmd)
    #     miou_list = miou_result.read().strip('\n').split('\n')
    #     # draw loss graph
    #     try:
    #         de.DrawExcel.draw_loss_file(input_data1=loss_list,
    #                                     input_data2=miou_list,
    #                                     output_path=os.path.join(EXCEL_FILE_PATH,
    #                                                              'loss_miou_trend_tag.' + str(device_index) + '.xlsx'))
    #     except ValueError as e:
    #         print(e)
    #     # compare with benchmark using cosine similarity
    #     mini_num = len(loss_list) if len(loss_list) < len(benchmark_loss) else len(benchmark_loss)
    #     npu_loss_list = loss_list[0:mini_num]
    #     benchmark_loss_list = benchmark_loss[0:mini_num]
    #     cos_sim = _cosine_similarity(np.asanyarray(npu_loss_list, dtype=float),
    #                                  np.asanyarray(benchmark_loss_list, dtype=float))
    #     print('cosine similarity %f is satisfied requirements' % cos_sim) if cos_sim > 0.8 else \
    #         print('cosine similarity %f is not satisfied requirements' % cos_sim)

