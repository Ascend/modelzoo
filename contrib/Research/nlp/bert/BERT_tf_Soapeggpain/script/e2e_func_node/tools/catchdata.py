# -*- coding:utf-8 -*-
'''
Created on 2020-03-10

@author: wwx371270
'''

import os, sys
import commands
import csv
import datetime
import time

route = commands.getoutput("pwd")
resultroute = route.replace("tools", "") + "result/"
now = time.strftime('%Y%m%d%H%M%S')
resultfile = resultroute + "report_" + now + ".csv"


def catchtime(fileroute):
    file_config = resultroute + str(fileroute) + "/train.sh"
    file_data = resultroute + str(fileroute) + "/train_0.log"
    cmd_config = "ls " + file_config + " --full-time|awk '{print $6,$7}'"
    cmd_data = "ls " + file_data + " --full-time|awk '{print $6,$7}'"
    time_begin = commands.getoutput(cmd_config).split(".")[0]
    time_end = commands.getoutput(cmd_data).split(".")[0]
    time_begin = datetime.datetime.strptime(time_begin, '%Y-%m-%d %H:%M:%S')
    time_end = datetime.datetime.strptime(time_end, '%Y-%m-%d %H:%M:%S')
    costtime = (time_end - time_begin).seconds
    return [costtime]


def catchconfig(fileroute):
    file_config = resultroute + str(fileroute) + "/train.sh"
    cmd = "cat " + str(file_config) + "| grep \"python3.7\""
    config = commands.getoutput(cmd).replace("--", "")
    config_list = config.split(" ")
    config_list_single = []
    for i in range(0, config_list.__len__()):
        # if "--batch_size" in config_list[i]:
        #     config_list_single.append(config_list[i].split("=")[1])
        # elif "--iterations_per_loop" in config_list[i]:
        #     config_list_single.append(config_list[i].split("=")[1])
        # elif "--max_train_steps" in config_list[i]:
        #     config_list_single.append(config_list[i].split("=")[1])
        # elif "--save_summary_steps" in config_list[i]:
        #     config_list_single.append(config_list[i].split("=")[1])
        # elif "--save_checkpoints" in config_list[i]:
        #     config_list_single.append(config_list[i].split("=")[1])
        # elif "--model_name" in config_list[i]:
        #     config_list_single.append(config_list[i].split("=")[1])
        # elif "--resnet_version" in config_list[i]:
        #     config_list_single.append(config_list[i].split("=")[1])
        if "batch_size" in config_list[i]:
            config_list_single = config_list[i] + "," + config_list[i + 1] + "," + config_list[i + 2] + "," + \
                                 config_list[i + 3] + "," + config_list[i + 4] + "," + config_list[i + 5] + "," + \
                                 config_list[i + 6] + "," + config_list[i + 7]

    return [config_list_single]


def catchdata(fileroute):
    file_data = resultroute + str(fileroute) + "/train_0.log"
    cmd = "cat " + str(file_data) + "| grep \"step: \""
    data = commands.getoutput(cmd)
    data = data.split("\n")
    return [data[-1]]


if __name__ == "__main__":
    commands.getoutput("rm -rf " + resultfile)
    fileroute = commands.getoutput("ls " + resultroute + " | grep cloud-localhost").split("\n")

    with open(resultfile, 'a+') as f:
        f_csv = csv.writer(f)
        # f_csv.writerow(
        #     ['time(s)', 'batch_size', 'iterations_per_loop', 'max_train_steps', 'save_summary_steps',
        #      'save_checkpoints_steps',
        #      'model_name', 'resnet_version', 'result'])
        f_csv.writerow(['time(s)', 'config', 'data or failfile', 'result'])

    for n in range(0, fileroute.__len__()):
        file_train0 = resultroute + str(fileroute[n]) + "/train_0.log"
        cmd = "cat " + file_train0 + " |grep \"turing train success\""
        result = commands.getoutput(cmd)
        if "turing train success" in result:
            with open(resultfile, 'a+') as f:
                row = catchtime(fileroute[n]) + catchconfig(fileroute[n]) + catchdata(fileroute[n]) + ['success']
                f_csv = csv.writer(f)
                f_csv.writerow(row)
                f.close()
        else:
            with open(resultfile, 'a+') as f:
                row = catchtime(fileroute[n]) + catchconfig(fileroute[n]) + [file_train0] + ['fail']
                f_csv = csv.writer(f)
                f_csv.writerow(row)
                f.close()
            print(file_train0 + " is in training or faild! ")
