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
import SSHConnection

resultroute = commands.getoutput("pwd").replace("tools", "") + "/result/"
csvroute = commands.getoutput("pwd").replace("tools", "") + "/testscript/result/"


# resultroute = route.replace("result", "") + "result/"


def catchtime(fileroute, train_file):
    file_config = resultroute + str(fileroute) + "/train.sh"
    file_data = commands.getoutput("ls " + train_file)
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
    config_list = commands.getoutput(cmd).split(" ")
    config = " "
    for n in range(0, config_list.__len__()):
        if "--" in config_list[n]:
            config += config_list[n]
            config += " "
    return [config]


def catchgpuconfig(fileroute):
    file_config = resultroute + str(fileroute) + "/train.sh"
    cmd = "cat " + str(file_config) + "| grep \"python3.7\""
    config_list = commands.getoutput(cmd).split(" ")
    config = " "
    for n in range(0, config_list.__len__()):
        if "--" in config_list[n] and "config_file" not in config_list[n] and "dir" not in config_list[
            n] and "data_url" not in config_list[n] and "DEVICE_ID" not in config_list[
            n]:
            config += config_list[n]
            config += " "
    return [config]


def catchfileflag(train_file):
    fileflag = 0
    file_data = commands.getoutput("ls " + train_file)
    cmd1 = "cat " + str(file_data) + "| grep \"resnet50_train\""
    cmd2 = "cat " + str(file_data) + "| grep \"VGG16_TF\""
    cmd3 = "cat " + str(file_data) + "| grep \"bert-nv\""
    cmd4 = "cat " + str(file_data) + "| grep \"RN50v1.5\""
    cmd5 = "cat " + str(file_data) + "| grep \"Alexnet_TF\""
    cmd6 = "cat " + str(file_data) + "| grep \"bert-nz-npu\""

    if commands.getoutput(cmd1) != "":
        fileflag = 1
    elif commands.getoutput(cmd2) != "":
        fileflag = 2
    elif commands.getoutput(cmd3) != "":
        fileflag = 3
    elif commands.getoutput(cmd4) != "":
        fileflag = 4
    elif commands.getoutput(cmd5) != "":
        fileflag = 5
    elif commands.getoutput(cmd6) != "":
        fileflag = 6
    return fileflag


# 获取训练数据
def catchdata(train_file):
    file_data = commands.getoutput("ls " + train_file)
    cmd = ""
    # 判断是哪种训练数据
    if catchfileflag(train_file) == 1:
        cmd = "cat " + str(file_data) + "| grep \"step: \""
    elif catchfileflag(train_file) == 2:
        cmd = "cat " + str(file_data) + "| grep \"total images/sec:\""
    elif catchfileflag(train_file) == 3:
        cmd = "cat " + str(file_data) + "| grep \"Step =\""
    elif catchfileflag(train_file) == 4:
        cmd = "cat " + str(file_data) + "| grep \"total_loss:\""
    elif catchfileflag(train_file) == 5:
        cmd = "cat " + str(file_data) + "| grep \"total images/sec:\""
    elif catchfileflag(train_file) == 6:
        cmd = "cat " + str(file_data) + "| grep \"Loss for final step:\""
    data = commands.getoutput(cmd)
    data = data.split("\n")
    return [data[-1]]


def catchlossdata(train_file):
    data = catchdata(train_file)[0].split(" ")
    loss = 0.0
    # 判断是哪种训练数据
    if catchfileflag(train_file) == 1:
        for i in range(0, data.__len__()):
            loss = data[-3].replace("total_loss:", "")
            loss = float(loss)
    if catchfileflag(train_file) == 3:
        for i in range(0, data.__len__()):
            if "Average" in data[i]:
                loss = float(data[i - 1].strip())
    if catchfileflag(train_file) == 4:
        loss = float(data[- 1].strip())
    if catchfileflag(train_file) == 6:
        loss = float(data[-1][:-1])
    return loss


def catchimagessecdata(train_file):
    data = catchdata(train_file)[0].split(" ")
    for i in range(0, data.__len__()):
        images_sec = float(data[-1])
    return images_sec


def catchhosterror(fileroute):
    file_route = "/var/log/npu/slog/container/" + fileroute + "/host-0/"
    cmd = "cd " + file_route + " && grep ERROR *"
    result = commands.getoutput(cmd)
    return result


def transfercsv(transfer_resultfile):
    transferroute = "/home/Fullrange/report/npusend/"
    excuteroute = "/home/Fullrange/Competition/"
    env = {
        'host': str(sys.argv[3]), 'username': 'root', 'password': str(sys.argv[4])
    }
    try:
        ssh = SSHConnection.SSHConnection(env)
    except:
        print("[ERROR]connect to env fail")
    else:
        process = ssh.exec_command("ps -ef | grep alluse.sh | grep -v grep")
        transfercmd = "cd /autotest/CI_daily && expect exp_scp.ex " + transfer_resultfile + " " + env[
            'username'] + "@" + env['host'] + ":" + transferroute + " " + env['password']
        print(transfercmd)
        if process == None or 'alluse.sh' not in process:
            ssh.exec_command("rm -rf " + transferroute + "*.csv")
            commands.getoutput("dos2unix " + transfer_resultfile)
            commands.getoutput(transfercmd)
            # 拉起GPU
            ssh.exec_command("bash " + excuteroute + "docker1.sh")
            commands.getoutput("rm -rf " + transfer_resultfile)
        if ssh:
            ssh.close()


def catchtotaldata():
    now = time.strftime('%Y%m%d%H%M%S')
    resultfile = csvroute + "report_total_" + now + ".csv"
    fileroute = commands.getoutput("ls " + resultroute + " | grep cloud-localhost").split("\n")

    with open(resultfile, 'a+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['time(s)', 'config', 'data or failfile', 'result'])
        f.close()

    for n in range(0, fileroute.__len__()):
        file_train = resultroute + str(fileroute[n]) + "/train_*.log"
        file_train = commands.getoutput("ls " + file_train).split("\n")
        for i in range(0, file_train.__len__()):
            fileflag = catchfileflag(file_train[i])
            cmd = "cat " + file_train[i] + " |grep \"turing train success\""
            result = commands.getoutput(cmd)
            if "turing train success" in result:
                # 判断是哪种训练数据
                # resnet50_HC
                if fileflag == 1:
                    if catchlossdata(file_train[i]) < 1000.0:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['success']
                    else:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['fail']
                    with open(resultfile, 'a+') as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow(row)
                        f.close()
                # vgg16
                elif fileflag == 2:
                    if catchimagessecdata(file_train[i]) > 500.0:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['success']
                    else:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['fail']
                    with open(resultfile, 'a+') as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow(row)
                        f.close()
                # bert_NV
                elif fileflag == 3:
                    if catchlossdata(file_train[i]) < 15.0:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['success']
                    else:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['fail']
                    with open(resultfile, 'a+') as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow(row)
                        f.close()
                # resnet50_NV
                elif fileflag == 4:
                    if catchlossdata(file_train[i]) < 1000.0:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['success']
                    else:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['fail']
                    with open(resultfile, 'a+') as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow(row)
                        f.close()
                # alexnet
                elif fileflag == 5:
                    if catchimagessecdata(file_train[i]) > 500.0:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['success']
                    else:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['fail']
                    with open(resultfile, 'a+') as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow(row)
                        f.close()
                # bert_NZ
                elif fileflag == 6:
                    if catchlossdata(file_train[i]) < 1000.0:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['success']
                    else:
                        row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + catchdata(
                            file_train[i]) + ['fail']
                    with open(resultfile, 'a+') as f:
                        f_csv = csv.writer(f)
                        f_csv.writerow(row)
                        f.close()


            else:
                row = catchtime(fileroute[n], file_train[i]) + catchconfig(fileroute[n]) + [file_train[i]] + [
                    'fail']
                with open(resultfile, 'a+') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(row)
                    f.close()


def catchsingledata():
    resultfile = csvroute + str(sys.argv[1]) + "_report.csv"
    transfer_resultfile = csvroute + str(sys.argv[1]) + "_transfer_report.csv"
    fileroute = sys.argv[2]
    ip = commands.getoutput(
        "ifconfig -a|grep inet|grep -v inet6|grep -v 192.168|awk '{print $2}'|tr -d addr: | head -n 2 | tail -n 1")

    if os.path.exists(resultfile) == False:
        with open(resultfile, 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['time(s)', 'IP', 'model_name', 'config', 'gpu_config', 'path', 'data', 'NPU_result'])
            f.close()

    if os.path.exists(transfer_resultfile) == False:
        with open(transfer_resultfile, 'a+') as f2:
            f_csv_fail = csv.writer(f2)
            f_csv_fail.writerow(['time(s)', 'IP', 'model_name', 'config', 'gpu_config', 'path', 'data', 'NPU_result'])
            f2.close()

    file_train = resultroute + str(fileroute) + "/train_*.log"
    file_train = commands.getoutput("ls " + file_train).split("\n")
    for i in range(0, file_train.__len__()):
        fileflag = catchfileflag(file_train[i])
        cmd = "cat " + file_train[i] + " |grep \"turing train success\""
        result = commands.getoutput(cmd)
        model_name = []
        if fileflag == 1:
            model_name.append("Resnet50_HC")
        elif fileflag == 2:
            model_name.append("VGG16_TF")
        elif fileflag == 3:
            if "base12" in str(sys.argv[1]):
                model_name.append("Bert_NV_base12")
            if "base6" in str(sys.argv[1]):
                model_name.append("Bert_NV_base6")
            if "large" in str(sys.argv[1]):
                model_name.append("Bert_NV_large")
        elif fileflag == 4:
            model_name.append("Resnet50_NV")
        elif fileflag == 5:
            model_name.append("Alexnet_TF")
        elif fileflag == 6:
            model_name.append("Bert_NZ")
        if "turing train success" in result:
            # 判断是哪种训练数据
            # resnet50_HC
            if fileflag == 1:
                if catchlossdata(file_train[i]) < 1000.0:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['success']
                else:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['fail']
                with open(resultfile, 'a+') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(row1)
                    f.close()
                with open(transfer_resultfile, 'a+') as f2:
                    f_csv_fail = csv.writer(f2)
                    f_csv_fail.writerow(row1)
                    f2.close()
                # transfer report
                transfercsv(transfer_resultfile)
            # vgg16
            elif fileflag == 2:
                if catchimagessecdata(file_train[i]) > 500.0:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['success']
                else:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['fail']
                with open(resultfile, 'a+') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(row1)
                    f.close()
                with open(transfer_resultfile, 'a+') as f2:
                    f_csv_fail = csv.writer(f2)
                    f_csv_fail.writerow(row1)
                    f2.close()
                # transfer report
                transfercsv(transfer_resultfile)
            # bert
            elif fileflag == 3:
                if catchlossdata(file_train[i]) < 15.0:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['success']
                else:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['fail']
                with open(resultfile, 'a+') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(row1)
                    f.close()
                with open(transfer_resultfile, 'a+') as f2:
                    f_csv_fail = csv.writer(f2)
                    f_csv_fail.writerow(row1)
                    f2.close()
                # transfer report
                transfercsv(transfer_resultfile)
            # resnet50_NV
            elif fileflag == 4:
                if catchlossdata(file_train[i]) < 1000.0:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['success']
                else:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['fail']
                with open(resultfile, 'a+') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(row1)
                    f.close()
                with open(transfer_resultfile, 'a+') as f2:
                    f_csv_fail = csv.writer(f2)
                    f_csv_fail.writerow(row1)
                    f2.close()
                # transfer report
                transfercsv(transfer_resultfile)
            # alexnet
            elif fileflag == 5:
                if catchimagessecdata(file_train[i]) > 500.0:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['success']
                else:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['fail']
                with open(resultfile, 'a+') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(row1)
                    f.close()
                with open(transfer_resultfile, 'a+') as f2:
                    f_csv_fail = csv.writer(f2)
                    f_csv_fail.writerow(row1)
                    f2.close()
                # transfer report
                transfercsv(transfer_resultfile)
            # bert_NZ
            if fileflag == 6:
                if catchlossdata(file_train[i]) < 1000.0:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['success']
                else:
                    row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                        fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + catchdata(
                        file_train[i]) + ['fail']
                with open(resultfile, 'a+') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(row1)
                    f.close()
                with open(transfer_resultfile, 'a+') as f2:
                    f_csv_fail = csv.writer(f2)
                    f_csv_fail.writerow(row1)
                    f2.close()
                # transfer report
                transfercsv(transfer_resultfile)

        else:
            row1 = catchtime(fileroute, file_train[i]) + [ip] + model_name + catchconfig(
                fileroute) + catchgpuconfig(fileroute) + [resultroute + str(fileroute)] + ['NA'] + ['fail']
            with open(resultfile, 'a+') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(row1)
                f.close()
            with open(transfer_resultfile, 'a+') as f2:
                f_csv_fail = csv.writer(f2)
                f_csv_fail.writerow(row1)
                f2.close()
            # transfer report
            transfercsv(transfer_resultfile)
            print(file_train[i] + " has faild! ")


if __name__ == "__main__":
    # catchtotaldata()
    catchsingledata()
