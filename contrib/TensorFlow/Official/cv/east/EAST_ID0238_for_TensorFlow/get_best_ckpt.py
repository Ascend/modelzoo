# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
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

import os
import time
import subprocess
from subprocess import Popen,PIPE

def get_ckpt_name(checkpoint_path):
    with open(checkpoint_path, "r") as f:
        data = f.readlines()
        line = data[0].strip('\n')
        ckpt_name = line.split(":")[1].replace('\"',"").replace(" ","")
        print("Newest ckpt_name:",ckpt_name)
        return ckpt_name

def execute_command(cmd):
    print('start executing cmd...')
    s = subprocess.Popen(str(cmd), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    stderrinfo, stdoutinfo = s.communicate()
    #print(stdoutinfo)
    return stdoutinfo

if __name__ == '__main__':
    checkpoint_path = "./checkpoint/checkpoint"
    ckpt_name_start = ""
    f1_score_best = 0
    output_dir="./best_ckpt/"
    if not os.path.isdir(output_dir):
          os.mkdir(output_dir)
    while True:
        ckpt_name = get_ckpt_name(checkpoint_path)
        if ckpt_name != ckpt_name_start:
            ckpt_name_start = ckpt_name
            #eval_res = os.popen("bash eval.sh")
            f1_score = 0
            print("Start to eval:")
            command = "bash eval.sh"
            with Popen(command, shell=True, cwd="./", stdout=PIPE, universal_newlines=True) as process:
                for line in process.stdout:
                    line = line.strip('\n')
                    #print(line)
                    if "\"hmean\":" in line:
                        f1_score = float(line.split("\"hmean\": ")[1].split(", ")[0])
                        print("{}: {}".format(ckpt_name,line))
                        print("f1_score:",f1_score)
                        if f1_score > f1_score_best:
                            f1_score_best = f1_score
                            #os.popen("rm -rf best_ckpt/*")
                            os.popen("cp ./checkpoint/{}* {}".format(ckpt_name,output_dir))
                            f = open('best_f1_score.txt','a')
                            f.write("{}  : {}\n".format(ckpt_name,line))
                            f.close()
                            break

        print("start to wait 100s!")
        time.sleep(100)
