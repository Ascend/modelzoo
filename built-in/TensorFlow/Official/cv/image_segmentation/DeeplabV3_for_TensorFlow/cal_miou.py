# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--result_log', default="",
                         help="""result log file.""")
args = parser.parse()

with open(args.result_log, 'r') as f:

    lines = f.readlines()
    eval_res = [l.strip('\n') for l in lines if l.startswith('eval/miou')]

    num_ckpt = len(eval_res)//21
    # print("{} checkpoints existed ".format(num_ckpt))
    for i in range(0,num_ckpt):
        miou_tot = 0
        for j in range(0,21):
            index = i * 21 + j
            iou_val = float(eval_res[index].split(':')[-1].strip())

            miou_tot += iou_val
        miou = miou_tot/21.0
        print("mean IOU is :",str(miou))




    
