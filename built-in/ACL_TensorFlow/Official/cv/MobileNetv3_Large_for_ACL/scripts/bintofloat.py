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

import os
import numpy as np
import sys

def bin2float(file, dtype):
    if dtype=="fp32":
        data = np.fromfile(file, dtype='float32')
    elif dtype=="fp16":
        data = np.fromfile(file, dtype='float16')
    elif dtype=="int32":
        data = np.fromfile(file, dtype=np.int32)
    elif dtype=="int8":
        data = np.fromfile(file, dtype=np.int8)
    else:
        print("unaccepted type")
    float_file = file + ".txt"
    print("save to file "+ float_file)
    np.savetxt(float_file, data.reshape(-1, 1), fmt="%.6f")


def bintofloat(filename, dtype):
    if os.path.isdir(filename):
        for file in os.listdir(filename):
            if(file != "." or file != ".."):
                bin2float(filename+file, dtype)
    else:
        bin2float(filename, dtype)


if __name__ == "__main__":
    print("param: "+sys.argv[1]+","+sys.argv[2])
    bintofloat(sys.argv[1], sys.argv[2])
