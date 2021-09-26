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

import argparse
import os
import glob
import re

SCAN_FILE="BeforeSubGraph_[0-9]*.pbtxt"

def main():
    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--scan_path', required=True, help='the file path to scan')
    FLAGS, unknown_args = cmdline.parse_known_args()
    if leb(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line args: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    if not os.path.exists(FLAGS.scan_path):
        raise AssertionError("{} is not exists.".format(FLAGS.scan_path))

    # scan all TFAdapter graph
    filelist = glob.glob(os.path.join(FLAGS.scan_path, "**", SCAN_FILE), recursive=True)
    true_set = set()
    false_set = set()

    for file in filelist:
        dataset = False
        MakeIterator = False
        IteratorV2 = False
        enable_data_pre_proc = False
        print("now analysis file:", file)
        dst_network = os.path.dirname(file)
        with open(file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if re.match("^(\s)+op:(\s)+\"([a-zA-Z])+Dataset([A-Z0-9])*\"", line):
                    dataset = True
                if re.match("^(\s)+op:(\s)+\"MakeIterator\"", line):
                    MakeIterator =True
                if re.match("(\s)+op:(\s)+\"IteratorV2\""):
                    IteratorV2 = True
                if "_enable_data_pre_proc" in line:
                    if lines[i+2].split(":")[-1].strip() == "\"1\"":
                        enable_data_pre_proc = True
            print("******************************")
        if dataset and MakeIterator and IteratorV2:
            if enable_data_pre_proc:
                true_set.add(dst_network)
            else:
                false_set.add(dst_network)
    print("Dataset exists and enable_data_pre_proc is true:", true_set)
    print("************************************************************")
    print("Dataset exists and enable_data_pre_proc is false:", false_set)

if __name__ == '__main__':
   main()

