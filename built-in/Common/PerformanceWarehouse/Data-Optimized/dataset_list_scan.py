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
    cmdline.add_argument('--scan_path', required=True,
                         help="""db file path.""")
    FLAGS, unknown_args = cmdline.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    if not os.path.exists(FLAGS.scan_path):
        raise AssertionError("{} is not exists.".format(FLAGS.scan_path))

    print("[INFO] [dataset_list_scan.py] start-----------------------------------------")
    # scan all gpu_info.csv
    # filelist_B = glob.glob(os.path.join(FLAGS.scan_path, "**", SCAN_FILE), recursive=True)
    filelist_B = list(os.popen('find %s -name "BeforeSubGraph*.pbtxt"' % FLAGS.scan_path))
    filelist_B = [file.strip() for file in filelist_B if ".pbtxt" in file]
    exist_dataset = False
    exist_prefetch = False
    exist_batch = False

    for file in filelist_B:
        with open(file, "r") as f:
            lines = f.readlines()
            dataset = 0
            PrefetchDataset = 0
            for i, line in enumerate(lines):
                if re.match("^(\s)+op:(\s)+\"([a-zA-Z])+Dataset([A-Z0-9])*\"", line):
                    dataset += 1
                    exist_dataset = True
                    if re.match("^(\s)+op:(\s)+\"PrefetchDataset\"", line):
                        PrefetchDataset += 1
                        exist_prefetch = True
                if re.match("^(\s)+op:(\s)+\"([a-zA-Z])+Batch([A-Z0-9])*\"", line):
                    exist_batch = True

            if dataset > 0 and PrefetchDataset == 0:
                print("[INFO] [dataset_list_scan.py] %s does not contain Prefetch()." % file)
            elif dataset > 10:
                print("[INFO] [dataset_list_scan.py] %s contain too much Methods of Dataset." % file)
            else:
                print("[INFO] [dataset_list_scan.py] [%s] datasetAPI num: %d , PrefetchDataset num: %d." % (file, dataset, PrefetchDataset))

    if exist_dataset:
        print("[INFO] [dataset_list_scan.py] network use tf.data.*dataset().")
    elif exist_batch:
        print("[INFO] [dataset_list_scan.py] network use tf.train.*batch(),please check num_threads.")
    else:
        print("[INFO] [dataset_list_scan.py] network might not use tf API to prepare dataset.")
    print("[INFO] [dataset_list_scan.py] end----------------------------------------")

if __name__ == "__main__":
    main()
