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
# coding=utf-8
import os
import argparse
import datetime
import moxing as mox

## Code dir: /home/work/user-job-dir/code
## Work dir: /home/work/workspace/device2
print("===>>>{}".format(os.getcwd()))
print(os.system('env'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="../output")
    parser.add_argument("--data_url", type=str, default="../dataset")
    config = parser.parse_args()

    # copy dataset from obs to local
    # dataset will be saved under /cache/ilsvrc2012_tfrecord while the results will be saved under /cache/results
    local_dir = '/cache/ilsvrc2012_tfrecord'
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to local dir:{}".format(config.data_url, local_dir))
    mox.file.copy_parallel(src_url=config.data_url, dst_url=local_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to local, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(local_dir)
    print("===>>>Files number:", len(files))

    #  run training
    print("===>>>Begin training:")
    os.system('bash /home/work/user-job-dir/code/run_1p.sh')
    print("===>>>Training finished:")

    #  copy results from local to obs
    local_dir = '/cache/result'
    remote_dir = os.path.join(config.train_url, 'result')
    if not mox.file.exists(remote_dir):
        mox.file.make_dirs(remote_dir)
    start = datetime.datetime.now()
    print("===>>>Copy files from local dir:{} to obs:{}".format(local_dir, remote_dir))
    mox.file.copy_parallel(src_url=local_dir, dst_url=remote_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from local to obs, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(local_dir)
    print("===>>>Files number:", len(files))

