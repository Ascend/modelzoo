# Copyright 2019 Huawei Technologies Co., Ltd
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

import os

cwd = os.getcwd()  # currrent path
os.makedirs("save_model",exist_ok=True)

os.makedirs("save_testbpd",exist_ok=True)
save_path = 'save_model/params_cifar.ckpt'
osr_dir=os.path.join(cwd , 'save_model')
osr=os.path.join(cwd , save_path)
# saver.save(sess, osr)
# TODO up to s3
import boto3
ACCESS_KEY = '7N2JK6JLDLW3DE3ESNWV'
SECRET_KEY = 'cJfnmOiPNe3PSwqyadPMkPr7wyj4ltl6Ao8E6SRY'
s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    endpoint_url='https://obs.cn-north-4.myhuaweicloud.com'
)
os.chdir(osr_dir)
all_files = os.listdir(osr_dir)
for ff in all_files:
    if not os.path.isfile(os.path.abspath(ff)):

        all_files.remove(ff)
    s3.upload_file(os.path.abspath(ff), 'realnvp', 'oo/tfcheck')

save_nppath = './save_testbpd/test_bpd_cifar.npz'
osd = os.path.join(cwd , save_nppath)
osd_dir = os.path.join(cwd , './save_testbpd/')
