# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
import sys

module_file_path = "xt/util/config.py"
config_file_path = "xt/framework/default_config.py"


def put_config(config, module):
    for root, subdir, filenames in os.walk("xt/" + module):
        for dir_name in subdir:
            dir_path = os.path.join(root, dir_name)
            files = os.listdir(dir_path)
            list_file = []
            for file in files:
                if os.path.isfile(os.path.join(dir_path, file)):
                    split_file = file.split('.')
                    if split_file[1] == "py" and split_file[0] != "__init__":
                        file = split_file[0]
                        list_file.append(file)
            config.update({str(dir_name): list_file})
        break


def create_module_dir_file(config, module_name):
    with open(module_file_path, 'w+') as fw:
        module_num = 0
        for module in module_name:
            fw.write(str(module) + " = ")
            fw.write(str(config[module_num]) + "\n")
            module_num += 1
        fw.close()


def modify_config_file(config_file, status):
    with open(config_file, 'w+') as fw:
        fw.write(str("LIB = ") + str(status) + "\n")
        fw.write('ACTOR_FILE = "xt/framework/act_launcher.py"')
        fw.close()


def main():
    config = [{}, {}, {}, {}]
    module_name = ['alg', 'agent', 'model', 'env']
    config_index = 0
    for module in module_name:
        put_config(config[config_index], module)
        config_index += 1

    create_module_dir_file(config, module_name)
    status = True
    modify_config_file(config_file_path, status)

    if "cython" not in sys.argv:
        if sys.version_info.major == 3:
            setup_cmd = "sudo python3 setup.py"
        else:
            setup_cmd = "sudo python2 setup.py "
        setup_cmd += " install "
    else:
        setup_cmd = "sudo python3 setup_cython.py install"

    print("execute setup command: %s" % setup_cmd)
    os.system(setup_cmd)

    status = False
    modify_config_file(config_file_path, status)
    os.remove(module_file_path)


if __name__ == "__main__":
    main()
