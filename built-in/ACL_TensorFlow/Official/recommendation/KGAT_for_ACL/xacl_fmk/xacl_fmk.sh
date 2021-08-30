#!/bin/bash
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

# Created by wang-bain on 2021/3/18.

build(){
    mkdir -p out && mkdir -p build && cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE && make
}

clean(){
    rm -rf out && rm -rf build
}

main(){
    action=$1
    if [ "A${action}" == "A" ]; then
        clean
        build
    elif [ "A${action}" == "Abuild" ]; then
        clean
        build
    else
        clean
    fi
}

main $@
