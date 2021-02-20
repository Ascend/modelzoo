#!/bin/bash

# Copyright(C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/build.sh"; exit ;} ; pwd)

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

export MX_SDK_HOME=${CUR_PATH}/../../..
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:${ASCEND_TOOLKIT_HOME}/acllib/lib64:${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins

# compile
if g++ main.cpp -I ${MX_SDK_HOME}/include/ \
             -I ${MX_SDK_HOME}/opensource/include/ \
             -I ${MX_SDK_HOME}/opensource/include/opencv4 \
             -L ${MX_SDK_HOME}/lib/ \
             -L ${MX_SDK_HOME}/opensource/lib/ \
             -L ${MX_SDK_HOME}/opensource/lib64/ \
             -std=c++11 \
             -D_GLIBCXX_USE_CXX11_ABI=0 \
             -fPIC -fstack-protector-all \
             -g -Wl,-z,relro,-z,now,-z,noexecstack -pie -O2 -Wall -lglog \
             -lmxbase -lstreammanager -lopencv_world \
             -o main;
then
  info "Build successfully."
else
  warn "Build failed."
fi
exit 0
