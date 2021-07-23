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


# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(XACL)

# Compile options
add_compile_options(-std=c++11)

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

if (NOT DEFINED ENV{RUN_PATH})
    set(RUN_PATH "/usr/local/Ascend")
else ()
    set(RUN_PATH $ENV{RUN_PATH})
endif ()

message(STATUS "RUN_PATH: ${RUN_PATH}")

# Header path
include_directories(${RUN_PATH}/acllib/include inc)

# add host lib path
link_directories(${RUN_PATH}/acllib/lib64/stub/)

SET(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath-link,$ENV{NPU_HOST_LIB}")

add_executable(xacl_fmk src/utils.cpp src/main.cpp)

target_link_libraries(xacl_fmk ascendcl stdc++)

install(TARGETS xacl_fmk DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
