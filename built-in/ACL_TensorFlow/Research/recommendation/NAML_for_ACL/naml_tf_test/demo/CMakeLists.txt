# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(test_acl_resnet)

# Compile options
add_compile_options(-std=c++11)

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../outputs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "../../outputs")
set(CMAKE_INSTALL_PREFIX "../../../run")
set(CMAKE_OUTPUT_DIR "out")
ADD_DEFINITIONS("-DENABLE_DVPP_INTERFACE")

#set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# Header path
include_directories(
../inc/
/usr/local/Ascend/acllib/include
#/usr/local/Ascend/driver/kernel/libc_sec/include
)

# add host lib path
link_directories(/usr/local/Ascend/acllib/lib64/stub)
#link_directories(/usr/local/Ascend/driver/lib64)
#link_directories(/usr/local/Ascend/atc/lib64)

add_executable(main test_acl.cpp)

#target_link_libraries(main c_sec slog acl_dvpp acl_cblas ascendcl pthread)
target_link_libraries(main acl_dvpp acl_cblas ascendcl pthread)
install(TARGETS main DESTINATION ${CMAKE_OUTPUT_DIR})


