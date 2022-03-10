# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1) 
# project information
project(ascendcl)
# Compile options
add_compile_options(-std=c++11)
# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../outputs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "../outputs")
set(CMAKE_INSTALL_PREFIX "../../../")
set(CMAKE_OUTPUT_DIR "out")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -g -Wall")

ADD_DEFINITIONS("-DENABLE_DVPP_INTERFACE -D_GLIBCXX_USE_CXX11_ABI=0")

if(DEFINED ENV{ASCEND710_DVPP})
        add_definitions(-DASCEND710_DVPP)
        message("set ASCEND_DVPP=true")
endif()

# Header path
include_directories(
inc
#/usr/include/gflags
$ENV{install_path}/acllib/include
$ENV{install_path}/driver/kernel/libc_sec/include
/usr/include
)

# add host lib path
link_directories($ENV{install_path}/acllib/lib64/stub)
#link_directories(/usr/local/Ascend/driver/lib64)
#link_directories(/usr/local/Ascend/atc/lib64)
#link_directories(/usr/local/lib)
link_directories(../thirdpart_lib)
     
# 设置需要编译的源文件
add_executable(benchmark main.cpp util.cpp post_process.cpp infer_engine.cpp)

# 设置共享库 RC为待扩展的offline模型
#target_link_libraries(benchmark acl_dvpp ascendcl pthread protobuf cryptopp)
target_link_libraries(benchmark acl_dvpp ascendcl pthread)
    
install(TARGETS benchmark DESTINATION ${CMAKE_OUTPUT_DIR})

