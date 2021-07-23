cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11 -O3 -D ENABLE_DVPP_INTERFACE -fPIE -fstack-protector-all)
project(CMCC_ASCEND_BENCHMARK)


include_directories(
#目前需要手工放
$ENV{ACL_ROOT_PATH}/acllib/include
$ENV{ACL_ROOT_PATH}/driver/kernel/libc_sec/include
./
data_input
inference
postprocess
preprocess
../../../opensource/gflags/output/include
)

add_definitions(-DENABLE_ACL_DVPP_INTERFACE)
ADD_SUBDIRECTORY(data_input)
ADD_SUBDIRECTORY(inference)
ADD_SUBDIRECTORY(preprocess)
ADD_SUBDIRECTORY(postprocess)

link_directories($ENV{ACL_ROOT_PATH}/acllib/lib64)
link_directories($ENV{ACL_ROOT_PATH}/atc/lib64)
link_directories($ENV{ACL_ROOT_PATH}/driver/lib64)


add_executable(benchmark main.cpp)
#target_link_libraries(benchmark data_input inference postprocess preprocess ascendcl runtime graph c_sec slog mmpa gflags pthread -Wl,-z,relro,-z,now,-z,noexecstack -pie -O3)
target_link_libraries(benchmark data_input inference postprocess preprocess ascendcl runtime graph c_sec gflags pthread -Wl,-z,relro,-z,now,-z,noexecstack -pie -O3)
