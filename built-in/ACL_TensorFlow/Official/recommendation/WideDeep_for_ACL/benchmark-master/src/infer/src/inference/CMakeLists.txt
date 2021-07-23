project(CMCC_ASCEND_BENCHMARK)
add_compile_options(-std=c++11 -O3 -fPIE -fstack-protector-all)
include_directories(
$ENV{ACL_ROOT_PATH}/acllib/include
$ENV{ACL_ROOT_PATH}/driver/kernel/libc_sec/include
../ 
./
)

link_directories($ENV{ACL_ROOT_PATH}/acllib/lib64)
link_directories($ENV{ACL_ROOT_PATH}/atc/lib64)

ADD_LIBRARY(inference inference.cpp)
#target_link_libraries(inference ascendcl runtime graph slog c_sec mmpa gflags pthread  -O3)
target_link_libraries(inference ascendcl runtime graph c_sec gflags pthread  -O3)
