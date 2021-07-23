project(CMCC_ASCEND_BENCHMARK)
add_compile_options(-std=c++11 -O3 -fPIE -fstack-protector-all)
include_directories(
$ENV{ACL_ROOT_PATH}/acllib/include
$ENV{ACL_ROOT_PATH}/driver/kernel/libc_sec/include
../
./
)
message("$ENV{ACL_LIB_PATH}")


link_directories($ENV{ACL_ROOT_PATH}/acllib/lib64)
link_directories($ENV{ACL_ROOT_PATH}/atc/lib64)

file(GLOB SRC ./*.cpp)
ADD_LIBRARY(preprocess ${SRC})
#target_link_libraries(preprocess ascendcl acl_dvpp runtime graph slog c_sec mmpa gflags pthread -O3)
target_link_libraries(preprocess ascendcl acl_dvpp runtime graph c_sec gflags pthread -O3)