# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(AME)

# Compile options
add_compile_options(-std=c++11)

add_definitions(-DENABLE_DVPP_INTERFACE)

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(LIB_PATH $ENV{NPU_HOST_LIB})

if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/acllib/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
include_directories(
    ${INC_PATH}/acllib/include/
    ../inc/
)

# add host lib path
link_directories(
    ${LIB_PATH}
)

add_executable(benchmark
        utils.cpp
       # dvpp_process.cpp
        model_process.cpp
       # singleOp_process.cpp
        sample_process.cpp
        main.cpp)

target_link_libraries(benchmark
        ascendcl acl_cblas acl_dvpp stdc++)

install(TARGETS benchmark DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
