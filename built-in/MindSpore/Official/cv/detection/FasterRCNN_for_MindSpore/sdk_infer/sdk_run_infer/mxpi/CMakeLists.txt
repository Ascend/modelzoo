cmake_minimum_required(VERSION 3.5.2)
project(msfasterrcnnpostprocessor)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(PLUGIN_NAME "msfasterrcnnpostprocessor")
set(TARGET_LIBRARY ${PLUGIN_NAME})
set(ASCEND_DIR $ENV{ASCEND_TOOLKIT}/latest/acllib)
get_filename_component(SDK_INSTALL_DIR ${PROJECT_SOURCE_DIR}/../../.. ABSOLUTE)


include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(${SDK_INSTALL_DIR}/include)
include_directories(${SDK_INSTALL_DIR}/opensource/include)
include_directories(${SDK_INSTALL_DIR}/opensource/include/gstreamer-1.0)
include_directories(${SDK_INSTALL_DIR}/opensource/include/glib-2.0)
include_directories(${SDK_INSTALL_DIR}/opensource/lib/glib-2.0/include)

link_directories(${SDK_INSTALL_DIR}/lib)
link_directories(${SDK_INSTALL_DIR}/opensource/lib/)


add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")
add_compile_options(-Dgoogle=mindxsdk_private)

add_definitions(-DENABLE_DVPP_INTERFACE)

#message("ASCEND_DIR:${ASCEND_DIR}.")
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${ASCEND_DIR}/include)

add_library(${TARGET_LIBRARY} SHARED MSFasterRcnnPostProcess.cpp MxpiMSFasterRcnnPostProcessor.cpp)

target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0)
target_link_libraries(${TARGET_LIBRARY} plugintoolkit mxpidatatype mxbase)
target_link_libraries(${TARGET_LIBRARY} -Wl,-z,relro,-z,now,-z,noexecstack -s)

install(TARGETS ${TARGET_LIBRARY} LIBRARY DESTINATION ${SDK_INSTALL_DIR}/lib/)

