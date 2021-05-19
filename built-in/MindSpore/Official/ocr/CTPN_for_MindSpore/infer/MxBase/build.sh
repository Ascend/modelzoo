#!/bin/bash

path_cur=$(dirname $0)

function check_env() {
    if [ ! "${ASCEND_VERSION}" ]; then
      export ASCEND_VERSION=nnrt/latest
      echo "Set ASCEND_VERSION to the default value: ${ASCEND_VERSION}"
    else
      echo "ASCEND_VERSION is set to ${ASCEND_VERSION} by user"
    fi

    if [ ! "${ARCH_PATTERN}" ]; then
      export ARCH_PATTERN=./
      echo "ARCH_PATTERN is set to the default value: ${ARCH_PATTERN}"
    else
      echo "ARCH_PATTERN is set to ${ARCH_PATTERN} by user"
    fi
}

function build_ctpn() {
    cd $path_cur || exit
    rm -rf build
    mkdir -p build
    cd build || exit
    cmake ..
    make
    ret=$?
    if [ ${ret} -ne 0 ]; then
      echo "Failed to build crnn."
      exit ${ret}
    fi
    make install
}

check_env
build_ctpn

