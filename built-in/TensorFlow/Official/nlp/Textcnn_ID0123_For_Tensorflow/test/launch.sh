#!/bin/bash

CMD=${@:-/bin/bash}

docker run -it \
    --shm-size=16g \
    --cap-add=SYS_PTRACE \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    --device=/dev/davinci7 \
    --device=/dev/davinci6 \
    --device=/dev/davinci5 \
    --device=/dev/davinci4 \
    --device=/dev/davinci3 \
    --device=/dev/davinci2 \
    --device=/dev/davinci1 \
    --device=/dev/davinci0 \
    -v /home:/home \
    -v /root/.pip/pip.conf:/root/.pip/pip.conf \
    -v /usr/bin/gdb:/usr/bin/gdb \
    -v /usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/common/ \
    -v /usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/driver/lib64/driver/ \
    -v /usr/local/Ascend/driver/tools/:/usr/local/Ascend/driver/tools/ \
    -v /autotest:/autotest \
    -v /usr/local/python3.7.5/lib/python3.7/site_packages/:/usr/local/python3.7.5/lib/python3.7/site_packages/ \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /var/log/npu/slog/:/var/log/npu/slog/ \
    -v /var/log/npu/profiling/:/var/log/npu/profiling/ \
    -v /var/log/npu/dump/:/var/log/npu/dump/ \
    -v /home/HwHiAiUser/plog/:/root/ascend/log/ \
    a1976acdb925 $CMD \