#!/bin/bash

docker_image=ssd_ms:v1.0
data_dir=/data/dataset/coco2017
ssd_code_dir=/home/sam/codes/
pretrained_models_dir=/home/sam/pretrained_models/

docker run -it --ipc=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
--privileged \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v ${ssd_code_dir}:${ssd_code_dir} \
-v ${data_dir}:${data_dir} \
-v ${pretrained_models_dir}:${pretrained_models_dir} \
-v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
-v /var/log/npu/slog/:/var/log/npu/slog/ \
-v /var/log/npu/profiling:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump/ \
-v /var/log/npu/:/usr/slog \
${docker_image} \
/bin/bash