#!/bin/bash
docker_image=$1
share_dir=$2
device_id=$3

echo "using docker image:$1"
echo "Mount host directory: $2"
echo "Using device to infer:$3"

if [ -z "${docker_image}" ]; then
    echo "please input docker_image"
    exit 1
fi

if [ ! -d "${share_dir}" ]; then
    echo "please input share directory that contains mxManufacture and models codes"
    exit 1
fi

if [ -z "${device_id}" ]; then
    echo "Waring: no device id specified, using default device0"
    device_id=0
fi

docker run -it \
--device=/dev/davinci${device_id} \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v ${share_dir}:${share_dir} \
${docker_image} \
/bin/bash