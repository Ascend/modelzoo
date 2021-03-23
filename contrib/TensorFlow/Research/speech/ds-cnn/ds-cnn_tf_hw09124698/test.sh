#/bin/bash

rm -rf /var/log/npu/slog/host-0/*
rm -rf /var/log/npulog/coredump/*
rm -rf /var/log/npulog/slog/device-os-$(($DEVICE_ID<3?0:4))/*
rm -rf /var/log/npulog/slog/device-$DEVICE_ID/*
python train.py > log.txt 2>&1
cp -r /var/log/npulog/coredump ./
cp -r /var/log/npulog/slog/device-os-$(($DEVICE_ID<3?0:4)) ./
cp -r /var/log/npulog/slog/device-$DEVICE_ID ./
cp -r /var/log/npu/slog/host-0 ./
rm -rf ./kernel_meta
find ./ -exec chmod 777 {} \;