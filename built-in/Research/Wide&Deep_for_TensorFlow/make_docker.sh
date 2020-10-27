#!/bin/bash
ostype=`uname -m`
currentDir=$(cd "$(dirname "$0")"; pwd)
if [ x"${ostype}" = xaarch64 ];
then
   rm -rf ./docker/*
   tar -vxf Ubuntu_18.04-docker.arm64v8.tar -C ./docker
   docker images | grep ei_images_arm_base > 1.txt
   if [ -s 1.txt ] ;then
      echo 'installed'
   else
      expect ${currentDir}/exp_scp.ex root@10.136.165.4:/turingDataset/EI_docker/ei_images_arm.tar /home/run_package huawei@123
      docker import ei_images_arm.tar ei_images_arm_base:18.04
   fi
   docker build -t 'ubuntu_arm_ei:18.04' ./
   
   docker import Ubuntu_18.04-docker.arm64v8.tar ubuntu_arm:18.04
else
   docker import Ubuntu_18.04-docker.x86_64.tar ubuntu:16.04
   docker build -t ubuntu:16.04 .
fi
