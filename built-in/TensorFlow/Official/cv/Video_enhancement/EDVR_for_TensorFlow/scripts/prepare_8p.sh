#!/bin/bash
cur_dir=`pwd`
root_dir=${cur_dir}

for i in $(seq 0 7)
do
    if [ ! -d "D$i" ];then
        mkdir ${root_dir}/D$i
        cd ${root_dir}/D$i
        ln -s ../ascendvsr ascendvsr
    	ln -s ../ascendcv ascendcv
        ln -s ../tools tools
        ln -s ../configs configs
        ln -s ../data data
	ln -s ../scripts scripts
    fi
done

