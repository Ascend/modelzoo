#!/bin/bash

#½Å±¾±¾ÉíÂ·¾¶
scriptpath=$(cd "$(dirname "$0")"; pwd)

path=/autotest/job_profiling
profiling_path=/var/log/npu/profiling


testcase=$1


mkdir -p ${path}/${testcase}/data
mkdir -p ${path}/${testcase}/result
mkdir -p ${path}/${testcase}/tmp
rm -rf ${scriptpath}/analysis_performance* ${scriptpath}/*file
cp -r ${profiling_path}/JOB* ${path}/${testcase}/data


python3 performanceanalysis.py


cp -r ${scriptpath}/analysis_performance* ${path}/${testcase}/result
cp -r ${scriptpath}/*file  ${path}/${testcase}/tmp
