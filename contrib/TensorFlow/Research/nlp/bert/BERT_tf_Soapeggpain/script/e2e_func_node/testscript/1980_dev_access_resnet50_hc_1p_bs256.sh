#!/bin/bash
export LANG=en_US.UTF-8
DEVICE_NUM=1
stepnum=10000

currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}
currtime=`date +%Y%m%d%H%M%S`
mkdir -p ${currentDir}/result

# source public.lib
. ${currentDir}/../lib/public.lib

# user testcase
casecsv="case_resnet50_HC.csv"
casenum=1

if [ "$1" =~ "only-eval"];
then
    casenum=51
fi
echo "casenum is ${casenum}"

# docker or host
exectype="host"

ostype=`uname -m`
if [ x"${ostype}" = xaarch64 ];
then
    # arm
    dockerImage="ubuntu_arm:18.04"
else
    # x86
    dockerImage="ubuntu:16.04"
fi

scriptname=`echo $(basename "$0") | awk -F"." '{print $1}'`
echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] current dir : ${currentDir}"

env_init info

${currentDir}/../e2e_test.sh ${casecsv} ${casenum} ${exectype} ${dockerImage} | tee ${currentDir}/result/${scriptname}.log

output_dir=`cat ${currentDir}/result/${scriptname}.log | grep "host mount dir" | awk '{print $6}'`
device_list=`cat ${currentDir}/result/${scriptname}.log | grep "\[INFO\] DEVICE_ID" | awk -F"=" '{print $2}'`
analysis_profiling 1 >> ${currentDir}/result/${scriptname}.log

cd $output_dir
grep FPS: train_0.log | cut -c 31-37 > ${currentDir}/FPS.log
python3.7 ${currentDir}/../tools/result_check.py 10000 100 ${currentDir} >> ${currentDir}/result/${scriptname}.log
python3.7 ${currentDir}/../tools/analysis_check_excel_1p.py 100 ${currentDir}/profiling/${scriptname} ${scriptname} ${currentDir}
Average=`grep "max_FPS" ${currentDir}/result/${scriptname}.log | awk -F ',' '{print $1}'|cut -c 9-20 `
echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] Average: ${Average}" >> ${currentDir}/result/${scriptname}.log
Base_FPS=1790
FPS_check=0
if [ $Average -gt $Base_FPS ];
then
   echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] ${scriptname} FPS check success" >> ${currentDir}/result/${scriptname}.log
   FPS_check=1
else
   echo "[`date +%Y%m%d-%H:%M:%S`] [ERROR] ${scriptname} FPS check fail" >> ${currentDir}/result/${scriptname}.log
   FPS_check=0
fi

fluctuation=`grep "max_FPS" ${currentDir}/result/${scriptname}.log | awk -F ',' '{print $4}'|cut -c 13-20 `
fluctuation_base=20
fluctuation_check=0
if [ $fluctuation -le $fluctuation_base ];
then
   echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] ${scriptname} FPS check fluctuation success" >> ${currentDir}/result/${scriptname}.log
   fluctuation_check=1
else
   echo "[`date +%Y%m%d-%H:%M:%S`] ERROR] ${scriptname} FPS check fluctuation  fail" >> ${currentDir}/result/${scriptname}.log
   fluctuation_check=0
fi


result_check=`grep  "turing train success" $output_dir/train_0.log |wc -l `
analysis_profiling_check=`grep "some profiling data is null" ${currentDir}/result/${scriptname}.log |wc -l`
let retcode=FPS_check+fluctuation_check+result_check
if [ "${retcode}" -eq 3 ] && [ $analysis_profiling_check -eq 0 ];
then
    echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] ${scriptname} return success" >> ${currentDir}/result/${scriptname}.log
	echo [${scriptname}]:success >>${currentDir}/result.txt
else
	echo "[`date +%Y%m%d-%H:%M:%S`] [ERROR] ${scriptname} return fail" >> ${currentDir}/result/${scriptname}.log
	echo [${scriptname}]:fail >>${currentDir}/result.txt
fi

