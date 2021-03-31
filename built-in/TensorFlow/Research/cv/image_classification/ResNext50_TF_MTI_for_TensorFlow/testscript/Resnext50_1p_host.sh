#!/bin/bash
export LANG=en_US.UTF-8

currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}
currtime=`date +%Y%m%d%H%M%S`
mkdir -p ${currentDir}/result

# source public.lib
. ${currentDir}/../lib/public.lib

# user testcase
casecsv="case_resnext50_host.csv"
casenum=1

# docker or host
exectype="host"
rm -rf "${currentDir}/../d_solution"


ostype=`uname -m`
if [ x"${ostype}" = xaarch64 ];
then
    # arm,ubuntu_arm:18.04
    dockerImage="ubuntu_arm:18.04"
else
    # x86
    dockerImage="ubuntu:16.04"
fi



${currentDir}/../e2e_test.sh ${casecsv} ${casenum} ${exectype} 

