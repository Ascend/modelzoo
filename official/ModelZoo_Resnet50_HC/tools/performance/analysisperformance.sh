#!/bin/bash

currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}
if [ $# -eq 1 ]
then
        devicenum=$1
else
        echo "========================================================================================="
        echo "usage:"
        echo "bash performanceAnalysis.sh devicenum"
        echo "eg:"
        echo "bash performanceAnalysis.sh 1"
        echo "========================================================================================="
        exit 1
fi

need_1=`pip3.7 show XlsxWriter | grep Name`
if [ x"${need_1}" = x ];
then
    pip3.7 install --no-index XlsxWriter-1.1.8-py2.py3-none-any.whl
fi

need_2=`pip3.7 show xlrd | grep Name`
if [ x"${need_2}" = x ];
then
    pip3.7 install --no-index xlrd-1.2.0-py2.py3-none-any.whl
fi

ret_1=`pip3.7 show XlsxWriter | grep Name`
ret_2=`pip3.7 show xlrd | grep Name`
if [ x"${ret_1}" = x -o x"${ret_2}" = x ];
then
    echo "[`date +%Y%m%d-%H:%M:%S`] [ERROR] XlsxWriter[${ret_1}],xlrd[${ret_2}]"
    exit 1
fi

rm -rf $currentDir/performanceAnalysis_1p/*_file/*
rm -rf $currentDir/performanceAnalysis/*_file/*

if [ ${devicenum} = 1 ]
then 
    rm -rf $currentDir/performanceAnalysis_1p/analysis*
    python3.7 $currentDir/performanceAnalysis_1p/performanceanalysis.py
    python3.7 $currentDir/performanceAnalysis_1p/mean.py $currentDir/performanceAnalysis_1p/analysis_performance_tag.*.xlsx > $currentDir/performanceAnalysis_1p/analysis_summary.txt
else
    rm -rf $currentDir/performanceAnalysis/analysis*
    python3.7 $currentDir/performanceAnalysis/performanceanalysis.py
    python3.7 $currentDir/performanceAnalysis/mean.py $currentDir/performanceAnalysis/analysis_performance_tag.1.xlsx >$currentDir/performanceAnalysis/analysis_summary.txt
fi
