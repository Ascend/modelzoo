#!/bin/bash
#set -x
sh_id=`ps -ef|grep "run_npu_8p.sh"|wc -l`
if [ $sh_id -gt 1 ];then
  echo "[INFO] Start kill run.sh"
  ps -ef|grep "run_npu_8p.sh"|xargs kill -9
  if [ $? == 0  ];then
    echo "[INFO] Kill run.sh success"
    sleep 5
  fi
else
  echo "[INFO] No run_npu process exist."
fi

py_id=`ps -ef|grep python|grep fast-rcnn|wc -l`
if [ $py_id -gt 0 ];then
  ps -ef|grep python|grep fast-rcnn|xargs kill -9
  if [ $? == 0  ];then
    echo "[INFO] Kill python success"
    sleep 5
  fi
else
  echo "[INFO] No fast-rcnn python process exist."
fi

echo `ps -ef|grep python|grep fast-rcnn`
