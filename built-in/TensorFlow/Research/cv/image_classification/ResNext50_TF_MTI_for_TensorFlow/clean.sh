#/bin/bash

#choose if remove slog and hisi_logs
read -p "clear /var/log/npu/slog/* and /var/log/npu/profiling/*: Y/N? " slog_flag
deleteSlog=$(echo $slog_flag | tr [a-z] [A-Z])
if [ x$deleteSlog != xN ];then
  echo "============start delete slog========================="
  rm -rf /var/log/npu/slog/*  &
  rm -rf /var/log/npu/profiling/*
  echo "=================END!================================="
fi


#choose if remove result_* folder
read -p "clear result/*: Y/N? " result_flag
deleteResult=$(echo $result_flag | tr [a-z] [A-Z])
if [ x$deleteResult != xN ];then
    echo "============start delete result logs================"
    rm -rf ./result*/* &
    echo "====================END!============================"
fi
