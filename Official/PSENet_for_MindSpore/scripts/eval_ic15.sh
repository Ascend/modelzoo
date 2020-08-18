#!/bin/bash
current_exec_path=$(pwd)
res_path=${current_exec_path}/res/submit_ic15/
eval_tool_path=${current_exec_path}/eval_ic15/

cd ${res_path}
zip ${eval_tool_path}/submit.zip ./*
cd ${eval_tool_path}
python ./script.py -s=submit.zip -g=gt.zip
cd ${current_exec_path}




