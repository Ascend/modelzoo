#!/bin/bash
a=866
c=6
b=`awk 'BEGIN{printf "%.2f\n", '${c}'*1000/'${a}'}'`
echo ${b}
