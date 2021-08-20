#!/bin/bash



##############################如下是环境的性能相关规格信息#####################################
disk_path=${1:-'/npu'}
Perform_flag=0

#输出信息，不符合预期的颜色为红色，正常为绿色
#echo -e "\033[32m************************本机benchmark性能规格校验\033[0m\033[31m(异常参数显示为红色)\033[0m\033[32m************\033[0m"
echo -e "\033[32m***************************本机benchmark性能规格校验***************************\033[0m"

#处理器，操作系统版本（aarch64/x86_64只能二选一，当前不支持其他值）
platform=$(uname -p)
if (("$platform"=="aarch64"))||(("$platform"=="x86_64"));then
echo -e "处理器为:\033[32m$platform\033[0m"
else
echo -e "处理器为:\033[31m$platform\033[0m"
Perform_flag=1
fi

#release=$(cat /etc/euleros-release)
#echo -e "操作系统版本为:\033[32m$release\033[0m"

#CPU型号，当前参考值Kunpeng-920
cpu_info=$(LANG=C lscpu | awk -F: '/Model name/ {print $2}'|sed 's/^[ ]*//g')
#if(("$cpu_info"=="Kunpeng-920"));then
echo -e "CPU型号为:\033[32m$cpu_info\033[0m"
#else
#echo -e "CPU型号为:\033[31m$cpu_info\033[0m"
#Perform_flag=1
#fi

#CPU逻辑内核数量，当前参考值192
cpu_core=$(cat /proc/cpuinfo | grep "processor" |wc -l)
if [ "$platform" == "aarch64" ];then
  if(($cpu_core < 192));then
    echo "192"
    echo -e "CPU内核数量:\033[31m$cpu_core （不满足性能要求）\033[0m"
    Perform_flag=1
  else
    echo -e "CPU内核数量:\033[32m$cpu_core （满足性能那个要求）\033[0m"
  fi

elif [ "$platform" == "x86_64" ];then
  if(($cpu_core < 96));then
    echo "96"
    echo -e "CPU内核数量:\033[31m$cpu_core （不满足性能要求）\033[0m"
    Perform_flag=1
  else
    echo -e "CPU内核数量:\033[32m$cpu_core （满足性能要求）\033[0m"
  fi
fi




#CPU使用统计，当前参考值70%
cpu_idle=$(sar -u 1 1| awk '/Average/{print $8}')
#if [ $(echo "$cpu_idle < 80"|bc) = 1 ];then
if [ $(awk -v x=$cpu_idle -v y=70 'BEGIN {printf"%d",x/y}') = 0 ];then
echo -e "CPU空闲率为:\033[31m$cpu_idle （不满足性能要求）\033[0m"
Perform_flag=1
else
echo -e "CPU空闲率为:\033[32m$cpu_idle （满足性能要求）\033[0m"
fi

#CPU最近15分钟平均负载，当前负载持续大于0.7*CPU逻辑核数就要开始调查防止恶化
#当前参考值0.3*CPU逻辑核数
load15=$(uptime | sed 's/,//g' | awk '{print $(NF)}')
#if [ $(echo "$load15 > 0.3*$cpu_core"|bc) = 1 ];then
if [ $(awk -v x=$load15 -v y=88 'BEGIN {printf"%d",x/y}') -ge 1 ];then
echo -e "CPU最近15分钟平均负载为:\033[31m$load15 （不满足性能要求）\033[0m"
Perform_flag=1
else
echo -e "CPU最近15分钟平均负载为:\033[32m$load15 （满足性能要求）\033[0m"
fi

#总内存容量，剩余内存容量，当前参考值总容量1030000M，可用百分比80%
mem_total=$(free -m | awk '/Mem/{print $2}')
mem_free=$(free -m | awk '/Mem/{print $NF}')
mem_leftpert=$(printf "%d" $((100*mem_free/mem_total)))
#if [ $(echo "$mem_total < 1030000"|bc) = 1 ] && [ $(echo "$mem_leftpert < 80"|bc) = 1 ];then
if [ $(awk -v x=$mem_total -v y=700000 'BEGIN {printf"%d",x/y}') = 0 ] && [ $(awk -v x=$mem_leftpert -v y=70 'BEGIN {printf"%d",x/y}') = 0 ];then
echo -e "本机总内存容量为:\033[31m$mem_total\033[0m，剩余可用内存容量为:\033[31m$mem_free （不满足性能要求）\033[0m"
Perform_flag=1
#elif [ $(echo "$mem_total < 1030000"|bc) = 1 ] && [ $(echo "$mem_leftpert >= 80"|bc) = 1 ];then
elif [ $(awk -v x=$mem_total -v y=700000 'BEGIN {printf"%d",x/y}' ) = 0 ] && [ $(awk -v x=$mem_leftpert -v y=70 'BEGIN {printf"%d",x/y}') = 1 ];then
echo -e "本机总内存容量为:\033[31m$mem_total\033[0m，剩余可用内存容量为:\033[32m$mem_free （不满足性能要求）\033[0m"
Perform_flag=1
#elif [ $(echo "$mem_total >= 1030000"|bc) = 1 ] && [ $(echo "$mem_leftpert >= 80"|bc) = 1 ];then
elif [ $(awk -v x=$mem_total -v y=700000 'BEGIN {printf"%d",x/y}') -ge 1 ] && [ $(awk -v x=$mem_leftpert -v y=70 'BEGIN {printf"%d",x/y}') = 1 ];then
echo -e "本机总内存容量为:\033[32m$mem_total\033[0m，剩余可用内存容量为:\033[32m$mem_free （满足性能要求）\033[0m"
#elif [ $(echo "$mem_total >= 1030000"|bc) = 1 ] && [ $(echo "$mem_leftpert < 80"|bc) = 1 ];then
elif [ $(awk -v x=$mem_total -v y=700000 'BEGIN {printf"%d",x/y}') -ge 1 ] && [ ：$(awk -v x=$mem_leftpert -v y=70 'BEGIN {printf"%d",x/y}') = 0 ];then
echo -e "本机总内存容量为:\033[32m$mem_total\033[0m，剩余可用内存容量为:\033[31m$mem_free （不满足性能要求）\033[0m"
Perform_flag=1
fi



#总swap容量，剩余swap，当前参考值总swap4095M，可用百分比80%
swap_total=$(free -m | awk '/Swap/{print $2}')
swap_free=$(free -m | awk '/Swap/{print $NF}')
swap_leftpert=$(printf "%d" $((swap_free*100/swap_total)))
#if [ $(echo "$swap_total < 4095"|bc) = 1 ] && [ $(echo "$swap_leftpert < 80"|bc) = 1 ];then
if [ $(awk -v x=$swap_total -v y=2000 'BEGIN {printf"%d",x/y}') = 0 ] && [ $(awk -v x=$swap_leftpert -v y=70 'BEGIN {printf"%d",x/y}') = 0 ];then
echo -e "本机swap总容量:\033[31m$swap_total\033[0m，剩余swap容量:\033[31m$swap_free （不满足性能要求）\033[0m"
Perform_flag=1
#elif [ $(echo "$swap_total < 2000"|bc) = 1 ] && [ $(echo "$swap_leftpert >= 80"|bc) = 1 ];then
elif [ $(awk -v x=$swap_total -v y=4095 'BEGIN {printf"%d",x/y}') = 0 ] && [ $(awk -v x=$swap_leftpert -v y=70 'BEGIN {printf"%d",x/y}') = 1 ];then
echo -e "本机swap总容量:\033[31m$swap_total\033[0m，剩余swap容量:\033[32m$swap_free （不满足性能要求）\033[0m"
Perform_flag=1
#elif [ $(echo "$swap_total >= 2000"|bc) = 1 ] && [ $(echo "$swap_leftpert >= 80"|bc) = 1 ];then
elif [ $(awk -v x=$swap_total -v y=4095 'BEGIN {printf"%d",x/y}') -ge 1 ] && [ $(awk -v x=$swap_leftpert -v y=70 'BEGIN {printf"%d",x/y}') = 1 ];then
echo -e "本机swap总容量:\033[32m$swap_total\033[0m，剩余swap容量:\033[32m$swap_free （满足性能要求）\033[0m"
#elif [ $(echo "$swap_total >= 2000"|bc) = 1 ] && [ $(echo "$swap_leftpert < 80"|bc) = 1 ];then
elif [ $(awk -v x=$swap_total -v y=4095 'BEGIN {printf"%d",x/y}') -ge 1 ] && [ $(awk -v x=$swap_leftpert -v y=70 'BEGIN {printf"%d",x/y}') = 0 ];then
echo -e "本机swap总容量:\033[32m$swap_total\033[0m，剩余swap容量:\033[31m$swap_free （不满足性能要求）\033[0m"
Perform_flag=1
fi

#高速盘检查
disk_stand="nvme"
lsblk_type=$(lsblk | grep "$disk_path" | awk '{print$6}')
if [ "$lsblk_type" == "disk" ];then
  disk_type="$(lsblk | grep "$disk_path" | awk '{print$1}')"
else
  lsblk_line=$(lsblk | grep "$disk_path\|disk" | grep -n "$disk_path" | sed -n '1p' | awk -F: '{print$1}')
  disk_line=$(($lsblk_line-1))
  disk_type=$(lsblk | grep  "$disk_path\|disk" | sed -n "$disk_line p" | awk '{print$1}')
fi

if [[ $disk_type == $disk_stand* ]];then
echo -e "高速盘校验：当前是\033[32m$disk_type盘 (满足性能要求)\033[0m"
else
echo -e "高速盘校验：当前是\033[31m非$disk_stand盘 (不满足性能要求)\033[0m"
fi

#磁盘信息----改成参数

disk=$(df -h)
disk_npu=$(df -h | grep "$disk_path" | awk '{print $5}' | tr -cd "[0-9]")
#if [ $(echo "$disk_npu >= 80"|bc) = 1 ];then
if [ $(awk -v x=$disk_npu -v y=95 'BEGIN {printf"%d",x/y}') = 1 ];then
echo -e "磁盘信息如下：\033[31m（$disk_path不满足性能要求）\033[0m"
echo -e "\033[31m$disk\033[0m"
Perform_flag=1
else
echo -e "磁盘信息如下：\033[32m（$disk_path满足性能要求）\033[0m"
echo -e "\033[32m$disk\033[0m"
fi

#IO统计信息，%util一般超过80%表示磁盘可能处于繁忙状态
#%idle小于70% IO压力就比较大了
#iowait表示CPU等待IO时间占整个CPU周期的百分比，如果iowait值超过50%，或者明显大于%sustem/%user以及%idle，表示IO可能存在问题

iostat=$(iostat -x | sed '1,2d')
iostat_idle=$(iostat -x | sed '1,3d'| sed '3,$d'| awk '{print $6}')
iostat_iowait=$(iostat -x | sed '1,3d'| sed '3,$d'| awk '{print $4}')
#if [ $(echo "$iostat_idle < 70"|bc) = 1 ] && [ $(echo "$iostat_iowait > 0.5"|bc) = 1 ];then
if [ $(awk -v x=$iostat_idle -v y=70 'BEGIN {printf"%d",x/y}') = 0 ] || [ $(awk -v x=$iostat_iowait -v y=0.5 'BEGIN {printf"%d",x/y}') -ge 1 ];then
echo -e "IO统计信息如下：\033[31m（不满足性能要求）\033[0m"
echo -e "\033[31m$iostat\033[0m"
Perform_flag=1
else
echo -e "IO统计信息如下：\033[32m（满足性能要求）\033[0m"
echo -e "\033[32m$iostat\033[0m"
fi

