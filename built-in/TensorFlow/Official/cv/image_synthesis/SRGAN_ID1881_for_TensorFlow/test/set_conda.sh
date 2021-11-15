#for para in $*
#do
#	if [[ $para == --conda_name* ]];then
#		conda_name=`echo ${para#*=}`
#		export PATH=/home/anaconda3/bin:$PATH
#		export LD_LIBRARY_PATH=/home/anaconda3/lib:$LD_LIBRARY_PATH
#	fi
#done
export PATH=/home/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/anaconda3/lib:$LD_LIBRARY_PATH