set -x
export POD_NAME=another0

execpath=${PWD}
rm -rf *.pbtxt
ulimit -c 0
unzip outdata.zip
python3.7 train.py --data_dir ./outdata/ --seq_length 3 --reconstr_weight 0.85 --smooth_weight 0.05 --ssim_weight 0.15 --icp_weight 0 --train_steps 1 --summary_freq 1 --checkpoint_dir ./checkpoints > train.log 2>&1

key1="\[GEOP\]"  #功能检查字


if [ `grep -c "$key1" "train.log"` -ne '0' ] && [ `grep -c "Epoch" "train.log"` -ne '0' ];then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

