cur_path=`pwd`/../

#集合通信
export RANK_SIZE=1
export RANK_TABLE_FILE=$cur_path/configs/${RANK_SIZE}p.json
RANK_ID_START=0

#参数配置
#base param, using user config in python scripts if not config in this shell
num_train_steps=200
ckpt_path=/checkpoints
pipeline_config=$cur_path/models/research/configs/ssd320_full_1gpus.config

#npu param
data_path="/data"

if [[ $1 == --help || $1 == -h ]];then 
	echo "usage: ./train_full_1p.sh <args>"

	echo ""
	echo "parameter explain:
	--num_train_steps           training steps
	--data_path                 source data of training
	--ckpt_path                  pre-checkpoint path
	--pipeline_config           pipeline config path
	-h/--help             Show help message
	"
	exit 1
fi

for para in $*
do
    if [[ $para == --num_train_steps* ]];then
		num_train_steps=`echo ${para#*=}`
	elif [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
	elif [[ $para == --ckpt_path* ]];then
		ckpt_path=`echo ${para#*=}`
	elif [[ $para == --pipeline_config* ]];then
		pipeline_config=`echo ${para#*=}`
    fi
done	

if [[ $data_path == "" ]];then
	echo "[Error] para \"data_path\" must be config"
	exit 1
fi

#############执行训练#########################
cd $cur_path/models/research

if [  -f ${pipeline_config}.bak ];then
   cp ${pipeline_config}.bak ${pipeline_config}
else
   cp ${pipeline_config} ${pipeline_config}.bak
fi

sed -i "s%/checkpoints%${ckpt_path}%p" ${pipeline_config} 
sed -i "s%/data/coco2017_tfrecords%${data_path}/coco2017_tfrecords%p" ${pipeline_config} 


start=$(date +%s)

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
 do
  export RANK_ID=$RANK_ID
  if [   -d $cur_path/test/output ];then
     rm -rf $cur_path/test/output/*
     mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
  else
     mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
  fi

  nohup python3 -u ./object_detection/model_main.py \
       --pipeline_config_path=${pipeline_config} \
       --model_dir=${ckpt_path} \
       --alsologtostder \
       --amp \
       "${@:1}"  > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$RANK_ID.log 2>&1 &
done
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#############结果处理#########################
cp -r ${ckpt_path} $cur_path/test/output/$ASCEND_DEVICE_ID
   
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
 do
   echo "--------Result on Device-$ASCEND_DEVICE_ID/RANK-$RANK_ID ----------"
   step_sec=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$RANK_ID.log|awk 'END {print $2}'`
   echo "Final Performance global_step/sec : $step_sec"
 done

echo "Final Training Duration sec : $e2etime"