cur_path=`pwd`/../

#环境变量
export install_path=/usr/local/Ascend 
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}
export LD_LIBRARY_PATH=${install_path}/driver/lib64/common/:${install_path}/driver/lib64/driver:$LD_LIBRARY_PATH # 仅容器训练场景配置
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/python3.7.5/lib/python3.7/site-packages:${install_path}/tfplugin/python/site-packages:${install_path}/fwkacllib/python/site-packages:$PYTHONPATH
export PYTHONPATH=$cur_path/models/research:$cur_path/models/research/slim:$PYTHONPATH
export JOB_ID=10087

#集合通信
export RANK_SIZE=1
export RANK_TABLE_FILE=$cur_path/configs/${RANK_SIZE}p.json
export ASCEND_DEVICE_ID=0
RANK_ID_START=0

#日志开关
export ASCEND_GLOBAL_LOG_LEVEL=3
/usr/local/Ascend/driver/tools/msnpureport -g error

#参数配置
#base param, using user config in python scripts if not config in this shell
batch_size=0
learning_rate=0
num_train_steps=0
#epochs=0
eval_count=0
ckpt_dir=$cur_path/checkpoints
pipeline_config=$cur_path/models/research/configs/ssd320_full_1gpus.config

#npu param
precision_mode="allow_fp32_to_fp16"
loss_scale_flag=0
loss_scale_value=0
loss_scale_type="static"
over_dump=False
data_dump_flag=0
data_dump_step=10
profiling=False
random_remove=False
data_path="/data"

if [[ $1 == --help || $1 == -h ]];then 
	echo "usage: ./train_full_1p.sh <args>"

	echo ""
	echo "parameter explain:
	--precision_mode            precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision), default is allow_fp32_to_fp16
	--loss_scale_value          loss scale value, default is 0, not use loss scale
	--loss_scale_type           loss scale type(dynamic/static), default is static
	--over_dump                 if or not over detection, default is False
	--data_dump_flag            data dump flag, default is 0
	--data_dump_step            data dump step, default is 10
	--profiling                 if or not profiling for performance debug, default is Fasle
	--random_remove             remove train random treament, default is False
	--batch_size                train batch size
	--learning_rate             leaning rate
	--num_train_steps           training steps
	--data_path                 source data of training
	--ckpt_count                save checkpoint counts
	--ckpt_dir                  pre-checkpoint path
	--pipeline_config           pipeline config path
	-h/--help             Show help message
	"
	exit 1
fi

for para in $*
do
    if [[ $para == --precision_mode* ]];then
       	precision_mode=`echo ${para#*=}`
	elif [[ $para == --loss_scale_value* ]];then
		loss_scale_value=`echo ${para#*=}`
	elif [[ $para == --loss_scale_type* ]];then
		loss_scale_type=`echo ${para#*=}`
	elif [[ $para == --over_dump* ]];then
		over_dump=`echo ${para#*=}`
	elif [[ $para == --data_dump_flag* ]];then
		data_dump_flag=`echo ${para#*=}`
	elif [[ $para == --data_dump_step* ]];then
		data_dump_step=`echo ${para#*=}`
	elif [[ $para == --profiling* ]];then
		profiling=`echo ${para#*=}`
	elif [[ $para == --batch_size* ]];then
		batch_size=`echo ${para#*=}`
	elif [[ $para == --learning_rate* ]];then
		learning_rate=`echo ${para#*=}`
	elif [[ $para == --num_train_steps* ]];then
		num_train_steps=`echo ${para#*=}`
	elif [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
	elif [[ $para == --ckpt_count* ]];then
		ckpt_count=`echo ${para#*=}`
	elif [[ $para == --random_remove* ]];then
		random_remove=`echo ${para#*=}`
	elif [[ $para == --ckpt_dir* ]];then
		ckpt_dir=`echo ${para#*=}`
	elif [[ $para == --pipeline_config* ]];then
		pipeline_config=`echo ${para#*=}`
    fi
done	

#if [[ $data_path == "" ]];then
#	echo "[Error] para \"data_path\" must be config"
#	exit 1
#fi

#############执行训练#########################
cd $cur_path/models/research

start=$(date +%s)

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
 do
  mkdir -p $cur_path/test/output/log
  nohup python3 -u ./object_detection/model_main.py \
       --pipeline_config_path=${pipeline_config} \
       --model_dir=${ckpt_dir} \
       --alsologtostder \
       --amp \
       --precision_mode=$precision_mode  \
       --loss_scale_value=$loss_scale_value \
       --loss_scale_type=$loss_scale_type \
       --over_dump=$over_dump \
       --data_dump_flag=$data_dump_flag \
       --data_dump_step=$data_dump_step \
       --profiling=$profiling \
       --random_remove=$random_remove \
       --config_override={  \
             train_config: {  \
               fine_tune_checkpoint: "$ckpt_dir/resnet_v1_50/model.ckpt"  \
             }  \
             train_input_reader: {  \
               tf_record_input_reader {  \
               input_path: "$data_path/coco2017_tfrecords/*train*"  \
               }  \
            }  \
            eval_input_reader: {  \
              tf_record_input_reader {  \
              input_path: "$data_path/coco2017_tfrecords/*val*"  \
              }  \
            }  \
	        }  \
       "${@:1}"  > $cur_path/test/output/log/train_$RANK_ID.log 2>&1 &
done
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#############结果处理#########################
cp -r ${ckpt_dir} $cur_path/test/output
   
sum_perf=0
sum_prec=0
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
 do
   echo "--------Result on Device-$RANK_ID ----------"
   #Precision = `grep -a 'Average Precision  (AP)'  $cur_path/test/output/log/train_$RANK_ID.log| grep -a 'IoU=0.50:0.95'` | awk '{print $13}'`
   step_sec=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/test/output/log/train_$RANK_ID.log|awk 'END {print $2}'`
   Performance=`awk 'BEGIN{printf "%.2f\n",'1000'/$step_sec}'`
   #echo "Final Performance MAP : $Precision"
   echo "Final Performance ms/step : $Performance"
   sum_perf=$((sum_perf+step_sec))
   sum_prec=$((sum_prec+Performance))
 done

average_perf=$((sum_perf/$RANK_SIZE))
average_prec=$((sum_prec/$RANK_SIZE))
 
echo "--------Final Result ----------"
#echo "Final Precision MAP : $average_prec"
echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"


 





