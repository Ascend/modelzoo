source npu_set_env.sh

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_4p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"
ln -s ${currentDir}/.data ${train_log_dir}/.data

python3.7 ${currentDir}/gru_8p.py \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed 123456 \
        --workers 160 \
        --print-freq 1 \
        --dist-url 'tcp://127.0.0.1:50000' \
        --dist-backend 'hccl' \
        --multiprocessing-distributed \
        --world-size 1 \
        --batch-size 2048 \
        --epoch 10 \
        --rank 0 \
        --device-list '0,1,2,3' \
        --amp \
        --bleu-npu 0 \
        --ckptpath ./seq2seq-gru-model.pth.tar   > ./gru_4p.log 2>&1 &

