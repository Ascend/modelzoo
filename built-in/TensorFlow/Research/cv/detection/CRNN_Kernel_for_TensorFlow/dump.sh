


source env.sh

#export PRINT_MODEL=1
export DUMP_GE_GRAPH=1
#export DUMP_GRAPH_LEVLE=3
#export SLOG_PRINT_TO_STDOUT=0



export RANK_SIZE=1
export DEVICE_ID=1
bash train_1p.sh ${DEVICE_ID} 40 0.015 &

