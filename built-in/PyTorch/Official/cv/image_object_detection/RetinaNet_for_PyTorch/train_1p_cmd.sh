#python3 -m torch.distributed.launch  tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py --gpus 1 --cfg-options optimizer.lr=0.005 seed=0 --launcher pytorch
#PORT=29500 ./tools/dist_train.sh configs/retinanet/retinanet_r50_fpn_1x_coco.py 1 --cfg-options optimizer.lr=0.005 seed=0
rm -rf kernel_meta/
/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PTCOPY_ENABLE=1
#export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
unset DUMP_GRAPH_LEVEL
unset DUMP_GE_GRAPH
export COMBINED_ENABLE=1
export DYNAMIC_COMPILE_ENABLE=0
export EXPERIMENTAL_DYNAMIC_PARTITION=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
#python tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py --seed 0 --gpu-ids 2 --opt-level O1 --resume-from work_dirs/epoch_1.pth
#python tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py --seed 0 --gpu-ids 2 --opt-level O1
PORT=29504 ./tools/dist_train.sh configs/retinanet/retinanet_r50_fpn_1x_coco.py 1 --cfg-options optimizer.lr=0.005 --seed 0 --gpu-ids 0 --opt-level O1
#PORT=29503 ./tools/dist_train.sh configs/retinanet/retinanet_r50_fpn_1x_coco.py 4 --cfg-options optimizer.lr=0.02 --seed 0 --gpu-ids 0