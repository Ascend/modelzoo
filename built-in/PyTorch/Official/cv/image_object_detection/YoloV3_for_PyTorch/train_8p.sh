source ./test/env.sh
PORT=29500 ./tools/dist_train.sh configs/yolo/yolov3_d53_320_273e_coco.py 8 --cfg-options optimizer.lr=0.0032 --seed 0 --local_rank 0

