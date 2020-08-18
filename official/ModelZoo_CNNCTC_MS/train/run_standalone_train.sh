#!/bin/bash
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
PATH1=$(get_real_path $1)

ulimit -u unlimited
export DEVICE_ID=0

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ./*.py ./train
cp ./scripts/*.sh ./train
cp -r ./src ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log
if [ -f $PATH1 ]
then
  python train.py --ckpt_path=$PATH1 &> log &
else
  python train.py &> log &
fi
cd ..
