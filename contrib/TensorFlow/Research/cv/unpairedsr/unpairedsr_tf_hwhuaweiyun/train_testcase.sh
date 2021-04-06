#! /bin/bash
# 下载部分的数据集
obsutil cp obs://unpairedsr/share/data/train/vggcrop_train_lp10_5.npy data/train
obsutil cp obs://unpairedsr/share/data/dev/SRtrainset_2_2.npy data/dev
obsutil cp obs://unpairedsr/share/data/test/LS3D_6000.npy data/test

ok=1

python3.7 main.py --train --max_epoch 0
if [[ ! -e 'output/loss_acc.txt' || `grep -c ^ output/loss_acc.txt` -ne 6 ]]; then
    ok=0
fi

python3.7 main.py --test > test.log
psnr=`grep -o "\(1[89]\.[[:digit:]]\{4\}\)" test.log`
if [ ${psnr}='' ]; then
    ok=0
fi

if [ ok ]; then
    echo 'Run testcase success!'
else
    echo 'Run testcase failed!'
fi
