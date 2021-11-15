#!/bin/bash

datadir=./data/reds
cur_dir=$('pwd')
if [ ! -d ${datadir} ]; then
    mkdir -p ${datadir}
fi

echo "[INFO] Downloading REDS dataset. This may take some time (up to hours) depending on the network condition."
# echo "This operation will be running backstage"
# screen -dS downloading
python3 scripts/download_REDS.py --root_dir ${datadir} --train_sharp --train_sharp_bicubic --val_sharp --val_sharp_bicubic

cd ${datadir}
unzip -q train_sharp.zip
unzip -q train_sharp_bicubic.zip
unzip -q val_sharp.zip
unzip -q val_sharp_bicubic.zip

cd ${cur_dir}
echo "[INFO] REDS dataset downloaded"

echo "[INFO] Regroup REDS dataset as in the paper"
if [ ! -d ${datadir}/images ]; then
    mkdir -p ${datadir}/images
fi
python3 scripts/regroup_reds_dataset.py ${datadir}
echo "[INFO] REDS dataset regrouped"

echo "[INFO] Prepare REDS dataset"
if [ ! -d ${datadir}/sets ]; then
    mkdir -p ${datadir}/sets
fi

python3 scripts/make_reds_dataset.py ${datadir}
echo "[INFO] REDS dataset prepared"