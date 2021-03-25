# !/bin/bash

datadir=./data/reds
if [ ! -d ${datadir} ]; then
    mkdir -p ${datadir}
fi

echo "[INFO] Downloading REDS dataset. This may take some time (up to hours) depending on the network condition."
# echo "This operation will be running backstage"
# screen -dS downloading
python3 download_REDS.py --root_dir ${data_dir} --train_sharp --train_sharp_bicubic --val_sharp --val_sharp_bicubic
echo "[INFO] REDS dataset downloaded"

echo "[INFO] Regroup REDS dataset as in the paper"
python3 regroup_reds_dataset.py

if [ ! -d ${datadir}/images ]; then
    mkdir -p ${datadir}/images
fi

mv ${datadir}/train_sharp ${datadir}/images/truth
mv ${datadir}/train_sharp_bicubic/X4 ${datadir}/images/X4
echo "[INFO] REDS dataset regrouped"

echo "[INFO] Prepare REDS dataset"
if [ ! -d ${datadir}/sets ]; then
    mkdir -p ${datadir}/sets
fi

python3 make_reds_dataset.py ${datadir}
echo "[INFO] REDS dataset prepared"


