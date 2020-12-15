cd ..
python eval.py \
  --model_path ./outputs/2020-10-15_time_14_58_10/0-34_180000.ckpt \
  --imgpath_val ./dataset/val2017 \
  --ann ./dataset/annotations/person_keypoints_val2017.json \
  > scripts/eval.log 2>&1 &
