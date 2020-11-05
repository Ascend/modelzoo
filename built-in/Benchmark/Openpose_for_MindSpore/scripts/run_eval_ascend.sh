cd ..
python eval.py --model_path path_to_your_eval_model.ckpt --imgpath_val ./dataset/val2017 --ann ./dataset/annotations/person_keypoints_val2017.json > scripts/eval.log 2>&1 &
