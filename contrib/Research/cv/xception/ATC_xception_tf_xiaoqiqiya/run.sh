
#train
python    run_xception.py  --model_path ./model/xception_model.ckpt  --data_path ./data/train_1.tfrecord  --output_path  ./model_save  --do_train True  --image_num  50000 --class_num  1000  --batch_size  64  --epoch  10 --learning_rate  0.001   --save_checkpoints_steps  100

#eval
python    run_xception.py  --model_path ./model/xception_model.ckpt  --data_path ./data/train_1.tfrecord    --image_num  50000 --class_num  1000  --batch_size  100  
       

          