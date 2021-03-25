start_time=`date +%s`
python3  IQN_tf.py \
--games=Pong \
--train_url=./model_save \
--load=0 \
--mode=train \
--loss_scale=1024 \
--target_replace_iter=100 \
--memory_capacity=100000 \
--batch_size=32 \
--envs_num 32 \
--learning_rate=0.0001 \
--step_num=25000000 \
--quant_num=64 \
--save_freq=1000
end_time=`date +%s`


echo execution time was `expr $end_time - $start_time` s.
          

