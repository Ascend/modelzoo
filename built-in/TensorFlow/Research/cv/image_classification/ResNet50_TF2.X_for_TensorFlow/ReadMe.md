## ResNet50执行指南

### 1.git路径

https://gitlab.huawei.com/pmail_turinghava/training_shop/tree/master/03-code

### 2.数据集路径

TF2.X的ResNet50 与 10.136.165.4 上TF1.15的ResNet50使用数据集相同，不做重复归档

```sh
# 数据集路径
/autotest/CI_daily/Resnet50_TF_daily/data/resnet50/imagenet_TF
```

### 3.执行命令

```shell
# 一键拉起脚本
./perf_bytedance_while.sh

# 脚本解析
loop_size=${} 		# 训练步数,建议设置10000，约在15分钟左右结束
export NPU_LOOP_SIZE=$loop_size 	# 必设环境变量，需要等于训练步数
cd $/tensorflow						# 可执行脚本路径
python3 resnet_ctl_imagenet_main.py --data_dir=/home/imagenet_TF --train_epochs=200 --batch_size=32 --train_steps=${loop_size} --num_accumulation_steps=0 --model_dir=./ckpt_new_318 --distribution_strategy=off --use_tf_while_loop=true --use_tf_function=true --enable_checkpoint_and_export --steps_per_loop=${loop_size} --skip_eval --drop_eval_remainder=True

# 关键参数
--data_dir					# 数据集路径，请根据实际情况修改
--train_epochs				# 迭代数,根据实际情况设置,一个迭代为40036步
--batch_size				# 当前场景请设置32batch拉起训练
--skip_eval					# 跳过eval
```



