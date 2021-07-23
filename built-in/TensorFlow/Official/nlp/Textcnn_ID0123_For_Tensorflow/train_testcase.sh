#python ./helper/cnews_group.py 
#python ./data/cnews_loader.py

#export EXPERIMENTAL_DYNAMIC_PARTITION=1
python3 run_cnn.py train > train.log 2>&1

# 测试
#export EXPERIMENTAL_DYNAMIC_PARTITION=1
python3 run_cnn.py test > test.log 2>&1

# 说明TS
# 1. 数据集划分
# 训练集：测试集：验证集为96：2：2

# 2. 主要参数设置
# hidden_dim = 1024, batch_size = 512  num_epochs = 100000  require_improvement = 10000 learning_rate = 1e-3 kernel_size = 5 num_filters = 1024 
# NPU Best Test Acc:  95.28%
echo "Run testcase success!"