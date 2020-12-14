环境
----------
    pytorch 1.5
    torchvision 0.5.0
    apex 0.1
    easydict 1.9
    lmdb 0.98
    PyYAML 5.3

Host侧训练步骤
----------
1.准数据集<br>
2.在config.yaml文件中把TRAIN_ROOT和TEST_ROOT设置成存放数据集的路径<br>
3.DEVICE_ID是用哪张卡，END_EPOCH是训练多少epoch，BATCH_SIZE_PER_GPU是batchsize<br>
4.执行bash run_1p.sh开启单p训练<br>
5.把8p_config.yaml里的ADDR修改成训练服务器ip，执行bash run_8p.sh开启8p训练<br>


Docker侧训练步骤
----------
    
1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

        docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径(训练和评测数据集都放到该路径下)；模型执行路径；比如：

        ./docker_start.sh pytorch:b020 /train/imagenet /home/CRNN_for_Pytorch

3.执行2进入容器后，执行命令hostname -I |awk '{print $1}'获取到ip，若yaml配置文件中有addr参数， 请将addr的值配置为ip后执行步骤一
  训练流程（环境安装除外）

测试结果
----------
    
训练日志路径：在训练脚本的同目录下result文件夹里，如：

        /home/CRNN_for_Pytorch/result/training_8p_job_20201121023601
