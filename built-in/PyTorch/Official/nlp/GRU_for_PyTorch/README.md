一、准数据集<br>
下载数据集，并在模型脚本目录创建data/multi30k，将数据集数据解压至该目录下。
mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz && tar -xf mmt16_task1_test.tar.gz -C data/multi30k && rm mmt16_task1_test.tar.gz

二、训练流程：
单卡训练流程：
	1.安装环境
	2.run_1p.sh文件修改字段--npu（单卡训练所使用的device id），为训练配置device_id，比如--npu 0
	3.cd到run_1p.sh文件的目录，执行bash run_1p.sh单卡脚本， 进行单卡训练
	4.run_1p.sh文件可配置内容如下：
        python3.7 gru_1p.py \
            --workers 40 \  #配置模型进程数
            --dist-url 'tcp://127.0.0.1:50000' \
            --world-size 1 \
            --npu 0 \   #使用device 
            --batch-size 512 \ #配置模型batchsize
            --epochs 10 \  #配置模型epochs
            --rank 0 \ 
            --amp \   #使用混合精度加速
    
多卡训练流程
	1.安装环境
	2.cd到run_8p.sh文件的目录，执行bash run_8p.sh等多卡脚本， 进行多卡训练	
	3.如需使用0,1,2,3卡，进行4p训练，则修改--device-list '0,1,2,3'，
	--workers和--batch-size减少相应倍数，分别修改为workers80，batch-size2048，2p、6p同理
        python3.7 gru_8p.py \
            --addr=$(hostname -I |awk '{print $1}') \ #自动获取主机ip
            --seed 123456 \ #固定随机种子
            --workers 160 \ #配置模型进程数
            --print-freq 1 \ #每*个step打印结果
            --dist-url 'tcp://127.0.0.1:50000' \
            --dist-backend 'hccl' \ 
            --multiprocessing-distributed \ #多p训练
            --world-size 1 \ 
            --batch-size 4096 \  #配置模型batchsize
            --epoch 10 \ #配置模型epoch
            --rank 0 \
            --device-list '0,1,2,3,4,5,6,7' \ #多p训练使用device
            --amp \  #使用混合精度加速
    	
三、Docker容器训练：

1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

    docker import ubuntuarmpytorch.tar pytorch:b020
2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

    ./docker_start.sh pytorch:b020 ./data/multi30k /home/GRU
3.执行步骤一训练流程（环境安装除外）
