# Contents

- [DenseNet121 Description](#densenet121-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)    
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)  
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training accuracy results](#training-accuracy-results)
        - [Training performance results](#training-performance-results)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)


# [DenseNet121 Description](#contents)

DenseNet121 is a convolution based neural network for the task of image classification. The paper describing the model can be found [here](https://arxiv.org/abs/1608.06993). HuaWei’s DenseNet121 is a implementation on [MindSpore](https://www.mindspore.cn/).

The repository also contains scripts to launch training and inference routines.

# [Model Architecture](#contents)

DenseNet121 builds on 4 densely connected block. In every dense block, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers. Concatenation is used. Each layer is receiving a “collective knowledge” from all preceding layers.



# [Dataset](#contents)

Dataset used: ImageNet 
The default configuration of the Dataset are as follows:
 - Training Dataset preprocess:
   - Input size of images is 224\*224
   - Range (min, max) of respective size of the original size to be cropped is (0.08, 1.0)
   - Range (min, max) of aspect ratio to be cropped is (0.75, 1.333)
   - Probability of the image being flipped set to 0.5
   - Randomly adjust the brightness, contrast, saturation (0.4, 0.4, 0.4)
   - Normalize the input image with respect to mean and standard deviation

 - Test Dataset preprocess:
   - Input size of images is 224\*224 (Resize to 256\*256 then crops images at the center)
   - Normalize the input image with respect to mean and standard deviation



# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/enable_mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware. 
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.



# [Environment Requirements](#contents)

- Hardware（Ascend）
  - Prepare hardware environment with Ascend AI processor. If you want to try Ascend  , please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources. 
- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)



# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows: 

  ```python
  # run training example
  python train.py --data_dir /PATH/TO/DATASET --pretrained /PATH/TO/PRETRAINED_CKPT --is_distributed 0 > train.log 2>&1 & 
  
  # run distributed training example
  sh scripts/run_distribute_train.sh 8 rank_table.json /PATH/TO/DATASET /PATH/TO/PRETRAINED_CKPT
  
  # run evaluation example
  python eval.py --data_dir /PATH/TO/DATASET --pretrained /PATH/TO/CHECKPOINT > eval.log 2>&1 & 
  OR
  sh scripts/run_distribute_eval.sh 8 rank_table.json /PATH/TO/DATASET /PATH/TO/CHECKPOINT
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.



# [Script Description](#contents)

## [Script and Sample Code](#contents)

```
├── model_zoo
    ├── README.md                          // descriptions about all the models
    ├── densenet121        
        ├── README.md                    // descriptions about densenet121
        ├── scripts 
        │   ├── run_distribute_train.sh             // shell script for distributed on Ascend
        │   ├── run_distribute_eval.sh              // shell script for evaluation on Ascend
        ├── src 
        │   ├── datasets             // dataset processing function
        │   ├── losses          
        │       ├──crossentropy.py            // densenet loss function
        │   ├── lr_scheduler           
        │       ├──lr_scheduler.py            // densenet learning rate schedule function
        │   ├── network            
        │       ├──densenet.py            // densenet architecture
        │   ├──optimizers            // densenet optimize function
        │   ├──utils            
        │       ├──logging.py            // logging function
        │       ├──var_init.py            // densenet variable init function
        │   ├── config.py             // network config
        ├── train.py               // training script 
        ├── eval.py               //  evaluation script 
```

## [Script Parameters](#contents)

You can modify the training behaviour through the various flags in the `train.py` script. Flags in the `train.py` script are as follows:

```
  --data_dir              train data dir
  --num_classes           num of classes in dataset（default:1000)
  --image_size            image size of the dataset
  --per_batch_size        mini-batch size (default: 256) per gpu
  --pretrained            path of pretrained model
  --lr_scheduler          type of LR schedule: exponential, cosine_annealing
  --lr                    initial learning rate
  --lr_epochs             epoch milestone of lr changing
  --lr_gamma              decrease lr by a factor of exponential lr_scheduler
  --eta_min               eta_min in cosine_annealing scheduler
  --T_max                 T_max in cosine_annealing scheduler
  --max_epoch             max epoch num to train the model
  --warmup_epochs         warmup epoch(when batchsize is large)
  --weight_decay          weight decay (default: 1e-4)
  --momentum              momentum(default: 0.9)
  --label_smooth          whether to use label smooth in CE
  --label_smooth_factor   smooth strength of original one-hot
  --log_interval          logging interval(dafault:100)
  --ckpt_path             path to save checkpoint
  --ckpt_interval         the interval to save checkpoint
  --is_save_on_master     save checkpoint on master or all rank
  --is_distributed        if multi device(default: 1)
  --rank                  local rank of distributed(default: 0)
  --group_size            world size of distributed(default: 1)
```



## [Training Process](#contents)

### Training 

- running on Ascend

  ```
  python train.py --data_dir /PATH/TO/DATASET --pretrained /PATH/TO/PRETRAINED_CKPT --is_distributed 0 > train.log 2>&1 & 
  ```
  
  The python command above will run in the background, The log and model checkpoint will be generated in `output/202x-xx-xx_time_xx_xx_xx/`. The loss value will be achieved as follows:
   
  ```
  2020-08-22 16:58:56,617:INFO:epoch[0], iter[5003], loss:4.367, mean_fps:0.00 imgs/sec
  2020-08-22 16:58:56,619:INFO:local passed
  2020-08-22 17:02:19,920:INFO:epoch[1], iter[10007], loss:3.193, mean_fps:6301.11 imgs/sec
  2020-08-22 17:02:19,921:INFO:local passed
  2020-08-22 17:05:43,112:INFO:epoch[2], iter[15011], loss:3.096, mean_fps:6304.53 imgs/sec
  2020-08-22 17:05:43,113:INFO:local passed
  ...
  ```



### Distributed Training

- running on Ascend

  ```
  sh scripts/run_distribute_train.sh 8 rank_table.json /PATH/TO/DATASET /PATH/TO/PRETRAINED_CKPT
  ```
  
  The above shell script will run distribute training in the background. You can view the results log and model checkpoint through the file `train[X]/output/202x-xx-xx_time_xx_xx_xx/`. The loss value will be achieved as follows:
  
  ```
  2020-08-22 16:58:54,556:INFO:epoch[0], iter[5003], loss:3.857, mean_fps:0.00 imgs/sec
  2020-08-22 17:02:19,188:INFO:epoch[1], iter[10007], loss:3.18, mean_fps:6260.18 imgs/sec
  2020-08-22 17:05:42,490:INFO:epoch[2], iter[15011], loss:2.621, mean_fps:6301.11 imgs/sec
  2020-08-22 17:09:05,686:INFO:epoch[3], iter[20015], loss:3.113, mean_fps:6304.37 imgs/sec
  2020-08-22 17:12:28,925:INFO:epoch[4], iter[25019], loss:3.29, mean_fps:6303.07 imgs/sec
  2020-08-22 17:15:52,167:INFO:epoch[5], iter[30023], loss:2.865, mean_fps:6302.98 imgs/sec
  ...
  ...
  ```



## [Evaluation Process](#contents)

### Evaluation

- evaluation on Ascend

  running the command below for evaluation. 
  
  ```
  python eval.py --data_dir /PATH/TO/DATASET --pretrained /PATH/TO/CHECKPOINT > eval.log 2>&1 & 
  OR
  sh scripts/run_distribute_eval.sh 8 rank_table.json /PATH/TO/DATASET /PATH/TO/CHECKPOINT
  ```
  
  The above python command will run in the background. You can view the results through the file "output/202x-xx-xx_time_xx_xx_xx/202x_xxxx.log". The accuracy of the test dataset will be as follows:
  
  ```
  2020-08-24 09:21:50,551:INFO:after allreduce eval: top1_correct=37657, tot=49920, acc=75.43%
  2020-08-24 09:21:50,551:INFO:after allreduce eval: top5_correct=46224, tot=49920, acc=92.60%
  ```




# [Model Description](#contents)
## [Performance](#contents)

### Training accuracy results

| Parameters          | Densenet                    |
| ------------------- | --------------------------- |
| Model Version       | Inception V1                |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/15/2020 (month/day/year) |
| MindSpore Version   | 1.0.0              |
| Dataset             | ImageNet                    |
| epochs              | 120                         |
| outputs             | probability                 |
| train performance   | Top1:75.13%; Top5:92.57%    |

### Training performance results

| Parameters          | Densenet                    |
| ------------------- | --------------------------- |
| Model Version       | Inception V1                |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/15/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                 |
| Dataset             | ImageNet                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| speed               | 1pc:760 img/s;8pc:6000 img/s|



# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py. 


# [ModelZoo Homepage](#contents)  
 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
