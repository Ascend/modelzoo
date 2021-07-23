#  YOLOv3_TensorFlow

## Table Of Contents

* [Description](#Description)
* [Requirements](#Requirements)
* [Default configuration](#Default-configuration)
  * [Optimizer](#Optimizer)
* [Quick start guide](#quick-start-guide)
  * [Prepare the dataset](#Prepare-the-dataset)
  * [Prepare the pretrained model](#Prepare-the-pretrained-model)
  * [Docker container scene](#Docker-container-scene)
  * [Check json](#Check-json)
  * [Key configuration changes](#Key-configuration-changes)
  * [Running the example](#Running-the-example)
    * [Training](#Training)
    * [Training process](#training-process)
    * [Evaluation](#Evaluation)
* [Advanced](#advanced)
  * [Command-line options](#Command-line-options)

## Description
YOLOv3 is currently a very widely used and effective target detection network, which has both high accuracy and performance. YOLOv3 is the third version of the YOLO series. Compared with YOLOv2, YOLOv3 mainly draws on some good solutions and integrates it into YOLO. While maintaining the speed advantage, it improves the prediction accuracy, especially the ability to recognize small objects. . The main improvements of YOLOv3 include: adjusting the network structure; using multi-scale features for object detection; and replacing softmax with Logistic for object classification.

- YOLOv3 model from: Redmon, Joseph, and Ali Farhadi. Yolov3: An incremental improvement.(https://pjreddie.com/media/files/papers/YOLOv3.pdf) 
- reference implementation: <https://github.com/wizyoung/YOLOv3_TensorFlow>   

## Requirements
Python version: 3.7.5  
Main Python Packages:
- tensorflow >= 1.15.0 (satisfied with NPU)
- opencv-python
- tqdm

Dataset:
- Download and preprocess COCO2014 or COCO2017 dataset for training and evaluation.

## Default configuration
The following sections introduce the default configurations and hyperparameters for Deeplabv3 model. 

### Optimizer
This model uses Momentum optimizer from Tensorflow with the following hyperparameters:

Training hyperparameters (single card-multi-scale):
- Batch size: 16
- Momentum: 0.9
- LR scheduler: cosine
- Base Learning rate: 0.0075
- Learning rate base batch size: 64
- loss scale: 128
- Weight decay: 0.0005
- Batch norm decay: 0.99
- Warm up epoch: 3
- train epoch: 200

Training hyperparameter (single card-single scale):
- Batch size: 32
- Momentum: 0.9
- LR scheduler: cosine
- Base Learning rate: 0.005
- Learning rate base batch size: 64
- loss scale: 128
- Weight decay: 0.0005
- Batch norm decay: 0.99
- Warm up epoch: 3
- train epoch: 200 

## Quick start guide

### Prepare the dataset

- Users are requested to prepare a dataset by themselves, including training set and verification set, optional including COCO2014, COCO2017, etc. The default read path in the training execution script is /opt/npu/dataset/coco, and some operations can be simplified through the soft chain real data to the default path.

- The currently provided training script uses the COCO2014 data set as an example, and data preprocessing operations are performed during the training process. If you change the data set, please modify the data set loading and preprocessing method in the training script before using the script.

(1)  According to the actual path of the COCO2014 data set, use `coco_trainval_anns.py` and `coco_minival_anns.py` to generate training and verification sample annotation files `coco2014_trainval.txt` and `coco2014_minival.txt`, and place them in the data directory. This code warehouse has generated annotation under data based on the default data set location. If the actual data set address location is consistent with the default location, this step can be skipped.

```python
python3 coco_trainval_anns.py
python3 coco_minival_anns.py
```
One line for one image, in the format like `image_index image_absolute_path img_width img_height box_1 box_2 ... box_n`.    
Box_x format: 
- `label_index x_min y_min x_max y_max`. (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)       
-  `image_index` is the line index which starts from zero. `label_index` is in range [0, class_num - 1].

For example:
```
0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
...
```

(2)  class_names file:
Generate the `data.names` file under `./data/` directory. Each line represents a class name.     
For example:     
```
bird
person
bike
...
```

The COCO dataset class names file is placed at `./data/coco.names`.

(3) prior anchor file:

Using the kmeans algorithm to get the prior anchors:

```
python get_kmeans.py
```

Then you will get 9 anchors and the average IoU. Save the anchors to a txt file.

The COCO dataset anchors offered by YOLO's author is placed at `./data/yolo_anchors.txt`, you can use that one too.

The yolo anchors computed by the kmeans script is on the resized image scale.  The default resize method is the letterbox resize, i.e., keep the original aspect ratio in the resized image. 


### Prepare the pretrained model

Please download the pre-trained model under the darknet framework by yourself.        
Place this weights file under directory `./data/darknet_weights/` and then use the script `convert_weight.py` to convert to the ckpt file of the TensorFlow framework:

```python
python3 convert_weight.py
```

Then the converted TensorFlow checkpoint file will be saved to `./data/darknet_weights/` directory.  

After the conversion of the ckpt model is completed, configure the corresponding location to `restore_path`.


### Docker container scene

- Compile image
```bash
docker build -t ascend-yolov3 .
```

- Start the container instance
```bash
bash docker_start.sh
```

Parameter Description:

```bash
#!/usr/bin/env bash
docker_image=$1 \   #Accept the first parameter as docker_image
data_dir=$2 \       #Accept the second parameter as the training data set path
model_dir=$3 \      #Accept the third parameter as the model execution path
docker run -it --ipc=host \
        --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \  #The number of cards used by docker, currently using 0~7 cards
        --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
        -v ${data_dir}:${data_dir} \    #Training data set path
        -v ${model_dir}:${model_dir} \  #Model execution path
        -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
        -v /var/log/npu/slog/:/var/log/npu/slog -v /var/log/npu/profiling/:/var/log/npu/profiling \
        -v /var/log/npu/dump/:/var/log/npu/dump -v /var/log/npu/:/usr/slog ${docker_image} \     #docker_image is the image name
        /bin/bash
```

After executing docker_start.sh with three parameters:
  - The generated docker_image
  - Dataset path
  - Model execution path
```bash
./docker_start.sh ${docker_image} ${data_dir} ${model_dir}
```



### Check json
Modify the *.json configuration file in the `hccl_config` directory, modify the corresponding IP to the current IP, and change the board_id to the ID of the motherboard of the machine.
Note: board_id is 0x0000 under X86, and 0x002f under arm.
1P rank_table json configuration file:

```
{
    "board_id": "0x0000",
    "chip_info": "910",
    "deploy_mode": "lab",
    "group_count": "1",
    "group_list": [
        {
            "device_num": "1",
            "server_num": "1",
            "group_name": "",
            "instance_count": "1",
            "instance_list": [
                {
                    "devices": [
                        {
                            "device_id": "0",
                            "device_ip": "192.168.100.101"
                        }
                    ],
                    "rank_id": "0",
                    "server_id": "0.0.0.0"
                }
           ]
        }
    ],
    "para_plane_nic_location": "device",
    "para_plane_nic_name": [
        "eth0"
    ],
    "para_plane_nic_num": "1",
    "status": "completed"
}
```

8P rank_table json configuration file:

```
{
    "board_id": "0x0000",
    "chip_info": "910",
    "deploy_mode": "lab",
    "group_count": "1",
    "group_list": [
        {
            "device_num": "8",
            "server_num": "1",
            "group_name": "",
            "instance_count": "8",
            "instance_list": [
                {
                    "devices": [
                        {
                            "device_id": "0",
                            "device_ip": "192.168.100.101"
                        }
                    ],
                    "rank_id": "0",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "1",
                            "device_ip": "192.168.101.101"
                        }
                    ],
                    "rank_id": "1",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "2",
                            "device_ip": "192.168.102.101"
                        }
                    ],
                    "rank_id": "2",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "3",
                            "device_ip": "192.168.103.101"
                        }
                    ],
                    "rank_id": "3",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "4",
                            "device_ip": "192.168.100.100"
                        }
                    ],
                    "rank_id": "4",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "5",
                            "device_ip": "192.168.101.100"
                        }
                    ],
                    "rank_id": "5",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "6",
                            "device_ip": "192.168.102.100"
                        }
                    ],
                    "rank_id": "6",
                    "server_id": "0.0.0.0"
                },
                {
                    "devices": [
                        {
                            "device_id": "7",
                            "device_ip": "192.168.103.100"
                        }
                    ],
                    "rank_id": "7",
                    "server_id": "0.0.0.0"
                }
            ]
        }
    ],
    "para_plane_nic_location": "device",
    "para_plane_nic_name": [
        "eth0",
        "eth1",
        "eth2",
        "eth3",
        "eth4",
        "eth5",
        "eth6",
        "eth7"
    ],
    "para_plane_nic_num": "8",
    "status": "completed"
}
```

### Key configuration changes
Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910 environment variable settings](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

Modify the configuration file args_single.py (single scale) or args_multi.py (multi-scale) to configure the relevant path to the user's actual path. The meaning of each field is shown in the table.
Configuration file description:

|field              | meaning                                                       |
| :-----------------| :-----------------------------------------------------------: |
| work_path         |Absolute path of the project (please configure before training)|
| train_file        |Training sample label file path                                |
| val_file          | Verify sample label file path                                 |
| restore_path      | Pre training model saving path                                |
| anchor_path       | Anchor file save path                                         |
| class_name_path   | Save path of category name file                               |


### Running the example

#### Training
```
bash npu_train.sh --data_path=... [--save_dir=... --batch_size=... --mode=... --rank_size=...]
```

bash npu_train.sh --help/-h for help info.

Check the `args.py` for more details. You should set the parameters yourself in your own specific task.

#### Training process

All the results of the training will be stored:

     1. nohup.out -- training task main_log
    
     2. training/t1/D0/train_0.log -- training host log
    
     3. training/t1/D0/training/train.log -- training perf log

#### Evaluation

Using `eval.sh` to evaluate the validation or test dataset. The parameters are as following:

```shell
bash eval.sh
```

Check the `eval.py` for more details. You could set the parameters yourself. 

You will get the mAP metrics results using official cocoapi.
Using `tail -f eval_*.out` to watching results of models.

## Advanced

### Command-line options

```
save_dir ='./training/'                            # save the path of ckpt
log_dir ='./training/logs/'                        # Path to save log files
progress_log_path ='./training/train.log'          # The path to save the training process log file

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
work_path ='../../../'
### Some paths
train_file = os.path.join(work_path,'./data/coco2014_trainval.txt')                  # Training annotation file path
val_file = os.path.join(work_path,'./data/coco2014_minival.txt')                     # Verify the label file path
restore_path = os.path.join(work_path,'./data/darknet_weights/darknet53.ckpt')       # pre-training model path
anchor_path = os.path.join(work_path,'./data/yolo_anchors.txt')                      # anchor file path
class_name_path = os.path.join(work_path,'./data/coco.names')                        # category name file path

### Distribution setting
num_gpus=int(os.environ['RANK_SIZE'])
iterations_per_loop=10                  # The number of small loops per sink

### Training releated numbersls

batch_size = 16                               # The batch size on each npu, multi-scale is 16 and single-scale is 32
img_size = [608, 608]                         # Model input width and height, multi-scale is 608*608, single-scale is 416*416
letterbox_resize = True                       # Whether to keep the input image aspect ratio, it needs to be turned on
total_epoches = 200                           # Total number of training epochs
train_evaluation_step = 1000                  # This parameter is currently invalid
val_evaluation_epoch = 2                      # This parameter is currently invalid
save_epoch = 10                               # Save the epoch interval of ckpt
batch_norm_decay = 0.99                       # decay parameter in bn operator
weight_decay = 5e-4                           # Weight decay parameter
global_step = 0                               # Specify to continue training from a specific step when continuing training

### tf.data parameters
num_threads = 8                              # Number of data preprocessing threads
prefetech_buffer = batch_size * 4            # The number of buffer prefetches in data preprocessing

### Learning rate and optimizer
optimizer_name ='momentum'                       # Optimizer, choose from [sgd, momentum, adam, rmsprop]
save_optimizer = True                            # Whether to save the optimizer in the ckpt file
learning_rate_base = 75e-4                       # Basic learning rate
learning_rate_base_batch_size = 64               # The basic batch size corresponding to the basic learning rate, the actual basic learning rate changes according to the batch size
```





