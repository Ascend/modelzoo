#  YOLOv3_TensorFlow

### 1. Introduction
This is npu implementation of [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) using TensorFlow modified from [YOLOv3_TensorFlow](https://github.com/wizyoung/YOLOv3_TensorFlow).   

### 2. Requirements
Python version: 3.7.5  
Main Python Packages:
- tensorflow >= 1.15.0 (satisfied with NPU)
- opencv-python
- tqdm

### 3. Weights convertion
The pretrained darknet53 weights file can be downloaded [here](https://pjreddie.com/media/files/darknet53.conv.74).        
Place this weights file under directory `./data/darknet_weights/` and then run:
```python
python3 convert_weight.py
```
Then the converted TensorFlow checkpoint file will be saved to `./data/darknet_weights/` directory.  
In this repo, conerted weight is contained. 

### 4. Training
#### 4.1 Data preparation 
0. dataset
To compare with official implement, for example, we use [get_coco_dataset.sh](https://github.com/pjreddie/darknet/blob/master/scripts/get_coco_dataset.sh) to prepare our dataset.

1. annotation file
Using script generate `coco2014_trainval.txt/coco2014_minival.txt` files under `./data/` directory.
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

#### 4.2 Training
1. single scale
Using `npu_train_*p_single.sh`. The hyper-parameters and the corresponding annotations can be found in `args_single.py`:

```shell
bash npu_train_1p_single.sh 
or 
bash npu_train_8p_single.sh
```

2. multi scale
Using `npu_train_*p_multi.sh`. The hyper-parameters and the corresponding annotations can be found in `args_multi.py`:

```shell
bash npu_train_1p_multi.sh 
or 
bash npu_train_8p_multi.sh
```

Check the `args.py` for more details. You should set the parameters yourself in your own specific task.

3. training details
     1. nohup.out -- training task main_log
     2. ./training/t1/D0/train_0.log -- training host log
     3. training/t1/D0/training/train.log -- training perf log

### 5. Evaluation

Using `eval.sh` to evaluate the validation or test dataset. The parameters are as following:

```shell
bash eval.sh
```

Check the `eval.py` for more details. You could set the parameters yourself. 

You will get the mAP metrics results using official cocoapi.
Using `tail -f eval_*.out` to watching results of models.


### 6. Training result

| Model                 | Npu_nums | mAP      | FPS       |
| :-------------------- | :------: | :------: | :------:  |
| single_scale          | 8        |    30.0  | 740       |
| multi_scale           | 8        |    31.0  | 340       |
| single_scale          | 1        |    ----  | 96        |
| multi_scale           | 1        |    ----  | 44        |




-------

### Credits:

I referred to many fantastic repos during the implementation:

[YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)

[qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)

[eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

[pjreddie/darknet](https://github.com/pjreddie/darknet)

[dmlc/gluon-cv](https://github.com/dmlc/gluon-cv/tree/master/scripts/detection/yolo)

