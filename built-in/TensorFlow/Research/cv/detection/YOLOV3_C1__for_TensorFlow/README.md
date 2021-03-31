

## part 1. Quick start

1. You are supposed  to install some dependencies before getting out hands with these codes.

```bashrc
$ pip3 install -r ./docs/requirements.txt
```

   2.Exporting loaded COCO weights as TF checkpoint(`yolov3_coco.ckpt`)

```bashrc
$ cd checkpoint
$ scp -r root@10.136.165.4:/turingDataset/ModelZoo_YOLOV3_C1_TF/checkpoint/* ./
```

## part 2. Train

Two files are required as follows:

- [`dataset.txt`](https://raw.githubusercontent.com/YunYang1994/tensorflow-yolov3/master/data/dataset/voc_train.txt): 

```
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height
```

- [`class.names`](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/data/classes/coco.names):

```
person
bicycle
car
...
toothbrush
```

### 2.1 Train on VOC dataset

Extract all of these tars into one directory and rename them, which should have the following basic structure.

```bashrc
VOC           # path:  /home/yang/dataset/VOC
├── test
|    └──VOCdevkit
|        └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
         └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
         └──VOC2012 (from VOCtrainval_11-May-2012.tar)
                     
$ python scripts/voc_annotation.py --data_path /home/yang/test/VOC
```

Then edit your `./core/config.py` to make some necessary configurations

```bashrc
__C.YOLO.CLASSES                = "./data/classes/voc.names"
__C.TRAIN.ANNOT_PATH            = "./data/dataset/voc_train.txt"
__C.TEST.ANNOT_PATH             = "./data/dataset/voc_test.txt"
```

##### train from COCO weights:

```bashrc
$ python3 train.py
```

### 2.2 Evaluate on VOC dataset

```
$ python3 evaluate.py
$ cd mAP
$ python3 main.py -na
```



