文件作用说明：

1.  modify_yolov5.py：onnx算子修改脚本 
2.  env.sh：ATC工具环境变量配置脚本
3.  parse_json.py： coco数据集标签json文件解析脚本 
4.  preprocess_yolov5_pytorch.py： 二进制数据集预处理脚本
5.  get_coco_info.py： yolov5.info生成脚本 
6.  bin_to_predict_yolo_pytorch.py： benchmark输出bin文件解析脚本
7.  map_calculate.py： 精度统计脚本
8.  require.txt：脚本运行所需的第三方库
9.  benchmark工具源码地址：https://gitee.com/ascend/cann-benchmark/tree/master/infer

推理端到端步骤：

（1） git clone 开源仓https://github.com/ultralytics/yolov5/， 并下载对应的权重文件，修改**models/export.py**脚本生成onnx文件，注意目前onnx版本须选择11

```
git clone https://github.com/ultralytics/yolov5/releases
python3.7 models/export.py --weights ./yolov5s.pt --img 640 --batch 1
```

（2）运行modify_yolov5.py修改生成的onnx文件

```
python3.7 modify_yolov5.py
```

（3）配置环境变量转换om模型

```
source env.sh
atc --model=modify_yolov5.onnx --framework=5 --output=yolov5_bs1 --input_format=NCHW --log=info --soc_version=Ascend310 --input_shape="images:1,12,320,320" --out_nodes="Reshape_573:0;Reshape_589:0;Reshape_605:0" --enable_small_channel=1
```

（4）解析数据集

下载coco2014数据集val2014和label文件**instances_valminusminival2014.json**，运行**parse_json.py**解析数据集

```
python3.7 parse_json.py
```

生成coco2014.names和coco_2014.info以及gronud-truth文件夹

（5）数据预处理

运行脚本preprocess_yolov5_pytorch.py处理数据集

```
python3.7 preprocess_yolov5_pytorch.py coco_2014.info yolov5_bin
```

（6）benchmark推理

运行get_coco_info.py生成info文件

```
python3.7 get_coco_info.py yolo_coco_bin_tf coco_2014.info yolov5.info
```

执行benchmark命令，结果保存在同级目录 result/dumpOutput_device0/

```
./benchmark -model_type=vision -batch_size=1 -device_id=0 -om_path=yolov5_bs1.om -input_width=320 -input_height=320 -input_text_path=yolov5.info -useDvpp=false -output_binary=true
```

（7）后处理

运行 bin_to_predict_yolo_pytorch.py 解析模型输出

```
python3.7 bin_to_predict_yolo_pytorch.py  --bin_data_path result/dumpOutput_device0/  --det_results_path  detection-results/ --origin_jpg_path /root/dataset/coco2014/val2014/ --coco_class_names /root/dataset/coco2014/coco2014.names --model_type yolov3
```

运行map_cauculate.py统计mAP值

```
python3 map_calculate.py --label_path  ./ground-truth  --npu_txt_path ./detection-results -na -np
```

