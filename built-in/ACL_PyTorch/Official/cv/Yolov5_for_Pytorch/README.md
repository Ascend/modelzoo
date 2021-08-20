文件作用说明：

1.  modify_yolov5.py：onnx算子修改脚本 
2.  env.sh：ATC工具环境变量配置脚本
3.  parse_json.py： coco数据集标签json文件解析脚本 
4.  preprocess_yolov5_pytorch.py： 二进制数据集预处理脚本
5.  get_coco_info.py： yolov5.info生成脚本 
6.  bin_to_predict_yolo_pytorch.py： benchmark输出bin文件解析脚本
7.  map_calculate.py： 精度统计脚本
8.  require.txt：脚本运行所需的第三方库

推理端到端步骤：

（1） git clone 开源仓https://github.com/ultralytics/yolov5， 指定tag:v2.0，并下载对应的权重文件，修改**models/export.py**脚本生成onnx文件，注意目前onnx版本应选择11。

```shell
git clone -b v2.0 https://github.com/ultralytics/yolov5.git
python3.7 models/export.py --weights ./yolov5s.pt --img-size 640 --batch-size 1
```

（2）简化模型

  a.  对导出的onnx模型使用onnx-simplifer工具进行简化

```shell
python3.7 -m onnxsim --skip-optimization yolov5s.onnx yolov5s_sim.onnx
```

  b.  运行modify_yolov5.py修改生成的onnx文件

```shell
python3.7 modify_yolov5.py yolov5s_sim.onnx
```
将生成名为yolov5s_sim_t.onnx的模型。

（3）配置环境变量转换om模型

```shell
source env.sh
atc --model=yolov5s_sim_t.onnx --framework=5 --output=yolov5s_bs1 --input_format=NCHW --log=error --soc_version=Ascend310 --input_shape="images:1,3,640,640" --enable_small_channel=1 --input_fp16_nodes="images" --output_type=FP16
```

（4）解析数据集

下载coco2017数据集val2017和label文件**instances_val2017.json**，运行**parse_json.py**解析数据集

```shell
python3.7 parse_json.py
```

生成coco2017.names和coco_2017.info以及gronud-truth文件夹

（5）数据预处理

运行脚本preprocess_yolov5_pytorch.py处理数据集

```shell
python3.7 preprocess_yolov5_pytorch.py coco_2017.info yolov5_bin
```

（6）benchmark推理

运行get_coco_info.py生成info文件

```shell
python3.7 get_coco_info.py yolov5_bin ./coco_2017.info ./yolov5.info
```

执行benchmark命令，结果保存在同级目录 result/dumpOutput_device0/

```shell
./benchmark -model_type=vision -batch_size=1 -device_id=0 -om_path=yolov5_bs1.om -input_width=640 -input_height=640 -input_text_path=yolov5.info -useDvpp=false -output_binary=true
```

（7）后处理

运行 bin_to_predict_yolo_pytorch.py 解析模型输出

```shell
python3.7 bin_to_predict_yolo_pytorch.py  --bin_data_path result/dumpOutput_device0/  --origin_jpg_path val2017 --coco_class_names coco2017.names --conf_thres 0.001 --iou_thres 0.65
```

运行map_cauculate.py统计mAP值

```shell
python3 map_calculate.py --ground_truth_json ./instances_val2017.json --detection_results_json ./predictions.json
```

