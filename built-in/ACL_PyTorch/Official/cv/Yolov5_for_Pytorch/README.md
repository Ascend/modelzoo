文件作用说明：

1. acl_net.py：pyacl接口封装脚本 
2. env.sh：ATC工具环境变量配置脚本
3. map_calculate.py： mAP精度统计脚本
4. modify_model.py：onnx模型修改脚本
5. om_infer.py：模型推理demo脚本
6. onnx2om.sh: onnx转om脚本
7. parse_json.py： coco数据集标签json文件解析脚本
8. requirements.txt：脚本运行所需的第三方库

推理端到端步骤：

（1） 下载代码
git clone 开源仓 https://github.com/ultralytics/yolov5 ，切换到tag:v2.0，

```shell
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v5.0  # 切换到v5.0代码,支持2.0->5.0代码，对于v3.0版本请切换到bugfix版本，git checkout 4d7f222
```
（2）对源码做简单修改

a. 修改models/common.py文件，对其中Focus的forward函数做修改，提升Slice算子性能
```python
class Focus(nn.Module):
    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # <==== 修改内容
        if torch.onnx.is_in_onnx_export():
            a, b = x[..., ::2, :].transpose(-2, -1), x[..., 1::2, :].transpose(-2, -1)
            c = torch.cat([a[..., ::2, :], b[..., ::2, :], a[..., 1::2, :], b[..., 1::2, :]], 1).transpose(-2, -1)
            return self.conv(c)
        else:
            return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))
        # =====>
```

b. 修改models/yolo.py脚本，使后处理部分不被导出
```python

class Detect(nn.Module):
    def forward(self, x):
        # ...
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # <==== 修改内容
            if torch.onnx.is_in_onnx_export():
                continue
            # =====>
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        # ...
```

c. 修改models/export.py文件，将转换的onnx算子版本设为11
```python
torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'], do_constant_folding=True,
                  output_names=['classes', 'boxes'] if y is None else ['output'])
```

d. 修改models/experimental.py文件，将其中的attempt_download()所在行注释掉
```python
def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # <==== 修改内容
        # attempt_download(w)
        # =====>
        ckpt = torch.load(w, map_location=map_location)  # load
```

e. 下载对应的权重文件置于yolov5目录下，运行脚本导出模型

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
python3.7 models/export.py --weights ./yolov5s.pt --img-size 640 --batch-size 1
```

（2）简化模型

a.  对导出的onnx模型使用onnx-simplifer工具进行简化

```shell
python3.7 -m onnxsim --skip-optimization yolov5s.onnx yolov5s_sim.onnx
```

b.  运行modify_yolov5.py修改生成的onnx文件，添加后处理算子

```shell
python3.7 modify_model.py --model=yolov5s_sim.onnx --conf-thres=0.4 --iou-thres=0.5
```
参数说明：
--model: 原始onnx模型
--conf-thres: 后处理算子置信度阈值
--iou-thres: 后处理算子IOU阈值
运行脚本后，将生成名为yolov5s_sim_t.onnx的模型。

（3）配置环境变量转换om模型

```shell
source env.sh
./onnx2om.sh yolov5s_sim_t.onnx yolov5s_sim_t 1
```

（4）解析数据集

下载coco2017数据集val2017和label文件**instances_val2017.json**，运行**parse_json.py**解析数据集

```shell
python3.7 parse_json.py
```

生成coco2017.names和coco_2017.info以及gronud-truth文件夹

（5）推理
配置环境变量，运行脚本进行推理

```shell
source env.sh  # 如果前面配置过，这里不用执行
python3.7 om_infer.py --model=yolov5s_bs1.om --img-path=./val2017 --batch-size=1
```
参数说明：
--model: om模型所在位置
--img-path: 是测试数据集图片所在路径
--batch-size: 模型batch size


（6）统计mAP值
运行map_cauculate.py

```shell
python3 map_calculate.py --ground_truth_json ./instances_val2017.json --detection_results_json ./predictions.json
```
参数说明：
--ground_truth_json: 标杆标注文件
--detection_results_json: om模型推理的结果文件

