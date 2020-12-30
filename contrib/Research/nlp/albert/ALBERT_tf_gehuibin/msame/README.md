# ALBERT msame 推理
## 概述
推理后的结果精度基本和训练的模型保持一致，具体推理详细见wiki [Albert-squad离线推理案例
](https://gitee.com/ascend/modelzoo/wikis/Albert-squad%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B?sort_id=3269923)
|  | F1| EM |
| :-----| ----: | :----: |
| albert_base(Ascend) | **82.4**| **79.4**|
| albert_base(msame推理) | 82.4 | 79.4 |
| albert_large(Ascend) | **85.5** | **82.5** |
| albert_large(msame推理) | 85.5 | 82.5 |


## 代码路径解释

```shell
msame
└─ 
  ├─README.md
  ├─input_ids .bin格式输入文件夹
  	├─00000.bin
  	└─...
  ├─input_mask .bin格式输入文件夹
  	├─00000.bin
  	└─...
  ├─p_masks .bin格式输入文件夹
  	├─00000.bin
  	└─...
  ├─segment_ids .bin格式输入文件夹
  	├─00000.bin
  	└─...
  ├─pb_albert_base_v2 albert base推理结果文件夹
  	├─model.pb
  	├─albert.pb
  	├─albert.om
        ├─output 推理后的结果
  	    ├─00000.txt
  	    └─...
  ├─pb_albert_large_v2 albert large推理结果文件夹
  	├─model.pb
  	├─albert.pb
  	├─albert.om
        ├─output 推理后的结果
  	    ├─00000.txt
  	    └─...
  ├─squad_v2 存放数据目录
  	├─train-v2.0.json 数据源文件
  	├─dev-v2.0.json 数据源文件
  	├─train.tfrecord 根据train-v2.0.json生成的文件
  	├─dev.tfrecord 根据dev-v2.0.json生成的文件
  	├─pred_left_file.pkl 根据dev-v2.0.json生成的文件

  ├─run.sh 启动脚本
```

---

## 准备数据和模型

首先需要把输入的数据都转换为.bin格式的数据，若需要转换，则需要上层文件夹的squad_v2文件夹作为原数据，可在上层目录夹的readme中下载

```
./convert_bin.sh
```

这里提供了转换后的数据，将input_ids.tar.gz,p_masks.tar.gz,segment_ids.tar.gz以及input_mask.tar.gz解压后得到文件夹

转换完成后，执行以下命令完成推理以及精度测量

```
./run.sh
```
run.sh中包含有几个命令
其中mode有两个选项base和large，对应两个模型的结果

```
mode=base 
input_dir=../squad_v2
input_id_path=./input_ids
input_mask_path=./input_masks
segment_id_path=./segment_ids
p_mask_path=./p_masks
idx_file=./idx.txt
config=../albert_${mode}_v2/albert_config.json
output_dir=./pb_albert_${mode}_model
ckpt_dir=../output_${mode}_v2

冻结图
python3.7 -m freeze_graph \
	--albert_config=${config} \
	--output_dir=${output_dir} \
	--ckpt_dir=${ckpt_dir}

atc转换
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=${output_dir}/albert.pb --framework=3 --output=${output_dir}/albert --soc_version=Ascend910 \
	--input_shape="input_ids:1,384;input_mask:1,384;segment_ids:1,384;p_mask:1,384" \
	--log=info \
	--out_nodes="end_logits/TopKV2:0;end_logits/TopKV2:1;end_logits/Reshape_1:0;end_logits/Reshape_2:0;answer_class/Squeeze:0"
msame推理
./msame --model ${output_dir}/albert.om --input ${input_id_path},${input_mask_path},${segment_id_path},${p_mask_path} --output ${output_dir}/output --outfmt TXT

测量精度
python3.7 evaluate.py --input_dir=$input_dir \
	--idx_file=$idx_file \
	--output_dir=$output_dir
```

测得的结果如下：
albert base的推理速度：

![输入图片说明](https://images.gitee.com/uploads/images/2020/1230/112859_52ca3260_8144926.png "sbase.png")

albert base的推理精度：

![输入图片说明](https://images.gitee.com/uploads/images/2020/1230/112924_a0fffc0b_8144926.png "jbase.png")

albert large的推理速度：

![输入图片说明](https://images.gitee.com/uploads/images/2020/1230/112947_dc22f7f3_8144926.png "slarge.png")

albert large的推理精度

![输入图片说明](https://images.gitee.com/uploads/images/2020/1230/113008_f0c53721_8144926.png "jlarge.png")


上述模型转换以及推理后的结果后的均可以在链接中下载[百度云链接](https://pan.baidu.com/s/1Wx2CtrutYPhR2pY2HL7ejg) 提取码: 8g4u

