文件作用说明：

- vgg16_ssd_pth2onnx.py，pth转换om脚本
- ATC转换脚本vgg16_ssd_atc.sh
- benchmark二进制文件
- 数据集预处理文件ssd_pth_preprocess.py
- ReadMe.md

推理端到端步骤：

1. 使用脚本vgg16_ssd_pth2onnx.py将pth文件导出为onnx文件

2. 下载权重文件到当前文件夹，
   下载源代码 git clone https://github.com/qfgaohao/pytorch-ssd 到当前文件夹，
   使用vgg_ssd_amct使用vgg_ssd_amct.sh脚本进行onnx模型量化，
   建议在data_mact文件夹内放4个32batch的由VOC2007数据集的图片转成的bin文件
   注意：脚本中导入sys.path.append(r"./pytorch-ssd")即是下载的源码

3. 运行vgg16_ssd_atc.sh脚本转换om模型
  可将--input_shape="actual_input_1:1,3,300,300" 改成想测试的shape，如 （16，3，300，300）测试16batch的onnx

4. 用ssd_pth_preprocess.py脚本处理数据集， 
   
   python3 ssd_pth_preprocess.py vgg16_ssd ./VOC2007/JPEGImages/ ./prep_bin
   第一个为预处理脚本，第二个为原图片文件，第三个为处理生成的bin文件路径
   
5. 使用get_info.py脚本获取预处理info文件

   python3.7 get_info.py bin ./prep_bin ./vgg16_ssd.info 300 300

6. ./benchmark.x86_64 -model_type=vision -om_path=vgg16_ssd_bs1.om -device_id=0 -batch_size=1 -input_text_path=vgg16_ssd.info -input_width=300 -input_height=300 -useDvpp=false -output_binary=true

   运行benchmark推理，结果保存在 ./result 目录下

7. python3 ssd_pth_postprocess.py ./VOC2007/ ./voc-model-labels.txt ./result/dumpOutput_device0/ ./eval_results/

   验证推理结果

​       

