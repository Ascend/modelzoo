1、【目的】
验证在910训练出来的tf resnet50模型，到310上推理精度的一致性

2、【开发代码路径】
https://gitlab.huawei.com/pmail_turinghava/training_shop/tree/master/03-code/Resnet50_TF/01-generalization/01-inference/02-offlineInfer

3、【环境安装依赖】
1) driver、firmware、acllib、atc、opp、toolkit分包按默认路径安装完成
2) 环境变量    LD_LIBRARY_PATH有aclib、driver、atc、opp的lib库路径，例如：
   LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64:/usr/local/Ascend/atc/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/toolkit/lib64:/usr/local/Ascend/add-ons
   环境变量ASCEND_OPP_PATH有opp的路径，例如：
   ASCEND_OPP_PATH=/usr/local/Ascend/atc/opp
   环境变量PATH有分包atc相关路径，例如：
   PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/usr/local/python3.7.5/bin:/usr/local/Ascend/atc/ccec_compiler/bin:/usr/local/Ascend/atc/bin:/usr/local/Ascend/toolkit/tools/ide_daemon/bin
   环境变量PYTHONPATH有atc相关python路径，例如：
   PYTHONPATH=:/usr/local/Ascend/atc/python/site-packages/te:/usr/local/Ascend/atc/python/site-packages/topi:/usr/local/Ascend/atc/python/site-packages/auto_tune.egg:/usr/local/Ascend/atc/python/si
3)  python库numpy，pip3 install numpy
4)  python库PIL，pip3 install pillow
5)  python库cv2,pip3 install cv2
6)  python库pandas，pip3 install pandas
	可能出现的问题：ModuleNotFoundError: No module named '_bz2',若出现该报错请参考：https://www.cnblogs.com/ikww/p/11544020.html
7）tensorflow， pip3 install tensorflow==1.15.0

4、【App进入】
cd  **/02-offlineInfer

5、【模型和数据集准备】
机器地址：10.137.55.145 autotest/autotest
1）【模型路径】
/Integrate_automated_testdata/model/Davinci_Om/Public/910to310

2）【图片】
/Integrate_automated_testdata/dataset/Public/910to310

3）【pb文件路径及转换命令】
resent pb文件路径：/Integrate_automated_testdata/model/Original_Model/Public/tensorflow/910to310
注：带dvpp后缀的pb指的是在训练上使用dvpp做预处理; 带tf后缀的pb指的是在训练上使用tensorflow做预处理

atc命令：
atc --model=resnet50_910to310_tf.pb --framework=3 --output=resnet50_tf_aipp_b1_fp16_input_fp32_output_fp32 --output_type=FP32 --soc_version=Ascend310 --input_shape="input_data:1,224,224,3" --insert_op_conf=test_aipp.cfg
atc --model=resnet50_910to310_tf.pb --framework=3 --output=resnet50_tf_noaipp_b1_fp16_input_fp32_output_fp32 --output_type=FP32 --soc_version=Ascend310 --input_shape="input_data:1,224,224,3"
atc --model=resnet50_910to310_dvpp.pb --framework=3 --output=resnet50_dvpp_aipp_b1_fp16_input_fp32_output_fp32 --output_type=FP32 --soc_version=Ascend310 --input_shape="input_data:1,224,224,3" --insert_op_conf=test_aipp.cfg
atc --model=resnet50_910to310_dvpp.pb --framework=3 --output=resnet50_dvpp_noaipp_b1_fp16_input_fp32_output_fp32 --output_type=FP32 --soc_version=Ascend310 --input_shape="input_data:1,224,224,3"


6、【模型、图片拷贝】
可将归档的模型直接拷贝到./model/resnet/下，也可以将转好的模型拷贝到/model/resnet 下
scp -r autotest@10.137.55.145:/Integrate_automated_testdata/model/Davinci_Om/Public/910to310/resnet50-tf/* ./model/resnet/

scp -r autotest@10.137.55.145:/Integrate_automated_testdata/dataset/Public/910to310/resnet/ImageNet2012_50000 ./datasets/resnet/
scp -r autotest@10.137.55.145:/Integrate_automated_testdata/dataset/Public/910to310/resnet/ImageNet2012_50000_224_224_RGB_bin_fp32_tf ./datasets/resnet/
scp -r autotest@10.137.55.145:/Integrate_automated_testdata/dataset/Public/910to310/resnet/ImageNet2012_50000_224_224_RGB_bin_fp32_opencv ./datasets/resnet/



7、【App编译】
1)chmod 755 build.sh
2)执行命令 ./build.sh


8、【App运行】
执行以下命令：
cd scripts/
chmod 777 *

resnet50执行样例：
cd script/
chmod +x benchmark_tf.sh
./benchmark_tf.sh --batchSize=1 --modelType=resnet50 --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --shape=224 --deviceId=0 --modelPath=../../model/resnet/resnet50_tf_aipp_b1_fp16_input_fp32_output_fp32.om --dataPath=../../datasets/resnet/ImageNet2012_50000/
./benchmark_tf.sh --batchSize=1 --modelType=resnet50 --imgType=rgb --precision=fp16 --outputType=fp32 --useDvpp=0 --shape=224 --deviceId=0 --modelPath=../../model/resnet/resnet50_tf_noaipp_b1_fp16_input_fp32_output_fp32.om --dataPath=../../datasets/resnet/ImageNet2012_50000_224_224_RGB_bin_fp32_tf

注：带aipp的模型对应的数据集为ImageNet2012_50000; 
    不带aipp的模型对应数据集为ImageNet2012_50000_224_224_RGB_bin_fp32_tf和ImageNet2012_50000_224_224_RGB_bin_fp32_opencv，tf后缀的数据集使用tensorflow做预处理，opencv后缀的数据集使用opencv做预处理。
 
    --batchSize           Data number for one inference
    --modelType           model type(resnet50/yolov3/inceptionV3/2D_lung/bert)
    --imgType             input image format type(rgb/yuv/raw, default is yuv)
    --precision           precision type(fp16/fp32/int8, default is fp16)
    --outputType          inference output type(fp32/fp16/int8, default is fp32)
    --useDvpp             (0-no dvpp,1--use dvpp, default is 0)
    --deviceId            running device ID(>=0 <=7, default is 0)
    --shape               (only for yolov3 model input shape, support 416、720p、1080p, default is 416)
    --loopNum             Run number,default is 1
    --framework           To distinguish the model type(Caffe,MindSpore,TensorFlow),default is tensorflow
    --inputType           The input data type,default is fp32
    --modelPath           (../../model/resnet/resnet50_tf_aipp_b1_fp16_input_fp32_output_fp32.om)
    --dataPath            (../../datasets/resnet/ImageNet2012_50000/)
    -h/--help             Show help message


9、【输出数据】
scripts/data.csv
scripts/data.txt
