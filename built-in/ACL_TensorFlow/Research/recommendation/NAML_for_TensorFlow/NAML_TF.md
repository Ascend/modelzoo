demo编译方法：
1、host机器安装分包，安装路径为默认路径/usr/local/Ascend
2、进入naml_tf_test/demo目录，执行./build.sh进行编译
3、编译完成后，进入naml_tf_test/demo/run/out获取可执行程序main

推理执行方法：
./main naml_fp16_batch16.om  input/  16

naml_fp16_batch16.om    ----模型名字
input/     ----数据集
16         ----batchsize数

