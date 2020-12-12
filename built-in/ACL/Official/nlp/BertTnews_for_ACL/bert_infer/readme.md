1、【目的】
验证各种场景网络推理的测试用例的实现和结果比对。当前已实现同步推理，异步推理，多输入，动态AIPP等场景

2、【环境安装依赖】
1) 参考驱动和开发环境安装指南 完成1910 推理环境的安装

3、【demo编译】
Host侧运行：chmod +x build_host.sh
./build_host.sh
当前out目录下，已编译好了适用于1910 4.14.0-115.el7a.0.1.aarch64 形态的可执行文件，可以直接使用
如果是x86的环境，需要执行编译脚本重新编译执行

CTRLCPU运行：chmod +x build_device.sh
  ./build_device.sh

4、【json文件配置说明】
见数据字典文件：Inference_API.xlsx


5、【App运行】
执行以下命令：
(1)cd out/
./benchmark json文件路径
例如：./benchmark $JSONFILE

(2)CTRL CPU开放运行
scp -r out/ src/ datasets/ model/ HwHiAiUser@192.168.1.199:/home/HwHiAiUser/
ssh HwHiAiUser@192.168.1.199
cd /home/HwHiAiUser/out/
./benchmark $JSONFILE

6、【输出日志路径】
out/ACL_testcase.log

7、【输出数据】
性能和精度输出文件在out文件夹下，性能输出文件名以“perform_static”开头的txt文件，
精度输出以“precision_result”开头的txt文件，例如：
perform_static_dev_1_chn_0.txt，是运行在device 1上第0路推理的性能统计文件
precision_result_dev_1_chn_0.txt，是运行在device 1上第0路推理的精度计算结果


