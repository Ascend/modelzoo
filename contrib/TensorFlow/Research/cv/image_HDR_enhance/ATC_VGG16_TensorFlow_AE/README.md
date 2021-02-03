
图像HDR增强系统在华为200dk平台的部署代码说明



文件说明：

1. input: 测试图像文件夹
2. target：目标图像文件夹

3. model：离线模型文件夹，存放离线模型文件model.om

4. hdr_main.py: 模型推理主程序



运行流程：（环境：华为开发板200dk,并且已经部署好相关python3环境）

1. 在本地ubuntu18.04的Mindstudio开发环境中对模型训练得到的model.pb文件进行模型转换，得到model.om文件，放到model/文件夹下

2. 将整个hdr_200dk文件夹拷贝到华为开发板200dk的/home/HwHiAiUser/HIAI_PROJECTS/文件夹下，命令：scp -r PycharmProjects/hdr_200dk  HwHiAiUser@192.168.0.101:/home/HwHiAiUser/HIAI_PROJECTS/，文件夹位置和开发板ip地址根据实际情况填写

3. 登录华为200dk开发板，进入/home/HwHiAiUser/HIAI_PROJECTS/hdr_200dk/文件夹下，运行hdr_main.py文件，命令：python3 hdr_main.py，得到输出图像，在output文件夹下

4. 将output文件夹拷贝回本地电脑PycharmProjects/hdr_200dk/文件夹下，查看图像的HDR增强效果，命令：scp -r HwHiAiUser@192.168.0.101:/home/HwHiAiUser/HIAI_PROJECTS/hdr_200dk/output PycharmProjects/hdr_200dk/


