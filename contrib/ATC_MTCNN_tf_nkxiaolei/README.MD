1、原始模型https://github.com/AITTSMD/MTCNN-Tensorflow, 下载对应的ckpt，转成pb模型

2、转om模型，obs链接：obs://nkxiaolei88/ATC Model/MTCNN/MTCNN_onet.om 、MTCNN_pnet.om 、MTCNN_rnet.om

3、将下载的om文件放在model文件夹

4、使用nsame推理工具，参考如下命令，发起推理性能测试： ./msame --model model/MTCNN_onet.om --output output/ --loop 100
其他模型以此类推

onet:
1Batch，shape:1x48x48x3，不带AIPP，平均推理性能0.76ms

pnet:
1Batch，shape:1x224x224x3，不带AIPP，平均推理性能1.19ms

rnet:
1Batch，shape:1x24x24x3，不带AIPP，平均推理性能0.65ms

5、精度测试：暂未开始