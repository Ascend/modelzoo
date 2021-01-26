### demo 文件夹内容说明
- pth转换onnx脚本: export_onnx.py
- ATC转换脚本 run.sh / auto_tune.sh，及aipp配置文件 aipp.conf
- benchmark 二进制文件: benchmark.x86_64
- 数据集信息 ImgPSENet.info 及二进制数据集信息 PSENet.info
- 二进制数据集预处理脚本: preprocess_psenet_pytorch.py
- 数据集标签: gt.zip
- 二进制后处理脚本: pth_bintotxt_nearest.py, pth_bintotxt_bilinear.py, pypse.py
- 精度评测脚本: Post-processing文件夹，script.py
