# Roma

## Roma简介

Roma库用于Vega程序运行在云道上的非侵入式的辅助库，主要提供如下功能：

1. 根据输入的区域信息，设置缺省路径。
2. 支持向云道S3上传和下载数据。
3. 获取云道的集群信息，用于构建集群。

**说明：**

**Roma库默认安装在Vega提供的云道镜像中，不对内和对外提供源码和安装包。**

## 使用方法

在Vega运行前，引入Roma：

```python
import vega
from roma import init_env


if __name__ == "__main__":
    init_env("hn1-y")
    vega.run("vega.yml")
```

目前支持的区域有：

| 区域 | 区域标识 |
| --- | --- |
| 北京4 绿区 | bj4-g |
| 北京4 黄区 | bj4-y |
| 华北1 绿区 | hb1-g |
| 华北1 黄区 | hb1-y |
| 华南1 黄区 | hn1-y |
