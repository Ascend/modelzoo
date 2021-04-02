### 1. 模型转换

#### 1.1 TensorFlow 模型转 pb 文件

预训练的 TensorFlow 模型文件：[下载](https://unpairedsr.obs.cn-north-4.myhuaweicloud.com:443/share/output.zip?AccessKeyId=GB2XLA5SI9AWHCYX2NLT&Expires=1648392810&Signature=3DjXn0/os1rVt3FM4FOb3DsNmXA%3D)

将 TensorFlow 模型文件转换为 pb 文件：

```
python ckpt2pb.py --ckpt_dir /path/to/ckpt/files --pb_dir /path/to/pb/files
```

参数说明：

* ckpt_dir：TensorFlow 模型文件所在目录
* pb_dir：pb 文件保存目录

#### 1.2 pb 文件转 om 模型

使用 atc 工具将 pb 文件转换为 om 模型文件：

```
atc --model=/path/to/unpairedsr.pb --framework=3 --input_shape="input:1,16,16,3" --output=unpairedsr --soc_version=Ascend310
```

转换后得到文件：unpairedsr.om

### 2. 模型推理

#### 2.1 数据预处理

原始数据文件：[下载](https://unpairedsr.obs.cn-north-4.myhuaweicloud.com:443/share/data.zip?AccessKeyId=GB2XLA5SI9AWHCYX2NLT&Expires=1648392862&Signature=cnOGtNnH523NR4H64BB6sfNbg08%3D)

将**测试数据**转换为 bin 文件

```
python img2bin.py --input /path/to/test/data --output /path/to/bin/files
```

参数说明：

* input：测试数据所在目录
* output：bin文件保存目录，其下有两个子目录，lr表示低分辨率，hr表示高分辨率

#### 2.2 加载模型推理

使用 [msame](https://gitee.com/ascend/tools/tree/master/msame) 工具推理：

```
./msame --model /path/to/om/file --input /path/to/lr/bin --output /path/to/model/output
```

参数说明：

* model：[1.2](#1.2-pb-文件转-om-模型) 中 om 模型所在路径
* input：[2.1](#2.1-数据预处理) 中的 lr 目录
* output：模型输出目录

#### 2.3 精度计算

运行：

```
python evaluate.py --real_hr /path/to/real/hr/bin --fake_hr /path/to/predicted/hr/bin
```

参数说明：

* real_hr：真实高分辨率图片，[2.1](#2.1-数据预处理) 中的 hr 目录
* fake_hr：模型生成的高分辨率图片，[2.2](#2.2 -加载模型推理) 中的模型输出

### 3. 性能与精度对比

在 GPU 环境上，使用 pb 文件测试精度与运行时间：

```
python evaluate_gpu.py --pb_path /path/to/pb/file --data_dir /path/to/test/data --gpu device_id
```

参数说明：

* pb_path：pb 文件路径
* data_dir：测试数据目录
* gpu：设备编号（默认值为 0）

**注意**：此处使用的 pb 文件即 [1.1](#1.1-TensorFlow-模型转-pb-文件) 中得到的 pb 文件，由 Ascend 910 上训练得到的 TensorFlow 模型文件转换得到。

结果对比如下表：

|                        |  Tesla T4  | Ascend 310 |
| :--------------------: | :--------: | :--------: |
|          PSNR          | 19.8656 dB | 19.8658 dB |
| Inference average time |  12.72 ms  |  3.61 ms   |

### 4. 资源下载

* pb 文件：[下载](https://unpairedsr.obs.cn-north-4.myhuaweicloud.com:443/share/offline%20inference/unpairedsr.pb?AccessKeyId=GB2XLA5SI9AWHCYX2NLT&Expires=1647782002&Signature=9tKBOQXsAhV/RHQa/LSvOzFcm7E%3D)
* om 文件：[下载](https://unpairedsr.obs.cn-north-4.myhuaweicloud.com:443/share/offline%20inference/unpairedsr.om?AccessKeyId=GB2XLA5SI9AWHCYX2NLT&Expires=1648393179&Signature=FCZPlzExdO%2B64g7jFdVTuyk8avA%3D)
* 数据 bin 文件：[下载](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=WrkTg8PzzdAMSqfXzuGlEoPHJVkuOxg0zuDH1271OXQZDVQPHn/B4tObo9vCz5MUzmD1/1iG3/n44iZpfFQganJk7y0NZtrN+PeK7/unvLf0fr5hZR0VSqgFbG7OJI9bB44cxY9fy6vGJnWP4zaQG26E5nqBxMz7ljFWUs+PF9GW1RvPZCHHyadE1j2nzwDFmvCg+D3+LY8V1kqXG14X1sBkIjcufdk4LlK5xc4XllItXrbGVd+uFJFwUa9rINFY5sn9PPAhhl9Se4T0lyx2ngacCDVl7BkfE6BedLIydEXVeG9nq5VBaeZRkfTUKT0K8Dwmyh+SSXPeb90joPBINnYlBrHzptqGtpumK60tuiUBfAmrUztgKu88bdDieJDMLJpMqNr0uel6qCdtBOeNfXGFJk7/cuTCkLHctQ282paAwjJCa2wNmyW4Zm5upfW10Uoo/2SqTA6as2gDcofIQNM7/tyZbDzmgAlG5Xg1A1xSF/kubwz4FR14YzTtsfMBtu+rA+upjsrZxPq7k3MnbZirH9j4KBjMJFdkMn1q8mLKgqFF4jduYXHUXU4zpyAJsKu0YTLRwD8BEsMOvAujzI1e4o7eoxMczGJJXp1XPYQKrJcK/UTkSJ2JsAs+2t4JQ2E0ljYTzmQvm8xA/y/zHA==)（提取码：123456）