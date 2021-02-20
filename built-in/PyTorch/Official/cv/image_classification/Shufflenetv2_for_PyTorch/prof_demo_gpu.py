# -*- coding: utf-8 -*-
"""prof_demo_gpu.py
"""

import torch
import torch.nn as nn
import torch.optim as optim


def build_model(arch):
    # 请自定义模型并加载预训练模型
    import models as models
    model = models.__dict__[arch]()
    model = model.cuda()
    model.eval()  # 注意设置eval模式
    return model


def get_raw_data():
    # 请自定义获取数据方式，请勿将原始数据上传至代码仓
    from PIL import Image
    from urllib.request import urlretrieve
    IMAGE_URL = 'https://bbs-img.huaweicloud.com/blogs/img/thumb/1591951315139_8989_1363.png'
    urlretrieve(IMAGE_URL, 'tmp.jpg')
    img = Image.open("tmp.jpg")
    img = img.convert('RGB')
    return img


def pre_process(raw_data):
    # 请自定义模型预处理方法
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_list = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    input_data = transforms_list(raw_data)
    return input_data.unsqueeze(0)


def post_process(output_tensor):
    # 请自定义后处理方法
    return torch.argmax(output_tensor, 1)


if __name__ == '__main__':
    arch = 'shufflenet_v2_x1_0'

    # 1.获取原始数据
    raw_data = get_raw_data()

    # 2.构建模型并加载权重
    model = build_model(arch)

    # 3.预处理
    input_tensor = pre_process(raw_data)
    input_tensor = input_tensor.cuda()

    # 4. 执行forward+profiling
    with torch.autograd.profiler.profile(record_shapes=True, use_cuda=True) as prof:
        output_tensor = model(input_tensor)
        target = torch.randn(output_tensor.size()) # 用随机值代替
        target = target.cuda()
        criterion = nn.MSELoss().cuda()
        loss = criterion(output_tensor, target) # 使用均方误差损失
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(prof.key_averages().table())
    prof.export_chrome_trace("shufflenet_v2_gpu.prof")

    # 5. 后处理
    result = post_process(output_tensor)

    # 6. 打印
    print(result)