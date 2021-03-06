# -*- coding: utf-8 -*-
"""demo.py
"""

import torch

def build_model(arch, loc):
    # 请自定义模型并加载预训练模型
    import models as models
    model = models.__dict__[arch]()
    model = model.to(loc)
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
    loc = 'npu:0'
    torch.npu.set_device(loc)

    # 1.获取原始数据
    raw_data = get_raw_data()

    # 2.构建模型并加载权重
    # 获取checkpoint.pth方式：修改script/run_1p.sh脚本的参数epochs为10，执行bash script/run_1p.sh，训练10个epoch后模型自动生成
    model = build_model(arch, loc)
    ckpt = torch.load("checkpoint.pth", map_location=loc)

    try:
        # for normal model save
        model.load_state_dict(ckpt['state_dict'])
    except:
        # for ddp's model save
        state_dict_ddp = ckpt['state_dict']
        state_dict = {}
        for key, value in state_dict_old.items():
            key = key[7:]
            state_dict[key] = value
        model.load_state_dict(state_dict)

    # 3.预处理
    input_tensor = pre_process(raw_data)
    input_tensor = input_tensor.to(loc)

    # 4. 执行forward
    output_tensor = model(input_tensor)

    # 5. 后处理
    result = post_process(output_tensor)

    # 6. 打印
    print(result)