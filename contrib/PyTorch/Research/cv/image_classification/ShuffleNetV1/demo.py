# -*- coding: utf-8 -*-
"""demo.py
"""

import torch
import torch.npu
loc = 'npu:0'

def build_model():
    from network import ShuffleNetV1
    md = ShuffleNetV1(group=3, model_size='1.0x')
    md = md.to(loc)
    md.eval()
    pretrained = torch.load('trainedmodel.pth.tar', map_location=loc) # change this to the filename of the trained model!

    old_dict = pretrained['state_dict']
    state_dict = {}
    for key, value in old_dict.items():
        key = key[7:]
        state_dict[key] = value

    md.load_state_dict(state_dict)
    return md


def get_raw_data():
    from PIL import Image
    from urllib.request import urlretrieve
    IMAGE_URL = 'https://bbs-img.huaweicloud.com/blogs/img/thumb/1591951315139_8989_1363.png'
    urlretrieve(IMAGE_URL, 'tmp.jpg')
    img = Image.open("tmp.jpg")
    img = img.convert('RGB')
    return img


def pre_process(rd):
    '''
    from torchvision import transforms
    transforms_list = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    input_data = transforms_list(rd)
    return input_data.unsqueeze(0)
    '''
    from torchvision import transforms
    transforms_list = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    input_data = transforms_list(rd)
    return input_data.unsqueeze(0)


def post_process(out):
    return torch.argmax(out, 1)


if __name__ == '__main__':
    torch.npu.set_device(loc)
    # 1. 获取原始数据
    raw_data = get_raw_data()

    # 2. 构建模型
    model = build_model()

    # 3. 预处理
    input_tensor = pre_process(raw_data)
    input_tensor = input_tensor.to(loc)
    # 4. 执行forward
    output_tensor = model(input_tensor)
    output_tensor = output_tensor.cpu()
    # 5. 后处理
    result = post_process(output_tensor)
    # 6. 打印
    print(result)
