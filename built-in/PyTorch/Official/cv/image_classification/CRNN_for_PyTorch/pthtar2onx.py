# -*- coding: utf-8 -*-

import torch
import crnn
import torch.onnx
import torch._utils
from collections import OrderedDict


pth_file_path = "checkpoint_6_acc_0.7887.pth"


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k,v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(pth_file_path):
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = crnn.CRNN(32, 1, 37, 256)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 1, 32, 100)
    import onnx
    dynamic_axes = {"actual_imput_1": {0: "-1"}, "output1": {1: "-1"}}
    print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
    torch.onnx.export(model, dummy_input, "crnn_npu_dy.onnx", input_names = input_names, dynamic_axes = dynamic_axes,
                      output_names = output_names, opset_version=11)


if __name__ == "__main__":
    convert(pth_file_path)
