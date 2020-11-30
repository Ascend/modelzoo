import torch
from image_classfication import resnet
from image_classfication.resnet import resnet_version
import torch.onnx

from collections import OrderedDict

def proc_node_module(checkpoint,AttrName):
    new_state_dict = OrderedDict()
    for k,v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict
def convert():
    checkpoint = torch.load("./resnet50checkpoint.pth.tar", map_location='cpu')
    checkpoint['state_dict'] = proc_nodes_module(checkpoint, 'state_dict')
    model = resnet.build_resnet("resnet50","classic")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval();
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16,3,224,224)
    torch.onnx.export(model,dummy_input,"resnet50_npu_16.onnx", input_names = input_names, output_names = output_names, opset_version=11)
if __name__ == "__main__":
    convert()