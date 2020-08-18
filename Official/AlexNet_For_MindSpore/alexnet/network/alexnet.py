import mindspore.nn as nn
from mindspore.ops import operations as P
from alexnet.utils.var_init import default_recurisive_init, KaimingNormal
from mindspore.common import initializer as init
import math

def conv11x11(in_channels, out_channels, stride=4, padding=0, has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="same")


def conv5x5(in_channels, out_channels, stride=1, padding=0, has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="same")


def conv3x3(in_channels, out_channels, stride=1, padding=0, has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="same")


class Alexnet(nn.Cell):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.conv1 = conv11x11(in_channels=3, out_channels=64, has_bias=True)
        self.relu1 = P.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.conv2 = conv5x5(in_channels=64, out_channels=192, has_bias=True)
        self.relu2 = P.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.conv3 = conv3x3(in_channels=192, out_channels=384, has_bias=True)
        self.relu3 = P.ReLU()
        self.conv4 = conv3x3(in_channels=384, out_channels=256, has_bias=True)
        self.relu4 = P.ReLU()
        self.conv5 = conv3x3(in_channels=256, out_channels=256, has_bias=True)
        self.relu5 = P.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

        self.out_channels = 256

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        return x

    def get_out_channels(self):
        return self.out_channels


class AlexnetHead(nn.Cell):
    def __init__(self, num_classes, phase):
        super(AlexnetHead, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        # TODO add adaptive-pooling layer before nn.Dense(512*6*6, 4096). Only support 224*224 input right now.
        dropout_ratio = 0.65
        if phase == 'test':
            dropout_ratio = 1.0
        self.classifier = nn.SequentialCell([
            nn.Dense(256*6*6, 4096, has_bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, 4096, has_bias=True),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, num_classes, has_bias=True)
        ])

    def construct(self, x):
        b, c, h, w = self.shape(x)
        x = self.reshape(x, (b, -1))
        x = self.classifier(x)
        return x
    

class ImageClassificationNetwork(nn.Cell):
    def __init__(self, backbone, head):
        super(ImageClassificationNetwork, self).__init__()
        self.backbone = backbone
        self.head = head  
        default_recurisive_init(self)
        self.custom_init_weight()
    
    def custom_init_weight(self):
        for name, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.default_input = init.initializer(KaimingNormal(a=math.sqrt(5), mode='fan_out', nonlinearity='relu'), cell.weight.default_input.shape(), cell.weight.default_input.dtype()).to_tensor()
                if cell.bias is not None:
                    cell.bias.default_input = init.initializer('zeros', cell.bias.default_input.shape(), cell.bias.default_input.dtype()).to_tensor()
            elif isinstance(cell, nn.Dense):
                cell.weight.default_input = init.initializer(init.Normal(0.01), cell.weight.default_input.shape(), cell.weight.default_input.dtype()).to_tensor()
                if cell.bias is not None:
                    cell.bias.default_input = init.initializer('zeros', cell.bias.default_input.shape(), cell.bias.default_input.dtype()).to_tensor()

    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

class ALEXNET(ImageClassificationNetwork):
    def __init__(self, backbone_name, num_classes, phase):
        self.backbone_name = backbone_name
        backbone = Alexnet()
        head = AlexnetHead(num_classes=num_classes, phase=phase)
        super(ALEXNET, self).__init__(backbone, head)
       

def get_network(backbone_name, num_classes, phase='train'):
    try:
        if backbone_name in ['alexnet']:
            return ALEXNET(backbone_name, num_classes, phase)
    except:
        raise NotImplementedError('not implement {}'.format(backbone_name))