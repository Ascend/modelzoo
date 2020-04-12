import mindspore.nn as nn
from mindspore.ops import operations as P
from vgg16.utils.var_init import default_recurisive_init, KaimingNormal
from mindspore.common import initializer as init
import math
cfgs = {
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

class Vgg(nn.Cell):
    def __init__(self, cfg, batch_norm=False):
        # Important: When choose vgg, batch_size should <=64, otherwise will cause unknown error
        super(Vgg, self).__init__()
        self.layers = self._make_layer(cfg, batch_norm=batch_norm)

    def construct(self, x):
        x = self.layers(x)
        return x

    def _make_layer(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=v,
                                kernel_size=3,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        return nn.SequentialCell(layers)

class VGGHead(nn.Cell):
    def __init__(self, num_classes):
        super(VGGHead, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        # TODO add adaptive-pooling layer before nn.Dense(512*7*7, 4096). Only support 224*224 input right now.
        self.classifier = nn.SequentialCell([
            nn.Dense(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, num_classes)
        ])

    def construct(self, x):
        b, c, h, w = self.shape(x)
        # Only support w==h==7
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
                cell.weight.default_input = init.initializer(KaimingNormal(a=math.sqrt(5), mode='fan_out', nonlinearity='relu'), cell.weight.default_input.shape(), cell.weight.default_input.dtype())
                if cell.bias is not None:
                    cell.bias.default_input = init.initializer('zeros', cell.bias.default_input.shape(), cell.bias.default_input.dtype())
            elif isinstance(cell, nn.Dense):
                cell.weight.default_input = init.initializer(init.Normal(0.01), cell.weight.default_input.shape(), cell.weight.default_input.dtype())
                if cell.bias is not None:
                    cell.bias.default_input = init.initializer('zeros', cell.bias.default_input.shape(), cell.bias.default_input.dtype())       

    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

class VGG(ImageClassificationNetwork):
    def __init__(self, backbone_name, num_classes):
        self.backbone_name = backbone_name
        backbone = Vgg(cfg=cfgs['16'], batch_norm=False)
        head = VGGHead(num_classes=num_classes)
        super(VGG, self).__init__(backbone, head)


def get_network(backbone_name, num_classes):
    try:
        if backbone_name in ['vgg16']:
            return VGG(backbone_name, num_classes)
    except:
        raise NotImplementedError('not implement {}'.format(backbone_name))