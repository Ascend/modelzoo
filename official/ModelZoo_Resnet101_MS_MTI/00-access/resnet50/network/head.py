import mindspore.nn as nn
from resnet50.blocks.cunstom_op import GlobalAvgPooling

__all__ = ['CommonHead']

class CommonHead(nn.Cell):
    def __init__(self, num_classes, out_channels):
        super(CommonHead, self).__init__()
        self.avgpool = GlobalAvgPooling()
        self.fc = nn.Dense(out_channels, num_classes, has_bias=True).add_flags_recursive(fp16=True)

    def construct(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        return x