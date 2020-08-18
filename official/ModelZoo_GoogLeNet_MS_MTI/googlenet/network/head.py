import mindspore.nn as nn
from googlenet.blocks.cunstom_op import GlobalAvgPooling

__all__ = ['GoogLeNetHead']


class GoogLeNetHead(nn.Cell):
    def __init__(self, num_classes, out_channels, dropout_p=0.8, is_train=True):
        super(GoogLeNetHead, self).__init__()
        self.avgpool = GlobalAvgPooling()
        self.dropout = nn.Dropout(keep_prob=dropout_p)
        self.fc = nn.Dense(out_channels, num_classes, has_bias=False)
        self.is_train = is_train

    def construct(self, x):
        x = self.avgpool(x)
        if self.is_train:
            x = self.dropout(x)
        x = self.fc(x)
        return x