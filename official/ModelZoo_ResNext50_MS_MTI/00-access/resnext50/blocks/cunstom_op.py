import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype


class GlobalAvgPooling(nn.Cell):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(True)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
    
    def construct(self, x):
        x = self.mean(x, (2, 3)) 
        b, c, _, _ = self.shape(x)
        x = self.reshape(x, (b, c))
        return x


class SEBlock(nn.Cell):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()

        self.avg_pool = GlobalAvgPooling()
        self.fc1 = nn.Dense(channel, channel // reduction)
        self.relu = P.ReLU()
        self.fc2 = nn.Dense(channel // reduction, channel)
        self.sigmoid = P.Sigmoid()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.sum = P.Sum()
        self.cast = P.Cast()
     
    def construct(self, x):
        b, c, h, w = self.shape(x)
        y = self.avg_pool(x)

        y = self.reshape(y, (b, c))
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = self.reshape(y, (b, c, 1, 1))
        return x * y

class GroupConv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad_mode="pad", pad=0, groups=1, has_bias=False):
        super(GroupConv, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups
        self.convs = nn.CellList()
        self.op_split = P.Split(axis=1, output_num=self.groups)
        self.op_concat = P.Concat(axis=1)
        self.cast = P.Cast()
        for _ in range(groups):
            self.convs.append(nn.Conv2d(in_channels//groups, out_channels//groups, kernel_size=kernel_size, stride=stride, has_bias=has_bias,
                     padding=pad, pad_mode=pad_mode, group=1))

    def construct(self, x):
        features = self.op_split(x)
        outputs = ()
        for i in range(self.groups):
            outputs = outputs + (self.convs[i](self.cast(features[i], mstype.float32)),)
        out = self.op_concat(outputs)
        return out