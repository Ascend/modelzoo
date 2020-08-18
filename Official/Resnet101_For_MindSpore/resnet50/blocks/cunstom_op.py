import mindspore.nn as nn
from mindspore.ops import operations as P



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
