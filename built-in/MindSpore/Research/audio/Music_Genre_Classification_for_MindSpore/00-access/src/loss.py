from mindspore.ops import operations as P
from mindspore import nn
import mindspore as ms


class BCELoss(nn.Cell):
    def __init__(self, record=None):
        super(BCELoss, self).__init__(record)
        self.sm_scalar = P.ScalarSummary()
        self.cast = P.Cast()
        self.record = record
        self.weight = None
        self.bce = P.BinaryCrossEntropy()

    def construct(self, input, target):
        target = self.cast(target, ms.float32)
        loss = self.bce(input, target, self.weight)
        if self.record:
            self.sm_scalar("loss", loss)
        return loss