from mindspore.ops import operations as P
import mindspore.nn as nn


class XYLoss(nn.Cell):
    def __init__(self):
        super(XYLoss, self).__init__()
        self.cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = P.ReduceSum()

    def construct(self, object_mask, box_loss_scale, predict_xy, true_xy):
        xy_loss = object_mask * box_loss_scale * self.cross_entropy(predict_xy, true_xy)
        xy_loss = self.reduce_sum(xy_loss, ())
        return xy_loss


class WHLoss(nn.Cell):
    def __init__(self):
        super(WHLoss, self).__init__()
        self.square = P.Square()
        self.reduce_sum = P.ReduceSum()

    def construct(self, object_mask, box_loss_scale, predict_wh, true_wh):
        wh_loss = object_mask * box_loss_scale * 0.5 * P.Square()(true_wh - predict_wh)
        wh_loss = self.reduce_sum(wh_loss, ())
        return wh_loss


class ConfidenceLoss(nn.Cell):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = P.ReduceSum()

    def construct(self, object_mask, predict_confidence, ignore_mask):
        confidence_loss = self.cross_entropy(predict_confidence, object_mask)
        confidence_loss = object_mask * confidence_loss + (1 - object_mask) * confidence_loss * ignore_mask
        confidence_loss = self.reduce_sum(confidence_loss, ())
        return confidence_loss


class ClassLoss(nn.Cell):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = P.ReduceSum()

    def construct(self, object_mask, predict_class, class_probs):
        class_loss = object_mask * self.cross_entropy(predict_class, class_probs)
        class_loss = self.reduce_sum(class_loss, ())
        return class_loss

