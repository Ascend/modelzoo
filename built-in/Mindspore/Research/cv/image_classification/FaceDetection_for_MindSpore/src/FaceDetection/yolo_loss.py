# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Face detection loss."""
import numpy as np

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.nn.loss.loss import _Loss
from mindspore.nn import Dense, Cell
from mindspore import Tensor
from mindspore.common import dtype as mstype


class PtLinspace(Cell):
    def __init__(self):
        super(PtLinspace, self).__init__()
        self.TupleToArray = P.TupleToArray()

    def construct(self, start, end, steps):
        lin_x = ()
        step = (end - start + 1) / steps
        for i in range(start, end + 1, step):
            lin_x += (i,)
        lin_x = self.TupleToArray(lin_x)

        return lin_x


class MSELoss(_Loss):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.sum = P.Sum()
        self.mean = P.ReduceMean(keepdims=False)
        self.pow = P.Pow()
        self.sqrt = P.Sqrt()

    def construct(self, nembeddings1, nembeddings2):
        dist = nembeddings1 - nembeddings2
        dist_pow = self.pow(dist, 2.0)
        dist_sum = self.sum(dist_pow, 1)

        dist_sqrt = self.sqrt(dist_sum)
        loss = self.mean(dist_sqrt, 0)
        return loss


class YoloLoss(Cell):
    """ Computes yolo loss from darknet network output and target annotation.

    Args:
        num_classes (int): number of categories
        anchors (list): 2D list representing anchor boxes
        coord_scale (float): weight of bounding box coordinates
        no_object_scale (float): weight of regions without target boxes
        object_scale (float): weight of regions with target boxes
        class_scale (float): weight of categorical predictions
        thresh (float): minimum iou between a predicted box and ground truth for them to be considered matching
        seen (int): How many images the network has already been trained on.
    """
    def __init__(self, num_classes, anchors, anchors_mask, reduction=32, seen=0, coord_scale=1.0, no_object_scale=1.0,
                 object_scale=1.0, class_scale=1.0, thresh=0.5, head_idx=0.0):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors_mask)
        self.anchor_step = len(anchors[0])  # each scale has step anchors
        self.anchors = np.array(anchors, dtype=np.float32) / reduction  # scale every anchor for every scale
        self.tensor_anchors = Tensor(self.anchors, mstype.float32)
        self.anchors_mask = anchors_mask
        anchors_w = []
        anchors_h = []
        for i in range(len(anchors_mask)):
            anchors_w.append(self.anchors[self.anchors_mask[i]][0])
            anchors_h.append(self.anchors[self.anchors_mask[i]][1])
        self.anchors_w = Tensor(np.array(anchors_w).reshape(len(self.anchors_mask), 1))
        self.anchors_h = Tensor(np.array(anchors_h).reshape(len(self.anchors_mask), 1))

        self.reduction = reduction
        self.seen = seen
        self.head_idx = head_idx
        self.zero = Tensor(0)
        self.coord_scale = coord_scale
        self.no_object_scale = no_object_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.thresh = thresh

        self.info = {'avg_iou': 0, 'class': 0, 'obj': 0, 'no_obj': 0,
                     'recall50': 0, 'recall75': 0, 'obj_cur': 0, 'obj_all': 0,
                     'coord_xy': 0, 'coord_wh': 0}

        self.Shape = P.Shape()
        self.Reshape = P.Reshape()
        self.Sigmoid = P.Sigmoid()
        self.ZerosLike = P.ZerosLike()
        self.ScatterNd = P.ScatterNd()
        self.ScatterNdUpdate = P.ScatterNdUpdate()
        self.concat0 = P.Concat(0)
        self.concat0_2 = P.Concat(0)
        self.concat0_3 = P.Concat(0)
        self.concat0_4 = P.Concat(0)
        self.concat1 = P.Concat(1)
        self.concat1_2 = P.Concat(1)
        self.concat1_3 = P.Concat(1)
        self.concat1_4 = P.Concat(1)
        self.concat2 = P.Concat(2)
        self.concat2_2 = P.Concat(2)
        self.concat2_3 = P.Concat(2)
        self.concat2_4 = P.Concat(2)

        self.Tile = P.Tile()
        self.Transpose = P.Transpose()
        self.TupleToArray = P.TupleToArray()
        self.ScalarToArray = P.ScalarToArray()
        self.Cast = P.Cast()
        self.Exp = P.Exp()
        self.Sum = P.ReduceSum()
        self.Log = P.Log()
        self.TensorAdd = P.TensorAdd()
        self.RealDiv = P.RealDiv()
        self.Div = P.Div()
        self.SmoothL1Loss = P.SmoothL1Loss()
        self.Sub = P.Sub()
        self.Greater = P.Greater()
        self.GreaterEqual = P.GreaterEqual()
        self.Minimum = P.Minimum()
        self.Maximum = P.Maximum()
        self.Less = P.Less()
        self.OnesLike = P.OnesLike()
        self.Fill = P.Fill()
        self.Equal = P.Equal()
        self.BCE = P.SigmoidCrossEntropyWithLogits()
        self.CE = P.SoftmaxCrossEntropyWithLogits()
        self.DType = P.DType()
        self.PtLinspace = PtLinspace()
        self.OneHot = nn.OneHot(-1, self.num_classes, 1.0, 0.0)
        self.Squeeze2 = P.Squeeze(2)
        self.ArgMax = P.Argmax()
        self.ArgMaxWithValue1 = P.ArgMaxWithValue(1)
        self.ReduceSum = P.ReduceSum()
        self.Log = P.Log()
        self.GatherNd = P.GatherNd()
        self.Abs = P.Abs()
        self.Select = P.Select()
        self.IOU = P.IOU()

    def construct(self, output, coord_mask, conf_pos_mask, conf_neg_mask, cls_mask, t_coord, t_conf, t_cls, gt_list):
        """
        Compute Yolo loss.
        """
        output_d = self.Shape(output)
        num_batch = output_d[0]
        num_anchors = self.num_anchors

        num_classes = self.num_classes
        num_channels = output_d[1] / num_anchors
        height = output_d[2]
        width = output_d[3]
        output = self.Reshape(output, (num_batch, num_anchors, num_channels, height * width))
        coord_01 = output[:, :, :2]  # tx,ty
        coord_23 = output[:, :, 2:4]  # tw,th
        coord = self.concat2((coord_01, coord_23))
        conf = self.Squeeze2(output[:, :, 4:5, :])
        cls = output[:, :, 5:]
        cls = self.Reshape(cls, (num_batch*num_anchors, num_classes, height*width))
        perm = (0, 2, 1)
        cls = self.Transpose(cls, perm)
        cls_shp = self.Shape(cls)
        cls = self.Reshape(cls, (cls_shp[0] * cls_shp[1] * cls_shp[2] / num_classes, num_classes))

        lin_x = self.PtLinspace(0, width - 1, width)
        lin_x = self.Tile(lin_x, (height, ))
        lin_x = self.Cast(lin_x, mstype.float32)

        lin_y = self.PtLinspace(0, height - 1, height)
        lin_y = self.Reshape(lin_y, (height, 1))
        lin_y = self.Tile(lin_y, (1, width))
        lin_y = self.Reshape(lin_y, (self.Shape(lin_y)[0] * self.Shape(lin_y)[1], ))
        lin_y = self.Cast(lin_y, mstype.float32)

        anchor_w = self.anchors_w
        anchor_h = self.anchors_h
        anchor_w = self.Cast(anchor_w, mstype.float32)
        anchor_h = self.Cast(anchor_h, mstype.float32)
        coord_x = self.Sigmoid(coord[:, :, 0:1, :])
        pred_boxes_0 = self.Squeeze2(coord_x) + lin_x
        shape_pb0 = self.Shape(pred_boxes_0)
        pred_boxes_0 = self.Reshape(pred_boxes_0, (shape_pb0[0] * shape_pb0[1] * shape_pb0[2], 1))
        coord_y = self.Sigmoid(coord[:, :, 1:2, :])
        pred_boxes_1 = self.Squeeze2(coord_y) + lin_y
        shape_pb1 = self.Shape(pred_boxes_1)
        pred_boxes_1 = self.Reshape(pred_boxes_1, (shape_pb1[0] * shape_pb1[1] * shape_pb1[2], 1))
        pred_boxes_2 = self.Exp(self.Squeeze2(coord[:, :, 2:3, :])) * anchor_w
        shape_pb2 = self.Shape(pred_boxes_2)
        pred_boxes_2 = self.Reshape(pred_boxes_2, (shape_pb2[0] * shape_pb2[1] * shape_pb2[2], 1))
        pred_boxes_3 = self.Exp(self.Squeeze2(coord[:, :, 3:4, :])) * anchor_h
        shape_pb3 = self.Shape(pred_boxes_3)
        pred_boxes_3 = self.Reshape(pred_boxes_3, (shape_pb3[0] * shape_pb3[1] * shape_pb3[2], 1))

        pred_boxes_x1 = pred_boxes_0 - pred_boxes_2 / 2
        pred_boxes_y1 = pred_boxes_1 - pred_boxes_3 / 2
        pred_boxes_x2 = pred_boxes_0 + pred_boxes_2 / 2
        pred_boxes_y2 = pred_boxes_1 + pred_boxes_3 / 2
        pred_boxes_points = self.concat1_4((pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2))

        total_anchors = num_anchors * height * width
        mask_concat = None
        conf_neg_mask_zero = self.ZerosLike(conf_neg_mask)
        pred_boxes_points = pred_boxes_points * 64
        gt_list = gt_list * 64
        for b in range(num_batch):
            cur_pred_boxes = pred_boxes_points[b * total_anchors:(b + 1) * total_anchors]
            iou_gt_pred = self.IOU(self.Cast(cur_pred_boxes, mstype.float16), self.Cast(gt_list[b], mstype.float16))
            mask = self.Cast((iou_gt_pred > self.thresh), mstype.float16)
            mask = self.ReduceSum(mask, 0)
            mask = mask > 0
            shape_neg = self.Shape(conf_neg_mask[0])
            mask = self.Reshape(mask, (1, shape_neg[0], shape_neg[1]))
            if b == 0:
                mask_concat = mask
            else:
                mask_concat = self.concat0_2((mask_concat, mask))
        conf_neg_mask = self.Select(mask_concat, conf_neg_mask_zero, conf_neg_mask)

        coord_mask = self.Tile(coord_mask, (1, 1, 4, 1))
        coord_mask = coord_mask[:, :, :2]
        coord_center = coord[:, :, :2]
        t_coord_center = t_coord[:, :, :2]
        coord_wh = coord[:, :, 2:]
        t_coord_wh = t_coord[:, :, 2:]

        one_hot_label = None
        shape_cls_mask = None
        if num_classes > 1:
            shape_t_cls = self.Shape(t_cls)
            t_cls = self.Reshape(t_cls, (shape_t_cls[0] * shape_t_cls[1] * shape_t_cls[2],))
            one_hot_label = self.OneHot(self.Cast(t_cls, mstype.int32))

            shape_cls_mask = self.Shape(cls_mask)
            cls_mask = self.Reshape(cls_mask, (1, shape_cls_mask[0] * shape_cls_mask[1] * shape_cls_mask[2]))
        added_scale = 1.0 + self.head_idx * 0.5
        loss_coord_center = added_scale * 2.0 * 1.0 * self.coord_scale * self.Sum(
            coord_mask * self.BCE(coord_center, t_coord_center), ())

        loss_coord_wh = added_scale * 2.0 * 1.5 * self.coord_scale * self.Sum(
            coord_mask * self.SmoothL1Loss(coord_wh, t_coord_wh), ())

        loss_coord = 1.0 * (loss_coord_center + loss_coord_wh)

        loss_conf_pos = added_scale * 2.0 * self.object_scale * self.Sum(conf_pos_mask * self.BCE(conf, t_conf), ())
        loss_conf_neg = 1.0 * self.no_object_scale * self.Sum(conf_neg_mask * self.BCE(conf, t_conf), ())

        loss_conf = loss_conf_pos + loss_conf_neg

        loss_cls = None
        if num_classes > 1:
            loss_cls = self.class_scale * 1.0 * self.Sum(cls_mask * self.CE(cls, one_hot_label)[0], ())
        else:
            loss_cls = 0.0
            cls = self.Squeeze2(output[:, :, 5:6, :])
            loss_cls_pos = added_scale * 2.0 * self.object_scale * self.Sum(conf_pos_mask * self.BCE(cls, t_conf), ())
            loss_cls_neg = 1.0 * self.no_object_scale * self.Sum(conf_neg_mask * self.BCE(cls, t_conf), ())
            loss_cls = loss_cls_pos + loss_cls_neg

        loss_tot = loss_coord + 0.5 * loss_conf + 0.5 * loss_cls

        return loss_tot
