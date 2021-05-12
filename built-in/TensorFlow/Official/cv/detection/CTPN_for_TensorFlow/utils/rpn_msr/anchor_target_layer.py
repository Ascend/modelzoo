# -*- coding:utf-8 -*-
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import numpy.random as npr
from utils.bbox.bbox import bbox_overlaps
from utils.rpn_msr.bbox import bbox_overlap_tf


from utils.bbox.bbox_transform import bbox_transform
from utils.rpn_msr.config import Config as cfg
from utils.rpn_msr.generate_anchors import generate_anchors

DEBUG = False


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride=[16, ], anchor_scales=[16, ]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_cls_score: (1, H, W, Ax2) bg/fg scores of previous conv layer
    gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    """
    _anchors = generate_anchors(scales=np.array(anchor_scales))  # 鐢熸垚鍩烘湰鐨刟nchor,涓鍏9涓
    _num_anchors = _anchors.shape[0]  # 9涓猘nchor

    if DEBUG:
        print('anchors:')
        print(_anchors)
        print('anchor shapes:')
        print(np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        )))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0
    # map of shape (..., H, W)
    # height, width = rpn_cls_score.shape[1:3]

    im_info = im_info[0]  # 鍥惧儚鐨勯珮瀹藉強閫氶亾鏁
    if DEBUG:
        print("im_info: ", im_info)
    # 鍦╢eature-map涓婂畾浣峚nchor锛屽苟鍔犱笂delta锛屽緱鍒板湪瀹為檯鍥惧儚涓璦nchor鐨勭湡瀹炲潗鏍
    # Algorithm:
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    assert rpn_cls_score.shape[0] == 1, 'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]  # feature-map鐨勯珮瀹

    if DEBUG:
        print('AnchorTargetLayer: height', height, 'width', width)
        print('')
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))
        print('height, width: ({}, {})'.format(height, width))
        print('rpn: gt_boxes.shape', gt_boxes.shape)
        print('rpn: gt_boxes', gt_boxes)

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order
    # K is H x W
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()  # 鐢熸垚feature-map鍜岀湡瀹瀒mage涓奱nchor涔嬮棿鐨勫亸绉婚噺
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors  # 9涓猘nchor
    K = shifts.shape[0]  # 50*37锛宖eature-map鐨勫涔橀珮鐨勫ぇ灏
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))  # 鐩稿綋浜庡鍒跺楂樼殑缁村害锛岀劧鍚庣浉鍔
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    # 浠呬繚鐣欓偅浜涜繕鍦ㄥ浘鍍忓唴閮ㄧ殑anchor锛岃秴鍑哄浘鍍忕殑閮藉垹鎺
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]

    if DEBUG:
        print('total_anchors', total_anchors)
        print('inds_inside', len(inds_inside))

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]  # 淇濈暀閭ｄ簺鍦ㄥ浘鍍忓唴鐨刟nchor
    if DEBUG:
        print('anchors.shape', anchors.shape)
        print('num_anchor', _num_anchors)
        print('shifts shape', shifts.shape)
        print('K ', K)
    # 鑷虫锛宎nchor鍑嗗濂戒簡
    # --------------------------------------------------------------
    # label: 1 is positive, 0 is negative, -1 is dont care
    # (A)
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)  # 鍒濆鍖杔abel锛屽潎涓-1

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt), shape is A x G
    # 璁＄畻anchor鍜実t-box鐨刼verlap锛岀敤鏉ョ粰anchor涓婃爣绛
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))  # 鍋囪anchors鏈墄涓紝gt_boxes鏈墆涓紝杩斿洖鐨勬槸涓涓紙x,y锛夌殑鏁扮粍
    # 瀛樻斁姣忎竴涓猘nchor鍜屾瘡涓涓猤tbox涔嬮棿鐨刼verlap
    argmax_overlaps = overlaps.argmax(axis=1)  # (A)#鎵惧埌鍜屾瘡涓涓猤tbox锛宱verlap鏈澶х殑閭ｄ釜anchor
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)  # G#鎵惧埌姣忎釜浣嶇疆涓9涓猘nchor涓笌gtbox锛宱verlap鏈澶х殑閭ｄ釜
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if DEBUG:
        print('argmax_overlaps shape:' , argmax_overlaps.shape)
        print('gt_argmax_overlaps shape:', gt_argmax_overlaps.shape)
        print('gt_argmax_overlaps shape: ', gt_argmax_overlaps.shape)

    if not cfg.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0  # 鍏堢粰鑳屾櫙涓婃爣绛撅紝灏忎簬0.3overlap鐨

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1  # 姣忎釜浣嶇疆涓婄殑9涓猘nchor涓璷verlap鏈澶х殑璁や负鏄墠鏅
    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.RPN_POSITIVE_OVERLAP] = 1  # overlap澶т簬0.7鐨勮涓烘槸鍓嶆櫙

    if cfg.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    # 瀵规鏍锋湰杩涜閲囨牱锛屽鏋滄鏍锋湰鐨勬暟閲忓お澶氱殑璇
    # 闄愬埗姝ｆ牱鏈殑鏁伴噺涓嶈秴杩128涓
    num_fg = int(cfg.RPN_FG_FRACTION * cfg.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)  # 闅忔満鍘婚櫎鎺変竴浜涙鏍锋湰
        labels[disable_inds] = -1  # 鍙樹负-1

    # subsample negative labels if we have too many
    # 瀵硅礋鏍锋湰杩涜閲囨牱锛屽鏋滆礋鏍锋湰鐨勬暟閲忓お澶氱殑璇
    # 姝ｈ礋鏍锋湰鎬绘暟鏄256锛岄檺鍒舵鏍锋湰鏁扮洰鏈澶128锛
    # 濡傛灉姝ｆ牱鏈暟閲忓皬浜128锛屽樊鐨勯偅浜涘氨鐢ㄨ礋鏍锋湰琛ヤ笂锛屽噾榻256涓牱鏈
    num_bg = cfg.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        # print "was %s inds, disabling %s, now %s inds" % (
        # len(bg_inds), len(disable_inds), np.sum(labels == 0))

    # 鑷虫锛 涓婂ソ鏍囩锛屽紑濮嬭绠梤pn-box鐨勭湡鍊
    # --------------------------------------------------------------
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])  # 鏍规嵁anchor鍜実tbox璁＄畻寰楃湡鍊硷紙anchor鍜実tbox涔嬮棿鐨勫亸宸級

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.RPN_BBOX_INSIDE_WEIGHTS)  # 鍐呴儴鏉冮噸锛屽墠鏅氨缁1锛屽叾浠栨槸0

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.RPN_POSITIVE_WEIGHT < 0:  # 鏆傛椂浣跨敤uniform 鏉冮噸锛屼篃灏辨槸姝ｆ牱鏈槸1锛岃礋鏍锋湰鏄0
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0) + 1
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.RPN_POSITIVE_WEIGHT /
                            (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.RPN_POSITIVE_WEIGHT) /
                            (np.sum(labels == 0)) + 1)
    bbox_outside_weights[labels == 1, :] = positive_weights  # 澶栭儴鏉冮噸锛屽墠鏅槸1锛岃儗鏅槸0
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print('means:')
        print(means)
        print('stdevs:')
        print(stds)

    # map up to original set of anchors
    # 涓寮濮嬫槸灏嗚秴鍑哄浘鍍忚寖鍥寸殑anchor鐩存帴涓㈡帀鐨勶紝鐜板湪鍦ㄥ姞鍥炴潵
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)  # 杩欎簺anchor鐨刲abel鏄-1锛屼篃鍗砫ontcare
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)  # 杩欎簺anchor鐨勭湡鍊兼槸0锛屼篃鍗虫病鏈夊
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)  # 鍐呴儴鏉冮噸浠0濉厖
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)  # 澶栭儴鏉冮噸浠0濉厖

    if DEBUG:
        print('rpn: max max_overlap', np.max(max_overlaps))
        print('rpn: num_positive', np.sum(labels == 1))
        print('rpn: num_negative', np.sum(labels == 0))
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print('rpn: num_positive avg', _fg_sum / _count)
        print('rpn: num_negative avg', _bg_sum / _count)
        print("rpn keep: ",len(np.where(labels==1)[0]))
        print("rpn keep: ",len(np.where(labels==0)[0]))

    # labels
    labels = labels.reshape((1, height, width, A))  # reshap涓涓媗abel
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))  # reshape

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    if DEBUG:
        print("anchor target set")
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
