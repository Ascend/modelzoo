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
from utils.bbox.nms import nms

from utils.bbox.bbox_transform import bbox_transform_inv, clip_boxes
from utils.rpn_msr.config import Config as cfg
from utils.rpn_msr.generate_anchors import generate_anchors

DEBUG = False


def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, _feat_stride=[16, ], anchor_scales=[16, ]):
    """
    Parameters
    ----------
    rpn_cls_prob_reshape: (1 , H , W , Ax2) outputs of RPN, prob of bg or fg
                         NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)
    """

    _anchors = generate_anchors(scales=np.array(anchor_scales))  # 鐢熸垚鍩烘湰鐨9涓猘nchor
    _num_anchors = _anchors.shape[0]  # 9涓猘nchor

    im_info = im_info[0]  # 鍘熷鍥惧儚鐨勯珮瀹姐佺缉鏀惧昂搴

    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'

    pre_nms_topN = cfg.RPN_PRE_NMS_TOP_N  # 12000,鍦ㄥ仛nms涔嬪墠锛屾渶澶氫繚鐣欑殑鍊欓塨ox鏁扮洰
    post_nms_topN = cfg.RPN_POST_NMS_TOP_N  # 2000锛屽仛瀹宯ms涔嬪悗锛屾渶澶氫繚鐣欑殑box鐨勬暟鐩
    nms_thresh = cfg.RPN_NMS_THRESH  # nms鐢ㄥ弬鏁帮紝闃堝兼槸0.7
    min_size = cfg.RPN_MIN_SIZE  # 鍊欓塨ox鐨勬渶灏忓昂瀵革紝鐩墠鏄16锛岄珮瀹藉潎瑕佸ぇ浜16

    height, width = rpn_cls_prob_reshape.shape[1:3]  # feature-map鐨勯珮瀹
    width = width // 10

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    # (1, H, W, A)
    scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])[:, :, :, :, 1],
                        [1, height, width, _num_anchors])
    # 鎻愬彇鍒皁bject鐨勫垎鏁帮紝non-object鐨勬垜浠笉鍏冲績

    bbox_deltas = rpn_bbox_pred  # 妯″瀷杈撳嚭鐨刾red鏄浉瀵瑰硷紝闇瑕佽繘涓姝ュ鐞嗘垚鐪熷疄鍥惧儚涓殑鍧愭爣
    # im_info = bottom[2].data[0, :]

    if DEBUG:
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))

    # 1. Generate proposals from bbox deltas and shifted anchors
    if DEBUG:
        print('score map size: {}'.format(scores.shape))

    # Enumerate all shifts
    # 鍚宎nchor-target-layer-tf杩欎釜鏂囦欢涓鏍凤紝鐢熸垚anchor鐨剆hift锛岃繘涓姝ュ緱鍒版暣寮犲浘鍍忎笂鐨勬墍鏈塧nchor
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))  # 杩欓噷寰楀埌鐨刟nchor灏辨槸鏁村紶鍥惧儚涓婄殑鎵鏈塧nchor

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.reshape((-1, 4))  # (HxWxA, 4)

    # Same story for the scores:
    scores = scores.reshape((-1, 1))

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)  # 鍋氶嗗彉鎹紝寰楀埌box鍦ㄥ浘鍍忎笂鐨勭湡瀹炲潗鏍

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])  # 灏嗘墍鏈夌殑proposal淇缓涓涓嬶紝瓒呭嚭鍥惧儚鑼冨洿鐨勫皢浼氳淇壀鎺

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes(proposals, min_size)  # 绉婚櫎閭ｄ簺proposal灏忎簬涓瀹氬昂瀵哥殑proposal
    proposals = proposals[keep, :]  # 淇濈暀鍓╀笅鐨刾roposal
    scores = scores[keep]
    bbox_deltas = bbox_deltas[keep, :]

    # # remove irregular boxes, too fat too tall
    # keep = _filter_irregular_boxes(proposals)
    # proposals = proposals[keep, :]
    # scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]  # score鎸夊緱鍒嗙殑楂樹綆杩涜鎺掑簭
    if pre_nms_topN > 0:  # 淇濈暀12000涓猵roposal杩涘幓鍋歯ms
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]
    bbox_deltas = bbox_deltas[order, :]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)), nms_thresh)  # 杩涜nms鎿嶄綔锛屼繚鐣2000涓猵roposal
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]
    bbox_deltas = bbox_deltas[keep, :]

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    blob = np.hstack((scores.astype(np.float32, copy=False), proposals.astype(np.float32, copy=False)))

    return blob, bbox_deltas


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def _filter_irregular_boxes(boxes, min_ratio=0.2, max_ratio=5):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    rs = ws / hs
    keep = np.where((rs <= max_ratio) & (rs >= min_ratio))[0]
    return keep
