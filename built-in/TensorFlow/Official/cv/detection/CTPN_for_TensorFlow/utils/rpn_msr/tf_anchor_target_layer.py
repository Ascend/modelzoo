#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 14:21:00 2020

@author: jcm
"""


###########################################
# Date: 2020/11/06
# Author: Jincheng Ma
#
# anchor target layer  
# avoid dynamic shape during computation
# no subsampling operations for the fg samples
# using TF api
# ##########################################



import tensorflow as tf
from utils.rpn_msr.bbox import encode as encode_tf
from utils.rpn_msr.bbox import bbox_overlap_tf as bbox_overlap_tf
#from utils.rpn_msr.bbox import encode as encode_tf

from utils.rpn_msr.config import Config as cfg

import numpy as np
from utils.rpn_msr.generate_anchors import generate_anchors


#def anchor_target_layer(all_anchors, gt_boxes, im_shape):
def anchor_target_layer(rpn_cls_score, gt_boxes, bbox_pred, num_bbox, im_shape=[1216,1216], _feat_stride=[16,], anchor_scales=[16,]):



       # Keep only the coordinates of gt_boxes
       gt_boxes = gt_boxes[:, :4]
       gt_boxed = tf.convert_to_tensor(gt_boxes)


       #all_anchors = all_anchors[:, :4]
       _anchors = generate_anchors()
       _anchors = tf.convert_to_tensor(_anchors)

       height, width = rpn_cls_score.get_shape().as_list()[1:3]
       shift_x = tf.range(0, width) * _feat_stride
       shift_y = tf.range(0, height) * _feat_stride
       shift_x, shift_y = tf.meshgrid(shift_x, shift_y)  # in W H order
       # K is H x W

       shift_x_re = tf.reshape(shift_x,[-1,1])
       shift_y_re = tf.reshape(shift_y,[-1,1])
       shifts = tf.concat([shift_x_re,shift_y_re,shift_x_re,shift_y_re],axis=1)
       shifts = tf.expand_dims(shifts,axis=1) # shape from (K,4) -->( K,1,4)

       # add A anchors (1, A, 4) to
       # cell K shifts (K, 1, 4) to get
       # shift anchors (K, A, 4)
       # reshape to (K*A, 4) shifted anchors
       #A = _num_anchors  # 9个anchor
       A = 10  # 9个anchor
       K = shifts.get_shape().as_list()[0]  # 50*37，feature-map的宽乘高的大小

       all_anchors = (tf.expand_dims(_anchors,axis=0) + shifts)  # (1, A, 4 ) + (K, 1, 4)--> (K, A , 4)   相当于复制宽高的维度，然后相加
       all_anchors = tf.reshape(all_anchors,[K * A, 4])

       (x_min_anchor, y_min_anchor,
        x_max_anchor, y_max_anchor) = tf.unstack(all_anchors, axis=1)

       _allowed_border = 0
       anchor_filter = tf.logical_and(
           tf.logical_and(
               tf.greater_equal(x_min_anchor, -_allowed_border),
               tf.greater_equal(y_min_anchor, -_allowed_border)
           ),
           tf.logical_and(
               tf.less(x_max_anchor, im_shape[1] + _allowed_border),
               tf.less(y_max_anchor, im_shape[0] + _allowed_border)
           )
       )

       anchor_filter = tf.reshape(anchor_filter, [-1])

       index_valid = tf.cast(anchor_filter, tf.int8)

       # Filter anchors.
       anchors = tf.boolean_mask(
           all_anchors, anchor_filter, name='filter_anchors')
       bbox_pred = tf.reshape(bbox_pred ,[-1,4])
       bbox_pred = tf.boolean_mask(bbox_pred, anchor_filter)

       cls_score = tf.reshape(rpn_cls_score, [-1,2])
       cls_score = tf.boolean_mask(cls_score,anchor_filter)
       print("cls: ", cls_score)
       # calculate the overlap between anchor and gt_boxes
       overlaps = bbox_overlap_tf(tf.to_float(anchors),tf.to_float(gt_boxes))
       print("overlaps :", overlaps)
       # start labeling fg and bg boxes
       max_overlap = tf.reduce_max(overlaps, axis=1)
       argmax_overlaps = tf.argmax(overlaps,axis=1)
       gt_argmax_overlaps = tf.argmax(overlaps, axis=0)
     
       def pass_value(val):
           return val 
       max_num_fg = int(cfg.RPN_FG_FRACTION * cfg.RPN_BATCHSIZE)
       #num_bbox = tf.squeeze(num_bbox)
       #num_fg = tf.cond(tf.less(num_bbox,max_num_fg),
       #                 lambda: pass_value(num_bbox),
       #                 lambda:pass_value(max_num_fg))

       #_,fg_index= tf.math.top_k(max_overlap,k=num_fg)
       num_fg = 150
       _,fg_index= tf.math.top_k(max_overlap,k=150)
       #num_bg = cfg.RPN_BATCHSIZE - num_fg
       
       num_bg = 150
       #_,bg_index = tf.math.top_k(-max_overlap,k=num_bg)
       _,bg_index = tf.math.top_k(-max_overlap,k=150)
       

       # select anchor with fg_index and bg_index
       anchor_fg = tf.gather(anchors, fg_index)
       anchor_bg = tf.gather(anchors, bg_index)
       anchor_selected = tf.concat([anchor_fg,anchor_bg],axis=0)

       # select gt_boxes
       gt_box = tf.gather(gt_boxes, argmax_overlaps)
       gt_box_fg = tf.gather(gt_box, fg_index)
       gt_box_bg = tf.gather(gt_box, bg_index)
       gt_box_selected = tf.concat([gt_box_fg,gt_box_bg],axis=0)

       # select bbox_pred
       bbox_pred_fg = tf.gather(bbox_pred,fg_index)
       bbox_pred_bg = tf.gather(bbox_pred , bg_index)
       bbox_pred_selected = tf.concat([bbox_pred_fg,bbox_pred_bg],axis=0)

       # select cls_score
       cls_score_fg = tf.gather(cls_score, fg_index)
       cls_score_bg = tf.gather(cls_score, bg_index)
       cls_score_selected = tf.concat([cls_score_fg,cls_score_bg],axis=0)

       #prepare labels
       labels_fg = tf.ones(num_fg)
       labels_bg = tf.zeros(num_bg)
       labels = tf.concat([labels_fg,labels_bg],axis=0)
       labels = tf.cast(labels, tf.int64)

       #select bbox_target
       bbox_target = encode_tf(anchor_selected,gt_box_selected)


       # bbox weight
       bbox_inside_weights_fg = tf.reshape(tf.ones(num_fg*4),[-1, 4])
       bbox_inside_weights_bg = tf.reshape(tf.zeros(num_bg*4),[-1,4])
       bbox_inside_weights =tf.concat([bbox_inside_weights_fg,bbox_inside_weights_bg],axis=0)

       return labels, bbox_target,cls_score_selected,bbox_pred_selected,bbox_inside_weights,bbox_inside_weights

