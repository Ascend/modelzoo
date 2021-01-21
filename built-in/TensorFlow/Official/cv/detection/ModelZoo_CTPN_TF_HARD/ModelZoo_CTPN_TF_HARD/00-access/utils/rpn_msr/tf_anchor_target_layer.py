import tensorflow as tf
from utils.rpn_msr.bbox import encode as encode_tf
from utils.rpn_msr.bbox import bbox_overlap_tf as bbox_overlap_tf

from utils.rpn_msr.config import Config as cfg
from npu_bridge.npu_cpu import npu_cpu_ops as custom_op 

import numpy as np
from utils.rpn_msr.generate_anchors import generate_anchors

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

INPUTS_SHAPE=[FLAGS.inputs_height, FLAGS.inputs_width]


def anchor_target_layer(rpn_cls_score, gt_boxes, bbox_pred, im_shape=INPUTS_SHAPE, _feat_stride=[16,], anchor_scales=[16,]):



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
       A = 10  # 10 anchor
       K = shifts.get_shape().as_list()[0]  # H*W feature-map
       all_anchors = (tf.expand_dims(_anchors,axis=0) + shifts)  # (1, A, 4 ) + (K, 1, 4)--> (K, A , 4)   

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
       tf.add_to_collection("anchor_filter", anchor_filter) 
       index_valid = tf.cast(anchor_filter, tf.int8)

       # Filter anchors.
       anchors = tf.boolean_mask(
           all_anchors, anchor_filter, name='filter_anchors')
       bbox_pred = tf.reshape(bbox_pred ,[-1,4])
       bbox_pred = tf.boolean_mask(bbox_pred, anchor_filter)
       tf.add_to_collection("anchors", anchors)

       cls_pred_shape = tf.shape(rpn_cls_score)
       cls_pred_reshape = tf.reshape(rpn_cls_score, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])

       cls_score = tf.reshape(cls_pred_reshape, [-1,2])
       cls_score = tf.boolean_mask(cls_score,anchor_filter)
       # calculate the overlap between anchor and gt_boxes
       overlaps = bbox_overlap_tf(tf.to_float(anchors),tf.to_float(gt_boxes))
       
       # start labeling fg and bg boxes
       max_overlap = tf.reduce_max(overlaps, axis=1)
       tf.add_to_collection("max_overlap", max_overlap)
       argmax_overlaps = tf.argmax(overlaps,axis=1)
       gt_argmax_overlaps = tf.argmax(overlaps, axis=0)
     
       max_num_fg = int(cfg.RPN_FG_FRACTION * cfg.RPN_BATCHSIZE)
       
       len_labels, len_gt_box = tf.shape(overlaps)[0],tf.shape(overlaps)[1]
       inds = tf.reshape(tf.range(len_labels), (-1,1))
       inds = tf.cast(inds, tf.int64)
       
       gt_argmax_overlaps_reshape = tf.reshape(gt_argmax_overlaps,([-1]))
       one_hot_codec = tf.one_hot(gt_argmax_overlaps_reshape, depth=len_labels)
       mask_fg_1 = tf.reduce_sum(one_hot_codec,axis=0)
       mask_fg_1_b = tf.greater(mask_fg_1,0)

       # due to zero padding
       mask_fg_2 = tf.greater(max_overlap, 0)
       mask_fg_1_b = tf.logical_and(mask_fg_1_b,mask_fg_2)
       mask_fg = tf.greater_equal(max_overlap,cfg.RPN_POSITIVE_OVERLAP)
       mask_fg = tf.logical_or(mask_fg_1_b, mask_fg)
       
       mask_bg = tf.less_equal(max_overlap, cfg.RPN_NEGATIVE_OVERLAP)
       mask_bg_2 = tf.greater(max_overlap, 0)
       mask_bg = tf.logical_and(mask_bg, mask_bg_2)

       num_fg = int(cfg.RPN_FG_FRACTION * cfg.RPN_BATCHSIZE)
       num_bg = cfg.RPN_BATCHSIZE - num_fg
       
       # choose at most num_fg bbox and num_bg  bbox
       inds_fg, mask_fg_ret  = custom_op.randomchoicewithmask(x=mask_fg,count=num_fg)
       inds_bg, mask_bg_ret  = custom_op.randomchoicewithmask(x=mask_bg,count=num_bg)
       
       
       fg_index = tf.squeeze(inds_fg)
       bg_index = tf.squeeze(inds_bg)
       
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
       
       mask_fg_fp = tf.cast(mask_fg_ret,tf.float32)
       mask_bg_fp = tf.cast(tf.zeros(num_bg), tf.float32)
       mask_ret = tf.concat([mask_fg_fp,mask_bg_fp], axis=0)
       labels = tf.cast(mask_ret, tf.int32)

       #select bbox_target
       bbox_target = encode_tf(anchor_selected,gt_box_selected)

       # bbox weight
       bbox_inside_weights_fg = tf.reshape(tf.ones(num_fg*4),[-1, 4])
       bbox_inside_weights_bg = tf.reshape(tf.zeros(num_bg*4),[-1,4])
       bbox_inside_weights =tf.concat([bbox_inside_weights_fg,bbox_inside_weights_bg],axis=0)

       bbox_inside_weights = tf.cast(tf.reshape(labels,[-1,1]),tf.float32) * bbox_inside_weights
       return labels, bbox_target,cls_score_selected,bbox_pred_selected,bbox_inside_weights,bbox_inside_weights

