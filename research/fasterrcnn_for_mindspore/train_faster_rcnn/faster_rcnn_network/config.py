import math

class Config_Faster_Rcnn():
    img_meta = [768, 1280]

    #anchor
    feature_shapes = [(192, 320), (96, 160), (48, 80), (24, 40), (12, 20)]
    anchor_scales = [8]
    anchor_ratios = [0.5, 1.0, 2.0]
    anchor_strides = [4, 8, 16, 32, 64]
    num_anchors = len(anchor_ratios) * len(anchor_scales)

    #resnet
    resnet_block = [3, 4, 6, 3]
    resnet_in_channels = [64, 256, 512, 1024]
    resnet_out_channels = [256, 512, 1024, 2048]

    #fpn
    fpn_in_channels = [256, 512, 1024, 2048]
    fpn_out_channels = 256
    fpn_num_outs = 5

    #rpn
    rpn_in_channels = 256
    rpn_feat_channels = 256
    rpn_loss_cls_weight = 1.0
    rpn_loss_reg_weight = 1.0
    rpn_cls_out_channels = 1

    rpn_target_means = [0., 0., 0., 0.]
    rpn_target_stds = [1.0, 1.0, 1.0, 1.0]

    #bbox_assign_sampler
    neg_iou_thr = 0.3
    pos_iou_thr = 0.7
    min_pos_iou = 0.3
    num_bboxes = 245520
    num_gts = 128
    num_expected_neg = 256
    num_expected_pos = 128

    #proposal
    num_classes = 2
    use_sigmoid_cls = True

    #roi_align
    roi_layer = dict(type='RoIAlign', out_size=7, sample_num=2)
    roi_align_out_channels = 256
    roi_align_featmap_strides = [4, 8, 16, 32]
    roi_align_finest_scale = 56
    roi_sample_num = 640

    #bbox_assign_sampler_stage2
    neg_iou_thr_stage2 = 0.5
    pos_iou_thr_stage2 = 0.5
    min_pos_iou_stage2 = 0.5
    num_bboxes_stage2 = 2000
    num_expected_pos_stage2 = 128
    num_expected_neg_stage2 = 512
    num_expected_total_stage2 = 512
    
    #rcnn
    rcnn_num_layers = 2
    rcnn_in_channels = 256
    rcnn_fc_out_channels = 1024
    rcnn_num_bboxes_one_batch = (num_expected_pos_stage2 + num_expected_neg_stage2)
    rcnn_loss_cls_weight = 1
    rcnn_loss_reg_weight = 1

    rcnn_target_means = [0., 0., 0., 0.]
    rcnn_target_stds = [0.1, 0.1, 0.2, 0.2]

    rpn_proposal_nms_across_levels = False
    rpn_proposal_nms_pre = 2000
    rpn_proposal_nms_post = 2000
    rpn_proposal_max_num = 2000
    rpn_proposal_nms_thr = 0.7
    rpn_proposal_min_bbox_size = 0

    rpn_nms_across_levels = False
    rpn_nms_pre = 1000
    rpn_nms_post = 1000
    rpn_max_num = 1000
    rpn_nms_thr = 0.7
    rpn_min_bbox_min_size = 0
    test_score_thr = 0.05
    test_iou_thr=0.5
    test_max_per_img = 100
    test_batch_size = 1

    rpn_head_loss_type = "CrossEntropyLoss"
    rpn_head_use_sigmoid = True
    rpn_head_weight = 1.0
 
    #SGD
    base_lr = 0.02
    base_step = 58633
    total_epoch = 13
    warmup_step = 500
    warmup_mode = "linear"
    warmup_ratio = 1/3.0
    sgd_step = [8, 11]
    sgd_momentum = 0.9
