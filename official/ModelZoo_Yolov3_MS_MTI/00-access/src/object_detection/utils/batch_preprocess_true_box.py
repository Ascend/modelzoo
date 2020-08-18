from src.object_detection.yolo_v3.transforms import _preprocess_true_boxes
import numpy as np
import threading



def thread_batch_preprocess_true_box(annos, config, input_shape, result_index, batch_bbox_true_1, batch_bbox_true_2, batch_bbox_true_3, batch_gt_box1, batch_gt_box2, batch_gt_box3):
    i = 0
    for anno in annos:
        bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
            _preprocess_true_boxes(true_boxes=anno, anchors=config.anchor_scales, in_shape=input_shape,
                                    num_classes=config.num_classes, max_boxes=config.max_box,
                                    label_smooth=config.label_smooth, label_smooth_factor=config.label_smooth_factor)
        batch_bbox_true_1[result_index + i] = bbox_true_1
        batch_bbox_true_2[result_index + i] = bbox_true_2
        batch_bbox_true_3[result_index + i] = bbox_true_3 
        batch_gt_box1[result_index + i] = gt_box1
        batch_gt_box2[result_index + i] = gt_box2
        batch_gt_box3[result_index + i] = gt_box3
        i = i + 1

def batch_preprocess_true_box(annos, config, input_shape):
    batch_bbox_true_1 = []
    batch_bbox_true_2 = []
    batch_bbox_true_3 = []
    batch_gt_box1 = []
    batch_gt_box2 = []
    batch_gt_box3 = []
    threads = []
    i = 0
    step = 4
    for index in range(0, len(annos), step):
        for i in range(step):
            batch_bbox_true_1.append(None)
            batch_bbox_true_2.append(None)
            batch_bbox_true_3.append(None)
            batch_gt_box1.append(None)
            batch_gt_box2.append(None)
            batch_gt_box3.append(None)
        step_anno = annos[index: index + step]
        t = threading.Thread(target=thread_batch_preprocess_true_box, args=(step_anno, config, input_shape, index, batch_bbox_true_1, batch_bbox_true_2, batch_bbox_true_3, batch_gt_box1, batch_gt_box2, batch_gt_box3))
        t.start()
        threads.append(t)

    
    for t in threads:
        t.join()
    
    return np.array(batch_bbox_true_1), np.array(batch_bbox_true_2), np.array(batch_bbox_true_3), \
           np.array(batch_gt_box1), np.array(batch_gt_box2), np.array(batch_gt_box3)

    # return batch_bbox_true_1, batch_bbox_true_2, batch_bbox_true_3, batch_gt_box1, batch_gt_box2, batch_gt_box3

def batch_preprocess_true_box_single(annos, config, input_shape):
    batch_bbox_true_1 = []
    batch_bbox_true_2 = []
    batch_bbox_true_3 = []
    batch_gt_box1 = []
    batch_gt_box2 = []
    batch_gt_box3 = []
    for anno in annos:
        bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
            _preprocess_true_boxes(true_boxes=anno, anchors=config.anchor_scales, in_shape=input_shape,
                                    num_classes=config.num_classes, max_boxes=config.max_box,
                                    label_smooth=config.label_smooth, label_smooth_factor=config.label_smooth_factor)
        batch_bbox_true_1.append(bbox_true_1)
        batch_bbox_true_2.append(bbox_true_2)
        batch_bbox_true_3.append(bbox_true_3)
        batch_gt_box1.append(gt_box1)
        batch_gt_box2.append(gt_box2)
        batch_gt_box3.append(gt_box3)
    
    #print("non threads result:", batch_bbox_true_1, batch_bbox_true_2, batch_bbox_true_3, batch_gt_box1, batch_gt_box2, batch_gt_box3)

    #newbatch_bbox_true_1, newbatch_bbox_true_2, newbatch_bbox_true_3, newbatch_gt_box1, newbatch_gt_box2, newbatch_gt_box3 = \
    #    batch_preprocess_true_box_1(annos, config, input_shape)

    #print("threads result:", newbatch_bbox_true_1, newbatch_bbox_true_2, newbatch_bbox_true_3, newbatch_gt_box1, newbatch_gt_box2, newbatch_gt_box3)
    return np.array(batch_bbox_true_1), np.array(batch_bbox_true_2), np.array(batch_bbox_true_3), \
           np.array(batch_gt_box1), np.array(batch_gt_box2), np.array(batch_gt_box3)
