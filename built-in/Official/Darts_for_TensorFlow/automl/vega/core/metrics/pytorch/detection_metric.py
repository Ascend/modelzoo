# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of detection task by using coco tools."""
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .recall_eval import eval_recalls
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.metrics.pytorch.metrics import MetricBase


@ClassFactory.register(ClassType.METRIC)
class DetMetric(MetricBase):
    """Save and summary metric from mdc dataset using coco tools."""

    def __init__(self, gt_anno_path=None, eval_class_label=None):
        self.__metric_name__ = "DetMetric"
        self.gt_coco_files = gt_anno_path
        self.result_record = []
        self.eval_class_label = eval_class_label

    def __call__(self, output, target, *args, **kwargs):
        """Append input into result record cache.

        :param output: output data
        :param target: target data
        :return:
        """
        self.result_record.append((output, target.data[0]))
        return None

    def reset(self):
        """Reset states for new evaluation after each epoch."""
        self.result_record = []

    def summary(self):
        """Summary all record from result cache, and get performance."""
        json_results = []
        for idx in range(len(self.result_record)):
            result = self.result_record[idx][0]
            img_id = int(self.result_record[idx][1])
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = label + 1  # change label from 0-based to 1-based
                    json_results.append(data)
        if len(json_results) == 0:
            return {}
        eval_result = self.print_evaluation_scores(json_results, self.gt_coco_files)
        eval_result = eval_result['AP(bbox)']

        if self.eval_class_label is None:
            return eval_result
        else:
            eval_ap = {}
            for klass in self.eval_class_label:
                if klass in eval_result:
                    mAP = eval_result[klass][1] * 100
                    AP_small = eval_result[klass][3] * 100
                    AP_medium = eval_result[klass][4] * 100
                    AP_large = eval_result[klass][5] * 100
                    eval_ap[klass] = [mAP, AP_small, AP_medium, AP_large]
            return eval_ap

    def print_evaluation_scores(self, det_json_file, gt_json_file):
        """Print evaluation scores.

        :param det_json_file: dest json file
        :param gt_json_file: gt json file
        :return:
        """
        ret = {}
        coco = COCO(gt_json_file)
        cocoDt = coco.loadRes(det_json_file)
        cocoEval = COCOeval(coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        ret['AP(bbox)'] = cocoEval.stats
        return ret


def coco_eval(result_files, result_types, coco, max_dets=(100, 300, 1000)):
    """Calculate detection metric by using coco tools.

    :param result_files: predicted result files
    :type result_files: str(json file) or dict
    :param result_types: evaluate result types
    :type result_types: list of str
    :param coco: coco ground truth file or COCO from pycocotools lib
    :type coco: str or COCO
    :param max_dets: max detection number
    :type max_dets: tuple of int, default to (100, 300, 1000)
    :return: list of coco evaluation results
    :rtype: list of numpy 1D array
    """
    import mmcv
    for res_type in result_types:
        if res_type not in ['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints']:
            raise TypeError('wrong result type')
    if mmcv.is_str(coco):
        coco = COCO(coco)
    if not isinstance(coco, COCO):
        raise TypeError('coco should be type COCO')
    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_files, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return ar
    results = []
    for res_type in result_types:
        if isinstance(result_files, str):
            result_file = result_files
        elif isinstance(result_files, dict):
            result_file = result_files[res_type]
        else:
            raise TypeError('result_files must be a str or dict')
        if not result_file.endswith('.json'):
            raise TypeError('result_file must be json file')
        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        results.append(cocoEval.stats)
    return results


def fast_eval_recall(results,
                     coco,
                     max_dets,
                     iou_thrs=np.arange(0.5, 0.96, 0.05)):
    """Fast evaluation of recall metric.

    :param results: predicted result data or name
    :type results: str or list of numpy arrays
    :param coco: COCO from pycocotools lib
    :type coco: COCO
    :param max_dets: max detection number
    :type max_dets: tuple of int
    :param iou_thrs: IoU threshholds
    :type iou_thrs: numpy array of thresholds, default to np.arange(0.5, 0.96, 0.05)
    :return: average recall
    :rtype: float
    """
    import mmcv
    if mmcv.is_str(results) and results.endswith('.pkl'):
        results = mmcv.load(results)
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a pkl filename, not {}'.format(type(results)))
    gt_bboxes = []
    img_ids = coco.getImgIds()
    for i in range(len(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_ids[i])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs)
    ar = recalls.mean(axis=1)
    return ar


def xyxy2xywh(bbox):
    """Transform the bbox coordinate to [x,y ,w,h].

    :param bbox: the predict bounding box coordinate
    :type bbox: list
    :return: [x,y ,w,h]
    :rtype: list
    """
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]
