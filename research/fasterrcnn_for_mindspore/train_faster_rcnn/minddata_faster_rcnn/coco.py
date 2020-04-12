"""Accuracy."""
import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .coco_utils import *
from mindspore.nn.metrics import Metric


class COCOMetric(Metric):
    def __init__(self, dataset_coco, num_classes, max_num, bbox_json_dir, eval_types=["bbox"], single_result=True):
        super(COCOMetric, self).__init__()
        self._dataset = dataset_coco
        self._json_dir = bbox_json_dir
        self._single_result = single_result
        self._eval_types = eval_types
        self._max_num = max_num
        self._num_classes = num_classes
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self._results = []

    def update(self, *inputs):
        print("wangnan input len: ", len(inputs), inputs)
        all_bbox = inputs[0].asnumpy()
        all_label = inputs[1].asnumpy()
        all_mask = inputs[2].asnumpy()

        all_bbox_squee = np.squeeze(all_bbox)
        all_label_squee = np.squeeze(all_label)
        all_mask_squee = np.squeeze(all_mask)

        print("----> all_bbox.shape =", all_bbox.shape) #1*80000*5
        print("----> all_label_squee.shape =", all_label_squee.shape) #1*80000
        print("----> all_mask_squee.shape =", all_mask_squee.shape) #1*80000

        all_bboxes_tmp = []
        all_labels_tmp = []
        all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
        all_labels_tmp_mask = all_label_squee[all_mask_squee]

        if all_bboxes_tmp_mask.shape[0] > self._max_num:
            inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
            inds = inds[:self._max_num]
            all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
            all_labels_tmp_mask = all_labels_tmp_mask[inds]

        outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, self._num_classes)

        self._results.append(outputs_tmp)

    def eval(self):
        if len(self._results) == 0:
            raise RuntimeError('Accuary can not be calculated, because sample size is 0.')

        result_files = results2json(self._dataset, self._results, self._json_dir)
        return coco_eval(result_files, self._eval_types, self._dataset, single_result=self._single_result)
