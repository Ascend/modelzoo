import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .recall import eval_recalls


def coco_eval(result_files, result_types, coco, max_dets=(100, 300, 1000), single_result=False):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
        ]

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_files, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        gt_img_ids = coco.getImgIds()
        det_img_ids = coco_dets.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        
        tgt_ids = gt_img_ids if not single_result else det_img_ids

        if single_result :
            res_dict = dict()
            for id_i in tgt_ids:
                cocoEval = COCOeval(coco, coco_dets, iou_type)
                if res_type == 'proposal':
                    cocoEval.params.useCats = 0
                    cocoEval.params.maxDets = list(max_dets)

                cocoEval.params.imgIds = [id_i]
                print('current image id:', id_i, '  path:', coco.imgs[id_i]['file_name'])
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                res_dict.update({coco.imgs[id_i]['file_name']: cocoEval.stats[1]})

        cocoEval = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        cocoEval.params.imgIds = tgt_ids 
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        summary_metrics = {
        'Precision/mAP': cocoEval.stats[0],
        'Precision/mAP@.50IOU': cocoEval.stats[1],
        'Precision/mAP@.75IOU': cocoEval.stats[2],
        'Precision/mAP (small)': cocoEval.stats[3],
        'Precision/mAP (medium)': cocoEval.stats[4],
        'Precision/mAP (large)': cocoEval.stats[5],
        'Recall/AR@1': cocoEval.stats[6],
        'Recall/AR@10': cocoEval.stats[7],
        'Recall/AR@100': cocoEval.stats[8],
        'Recall/AR@100 (small)': cocoEval.stats[9],
        'Recall/AR@100 (medium)': cocoEval.stats[10],
        'Recall/AR@100 (large)': cocoEval.stats[11],
        }
    print('summary_metric: ', summary_metrics)
    #return summary_metrics
    return cocoEval.stats[1]
      

def fast_eval_recall(results,
                     coco,
                     max_dets,
                     iou_thrs=np.arange(0.5, 0.96, 0.05)):
    if mmcv.is_str(results):
        assert results.endswith('.pkl')
        results = mmcv.load(results)
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a filename, not {}'.
            format(type(results)))

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
        gt_bboxes, results, max_dets, iou_thrs, print_summary=False)
    ar = recalls.mean(axis=1)
    return ar


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    #
    #ann_file = '/opt/npu/liting/source/cocodataset/annotations/instances_val2017.json'
    ann_file = '/opt/npu/liting/source/cocodataset/small_sample_COCO/instances_2_images.json'
    dataset_coco = COCO(ann_file)
    img_ids = dataset_coco.getImgIds()
    #
    json_results = []
    #for idx in range(len(dataset)):
    dataset_len = dataset.get_dataset_size()*2
    for idx in range(dataset_len):
        #img_id = dataset.img_ids[idx]
        img_id = img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    #ann_file = '/opt/npu/liting/source/cocodataset/annotations/instances_val2017.json'
    #ann_file = '/opt/npu/liting/source/cocodataset/small_sample_COCO/instances_6_images.json'
    #ann_file = '/opt/npu/liting/source/cocodataset/annotations/instances_val2017_500_image.json'
    #dataset_coco = COCO(ann_file)
    cat_ids = dataset.getCatIds()
    #print("======> cat_ids =", cat_ids)
    #print("======> len(cat_ids) =", len(cat_ids))
    img_ids = dataset.getImgIds()
    #print("======> len(img_ids) =", len(img_ids))
    json_results = []
    #img_id = 0
    #for idx in range(len(dataset)):
    dataset_len = len(img_ids)
    #print("==============> dataset_len =", dataset_len)
    for idx in range(dataset_len):
        #img_id = dataset.img_ids[idx]
        img_id = img_ids[idx]
        #print("img_id =", img_id)
        if idx == len(results): break
        result = results[idx]
        #print("============> len(result) =", len(result))
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                #print("img_id =", img_id)
                data['bbox'] = xyxy2xywh(bboxes[i])
                #print("i =", i)
                #print("data['bbox'] =", data['bbox'])
                data['score'] = float(bboxes[i][4])
                #print("data['score'] =", data['score'])
                #data['category_id'] = dataset.cat_ids[label]    
                data['category_id'] = cat_ids[label]  
                #print("data['category_id'] =", data['category_id'])
                json_results.append(data)
    #print("=================> len(json_results) =", len(json_results))
    return json_results


def segm2json(dataset, results):
    bbox_json_results = []
    segm_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det, seg = results[idx]
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                bbox_json_results.append(data)

            # segm results
            # some detectors use different score for det and segm
            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results


def results2json(dataset, results, out_file):
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files

def json2results(input_file):
    result_files = dict()
    result_files['bbox'] = input_file 
    result_files['proposal'] = input_file 
    return result_files

def bbox2result_1image(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]
 
def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    batch_size = len(bboxes)
    assert batch_size == len(labels) == len(bboxes)
	
    print("#########----> all_bboxes =", bboxes)
    print("#########----> all_labels =", labels)
    print("#########----> batch_size =", batch_size)
    
    outputs = []
    for j in range(batch_size):
        bboxes_tmp = bboxes[j]
        labels_tmp = labels[j]
        print("#########----> bboxes =", bboxes)
        print("#########----> labels =", labels)

        if bboxes_tmp.shape[0] == 0:
            output = [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)]
        else:
            output = [bboxes_tmp[labels_tmp == i, :] for i in range(num_classes - 1)]
        outputs.append(output)
    print("#########----> outputs =", outputs)
    return outputs
     
    
