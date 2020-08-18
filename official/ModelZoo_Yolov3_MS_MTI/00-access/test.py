import os
from mindspore import context
devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Davinci", save_graphs=True, device_id=devid)

import argparse
import numpy as np
import datetime
import time
import sys
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mindspore import Tensor
from mindspore.train import Model, ParallelMode
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore as ms

from src.object_detection.yolo_v3.yolo import yolov3_darknet53
from src.common.utils.logging import get_logger
from src.object_detection.yolo_v3.yolo_dataset import create_yolo_dataset
from src.object_detection.yolo_v3.config import Config_yolov3


class Redirct(object):
    def __init__(self):
        self.content = ""

    def write(self, str):
        self.content += str

    def flush(self):
        self.content = ""


class DetectionEngine(object):
    def __init__(self, args):
        self.ignore_threshold = args.ignore_threshold
        self.labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                       'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                       'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                       'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                       'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.num_classes = len(self.labels)
        self.results = {}   # img_id->class
        self.file_path = ''  # path to save predict result
        self.save_prefix = args.outputs_dir
        self.annFile = args.annFile  # path to load annfile
        self._coco = COCO(self.annFile)
        self._img_ids = list(sorted(self._coco.imgs.keys()))
        self.det_boxes = []
        self.nms_thresh = args.nms_thresh
        self.coco_catIds = self._coco.getCatIds()

    def do_NMS_for_results(self):
        for img_id in self.results:
            for clsi in self.results[img_id]:
                dets = self.results[img_id][clsi]
                dets = np.array(dets)
                keep_index = self._nms(dets, self.nms_thresh)

                keep_box = [{'image_id': int(img_id),
                             'category_id': int(clsi),
                             'bbox': list(dets[i][:4].astype(float)),
                             'score': dets[i][4].astype(float)}
                            for i in keep_index]
                self.det_boxes.extend(keep_box)

    def _nms(self, dets, thresh):
        # conver xywh -> xmin ymin xmax ymax
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def write_result(self):
        import json
        import datetime
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.det_boxes, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def get_eval_result(self):
        cocoGt = COCO(self.annFile)
        cocoDt = cocoGt.loadRes(self.file_path)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        # !!!!
        # cocoEval.params.imgIds = self._img_ids[:320]
        cocoEval.evaluate()
        cocoEval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        cocoEval.summarize()
        sys.stdout = stdout
        return rdct.content

    def detect(self, outputs, batch, image_shape, image_id):
        outputs_num = len(outputs)
        # output [|32, 52, 52, 3, 85| ]
        for batch_id in range(batch):
            for out_id in range(outputs_num):
                # 32, 52, 52, 3, 85
                out_item = outputs[out_id]
                # 52, 52, 3, 85
                out_item_single = out_item[batch_id, :]
                # get number of items in one head, [B, gx, gy, anchors, 5+80]
                dimensions = out_item_single.shape[:-1]
                out_num = 1
                for d in dimensions:
                    out_num *= d
                ori_w, ori_h = image_shape[batch_id]
                img_id = int(image_id[batch_id])
                x = out_item_single[..., 0] * ori_w
                y = out_item_single[..., 1] * ori_h
                w = out_item_single[..., 2] * ori_w
                h = out_item_single[..., 3] * ori_h

                conf = out_item_single[..., 4:5]
                cls_emb = out_item_single[..., 5:]

                cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
                x = x.reshape(-1)
                y = y.reshape(-1)
                w = w.reshape(-1)
                h = h.reshape(-1)
                cls_emb = cls_emb.reshape(-1, 80)
                conf = conf.reshape(-1)
                cls_argmax = cls_argmax.reshape(-1)

                x_top_left = x - w/2.
                y_top_left = y - h/2.
                # creat all False
                flag = np.random.random(cls_emb.shape)>sys.maxsize
                for i in range(flag.shape[0]):
                    c = cls_argmax[i]
                    flag[i, c] = True
                confidence = cls_emb[flag]*conf
                for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left, w, h, confidence, cls_argmax):
                    if confi < self.ignore_threshold:
                        continue
                    if img_id not in self.results:
                        self.results[img_id] = defaultdict(list)
                    x_lefti = max(0, x_lefti)
                    y_lefti = max(0, y_lefti)
                    wi = min(wi, ori_w)
                    hi = min(hi, ori_h)
                    # transform catId to match coco
                    coco_clsi = self.coco_catIds[clsi]
                    self.results[img_id][coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])


def parse_args(cloud_args={}):
    parser = argparse.ArgumentParser('mindspore coco testing')

    # dataset related
    parser.add_argument('--data_dir', type=str, default='', help='train data dir')
    parser.add_argument('--per_batch_size', default=1, type=int, help='batch size for per gpu')

    # network related
    parser.add_argument('--pretrained', default='', type=str, help='model_path, local pretrained model to load')

    # logging related
    parser.add_argument('--log_path', type=str, default='outputs/', help='checkpoint save location')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')

    # detect_related
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='threshold for NMS')
    parser.add_argument('--annFile', type=str, default='', help='path to annotation')
    parser.add_argument('--testing_shape', type=str, default='', help='shape for test ')
    parser.add_argument('--ignore_threshold', type=float, default=0.001, help='threshold to throw low quality boxes')

    args, _ = parser.parse_known_args()
    args = merge_args(args, cloud_args)

    args.data_root = os.path.join(args.data_dir, 'val2014')
    args.annFile = os.path.join(args.data_dir, 'annotations/instances_val2014.json')


    return args


def merge_args(args, cloud_args):
    args_dict = vars(args)
    if isinstance(cloud_args, dict):
        for key in cloud_args.keys():
            val = cloud_args[key]
            if key in args_dict and val:
                arg_type = type(args_dict[key])
                if arg_type is not type(None):
                    val = arg_type(val)
                args_dict[key] = val
    return args

def conver_testing_shape(args):
    testing_shape = [int(args.testing_shape), int(args.testing_shape)]
    return testing_shape

def test(cloud_args={}):
    start_time = time.time()
    args = parse_args(cloud_args)

    # init distributed
    if args.is_distributed:
        # TODO implement distributed test
        raise NotImplementedError('distributed test has not implement')
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()

    # logger
    args.outputs_dir = os.path.join(args.log_path,
        datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    args.logger = get_logger(args.outputs_dir, args.rank)

    context.reset_auto_parallel_context()
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    else:
        parallel_mode = ParallelMode.STAND_ALONE
        degree = 1
    context.set_auto_parallel_context(parallel_mode=parallel_mode, mirror_mean=True, device_num=degree)

    args.logger.info('Creating Network....')
    network = yolov3_darknet53(is_training=False)

    args.logger.info(args.pretrained)
    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.pretrained))
    else:
        args.logger.info('{} not exists or not a pre-trained file'.format(args.pretrained))
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(args.pretrained))
        exit(1)

    data_root = args.data_root
    annFile = args.annFile

    config = Config_yolov3()
    if args.testing_shape:
        config.test_img_shape = conver_testing_shape(args)

    ds, data_size = create_yolo_dataset(data_root, annFile, is_training=False, batch_size=args.per_batch_size,
                                        max_epoch=1, device_num=args.group_size, rank=args.rank, shuffle=False, config=config)

    args.logger.info('testing shape : {}'.format(config.test_img_shape))
    args.logger.info('totol {} images to eval'.format(data_size))

    network.set_train(False)

    # init detection engine
    detection = DetectionEngine(args)
    
    input_shape = Tensor(tuple(config.test_img_shape), ms.float32)
    args.logger.info('Start inference....')
    for i, data in enumerate(ds.create_dict_iterator()):
        image = Tensor(data["image"])

        image_shape = Tensor(data["image_shape"])
        image_id = Tensor(data["img_id"])

        prediction = network(image, input_shape)
        output_big, output_me, output_small = prediction
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        image_id = image_id.asnumpy()
        image_shape = image_shape.asnumpy()

        detection.detect([output_small, output_me, output_big], args.per_batch_size, image_shape, image_id)
        if i % 1000 == 0:
            args.logger.info('Processing... {:.2f}% '.format(i*args.per_batch_size/data_size*100))
        # if i==320:
        #     break
    
    args.logger.info('Calculating mAP...')
    detection.do_NMS_for_results()
    result_file_path = detection.write_result()
    args.logger.info('result file path: {}'.format(result_file_path))
    eval_result = detection.get_eval_result()

    cost_time = time.time()-start_time
    args.logger.info('\n=============coco eval reulst=========\n' + eval_result)
    args.logger.info('testing cost time {:.2f}h'.format(cost_time/3600.))


if __name__ == "__main__":
    test()
