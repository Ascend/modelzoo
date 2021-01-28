# Copyright 2020 Huawei Technologies Co., Ltd
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
import os
import cv2
import argparse
import numpy as np
import pathlib
import sys
import glob
import tqdm

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent.parent))

from shapely.geometry import Polygon

from db_config import cfg
from DetectionIoUEvaluator import DetectionIoUEvaluator
from post_process import SegDetectorRepresenter
from utils import load_each_image_lable


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def resize_img(img, max_size=640):
    h, w, _ = img.shape

    ratio = float(max(h, w)) / max_size

    new_h = int((h / ratio // 32) * 32)
    new_w = int((w / ratio // 32) * 32)

    resized_img = cv2.resize(img, dsize=(new_w, new_h))

    input_img = np.zeros([max_size, max_size, 3])
    input_img[0:new_h, 0:new_w, :] = resized_img

    ratio_w = w / new_w
    ratio_h = h / new_h

    return input_img, (ratio_h, ratio_w), (new_h, new_w)


def get_args():
    parser = argparse.ArgumentParser(description='DB-tf')
    parser.add_argument('--imgpath',
                        default='./datasets/total_text/test_images/',
                        type=str)
    parser.add_argument('--show_res', default=True,type=bool)

    args = parser.parse_args()

    return args


def main(args):
    gt_file_dir = cfg.EVAL.LABEL_DIR
    results = []
    evaluator = DetectionIoUEvaluator()

    for file in tqdm.tqdm(list(glob.glob(os.path.join(args.imgpath, "*.jpg"), recursive=True))):

        img_name = file.split('/')[-1].split('.')[0]
        img = cv2.imread(file)
        img_copy = img.copy()

        img = img.astype(np.float32)
        h, w, _ = img.shape
        resized_img, ratio, size = resize_img(img, max_size=800)

        resized_img_path = os.path.join(str(__dir__.parent), "tmp/{}.jpg".format(img_name))

        cv2.imwrite(resized_img_path, resized_img)

        os.system("bash convert_img2bin_inference.sh {} {}".format(resized_img_path, img_name))

        for bin_file in list(glob.glob("./output/*/*.bin", recursive=True)):
            binarize_map = np.reshape(np.fromfile(bin_file, dtype='float32'), (-1, 800, 800, 1))
        
        decoder = SegDetectorRepresenter()
        boxes, scores = decoder([resized_img], binarize_map, True)

        boxes = boxes[0]
        area = h * w
        res_boxes = []
        res_scores = []
        for i, box in enumerate(boxes):
            box[:, 0] *= ratio[1]
            box[:, 1] *= ratio[0]
            if Polygon(box).convex_hull.area > cfg.FILTER_MIN_AREA * area:
                res_boxes.append(box)
                res_scores.append(scores[0][i])
        
        if args.show_res:
            make_dir('./show')
            cv2.imwrite('show/' + img_name + '_binarize_map.jpg', binarize_map[0][0:size[0], 0:size[1], :] * 255)
            for box in res_boxes:
                cv2.polylines(img_copy, [box.astype(np.int).reshape([-1, 1, 2])], True, (0, 255, 0))
            cv2.imwrite('show/' + img_name + '_show.jpg', img_copy)
        
        pre = []
        for p in res_boxes:
            pre.append({
                'points': [tuple(e) for e in p],
                'text': 123,
                'ignore': False,
            })
        gt_file_name = os.path.splitext(img_name)[0] + '.jpg.txt'
        label_info = load_each_image_lable(os.path.join(gt_file_dir, gt_file_name))
        results.append(evaluator.evaluate_image(label_info, pre))

    metrics = evaluator.combine_results(results)
    print(metrics)


if __name__ == "__main__":
    args = get_args()

    main(args)