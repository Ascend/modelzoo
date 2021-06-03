# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import json
import logging
import os
import time

import cv2
import mmcv
import numpy as np
from PIL import Image

from api.draw_predict import draw_label
from api.infer import SdkApi
from config import config as cfg
from eval.eval_by_sdk import get_eval_result


def parser_args():
    parser = argparse.ArgumentParser(description="maskrcnn inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=True,
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="config/maskrcnn_ms.pipeline",
        help="image file path. The default is 'config/maskrcnn_ms.pipeline'. ")
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="dvpp",
        help=
        "rgb: high-precision, dvpp: high performance. The default is 'dvpp'.")
    parser.add_argument(
        "--infer_mode",
        type=str,
        required=False,
        default="infer",
        help=
        "infer:only infer, eval: accuracy evaluation. The default is 'infer'.")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="../data/infer_result",
        help=
        "cache dir of inference result. The default is '../data/infer_result'."
    )

    parser.add_argument("--ann_file",
                        type=str,
                        required=False,
                        help="eval ann_file.")

    args = parser.parse_args()
    return args


def get_img_metas(file_name):
    img = Image.open(file_name)
    img_size = img.size

    org_width, org_height = img_size
    resize_ratio = cfg.MODEL_WIDTH / org_width
    if resize_ratio > cfg.MODEL_HEIGHT / org_height:
        resize_ratio = cfg.MODEL_HEIGHT / org_height

    img_metas = np.array([img_size[1], img_size[0]] +
                         [resize_ratio, resize_ratio])
    return img_metas


def process_img(img_file):
    img = cv2.imread(img_file)
    model_img = mmcv.imrescale(img, (cfg.MODEL_WIDTH, cfg.MODEL_HEIGHT))
    if model_img.shape[0] > cfg.MODEL_HEIGHT:
        model_img = mmcv.imrescale(model_img,
                                   (cfg.MODEL_HEIGHT, cfg.MODEL_HEIGHT))
    pad_img = np.zeros(
        (cfg.MODEL_HEIGHT, cfg.MODEL_WIDTH, 3)).astype(model_img.dtype)
    pad_img[0:model_img.shape[0], 0:model_img.shape[1], :] = model_img
    pad_img.astype(np.float16)
    return pad_img


def image_inference(pipeline_path, stream_name, img_dir, result_dir,
                    replace_last, model_type):
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0
    img_metas_plugin_id = 1
    print(f"\nBegin to inference for {img_dir}.\n\n")

    file_list = os.listdir(img_dir)
    total_len = len(file_list)
    for img_id, file_name in enumerate(file_list):
        if not file_name.lower().endswith((".jpg", "jpeg")):
            continue
        file_path = os.path.join(img_dir, file_name)
        save_path = os.path.join(result_dir,
                                 f"{os.path.splitext(file_name)[0]}.json")
        if not replace_last and os.path.exists(save_path):
            print(
                f"The infer result json({save_path}) has existed, will be skip."
            )
            continue

        try:
            if model_type == 'dvpp':
                with open(file_path, "rb") as fp:
                    data = fp.read()
                sdk_api.send_data_input(stream_name, img_data_plugin_id, data)
            else:
                img_np = process_img(file_path)
                sdk_api.send_img_input(stream_name,
                                       img_data_plugin_id, "appsrc0",
                                       img_np.tobytes(), img_np.shape)

            # set image data
            img_metas = get_img_metas(file_path).astype(np.float32)
            sdk_api.send_tensor_input(stream_name, img_metas_plugin_id,
                                      "appsrc1", img_metas.tobytes(), [1, 4],
                                      cfg.TENSOR_DTYPE_FLOAT32)

            start_time = time.time()
            result = sdk_api.get_result(stream_name)
            end_time = time.time() - start_time

            with open(save_path, "w") as fp:
                fp.write(json.dumps(result))
            print(
                f"End-2end inference, file_name: {file_path}, {img_id + 1}/{total_len}, elapsed_time: {end_time}.\n"
            )

            draw_label(save_path, file_path, result_dir)
        except Exception as ex:
            logging.exception("Unknown error, msg(%s).", ex)


if __name__ == "__main__":
    args = parser_args()

    replace_last = True
    stream_name = cfg.STREAM_NAME.encode("utf-8")
    image_inference(args.pipeline_path, stream_name, args.img_path,
                    args.infer_result_dir, replace_last, args.model_type)
    if args.infer_mode == "eval":
        print("Infer end.")
        print("Begin to eval...")
        get_eval_result(args.ann_file, args.infer_result_dir)
