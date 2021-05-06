# coding: utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import base64
import json
import os

import cv2
import numpy as np
from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi


def _parse_arg():
    parser = argparse.ArgumentParser(description="SDK infer")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="Specify the directory of dataset")
    parser.add_argument("-p", "--pipeline", type=str,
                        default="../pipeline/unet_simple_opencv.pipeline",
                        help="Specify the path of pipeline file")
    parser.add_argument("-o", "--output", type=str,
                        default="", help="Specify the infer output directory")
    return parser.parse_args()


def _get_dataset(dataset_dir):
    img_ids = sorted(next(os.walk(dataset_dir))[1])
    for img_id in img_ids:
        img_path = os.path.join(dataset_dir, img_id)
        yield (img_id,
               os.path.join(img_path, "images", f"{img_id}.png"),
               os.path.join(img_path, "masks"))


def _process_mask(mask_dir):
    mask = []
    for mask_path in os.listdir(mask_dir):
        mask_cur = cv2.imread(os.path.join(mask_dir, mask_path),
                              cv2.IMREAD_GRAYSCALE)
        mask.append(mask_cur)
    mask = np.max(mask, axis=0)
    return mask


def _get_stream_manager(pipeline_path):
    stream_mgr_api = StreamManagerApi()
    ret = stream_mgr_api.InitManager()
    if ret != 0:
        print(f"Failed to init Stream manager, ret={ret}")
        exit(1)

    with open(pipeline_path, 'rb') as f:
        pipeline_content = f.read()

    ret = stream_mgr_api.CreateMultipleStreams(pipeline_content)
    if ret != 0:
        print(f"Failed to create stream, ret={ret}")
        exit(1)
    return stream_mgr_api


def _do_infer_image(stream_mgr_api, image_path):
    stream_name = b'unet_mindspore'
    data_input = MxDataInput()
    with open(image_path, 'rb') as f:
        data_input.data = f.read()

    unique_id = stream_mgr_api.SendData(stream_name, 0, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit(1)

    infer_result = stream_mgr_api.GetResult(stream_name, unique_id)
    if infer_result.errorCode != 0:
        print(f"GetResult error. errorCode={infer_result.errorCode},"
              f"errorMsg={infer_result.data.decode()}")
        exit(1)

    infer_result_data = json.loads(infer_result.data.decode())
    content = json.loads(infer_result_data['metaData'][0]['content'])
    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][0]
    data_str = tensor_vec['dataStr']
    tensor_shape = tensor_vec['tensorShape']
    infer_array = np.frombuffer(base64.b64decode(data_str), dtype=np.float32)
    return infer_array.reshape(tensor_shape)


def _calculate_accuracy(infer_image, mask_image):
    mask_image = cv2.resize(mask_image, infer_image.shape[1:3])
    mask_image = mask_image / 255.0
    mask_image = (mask_image > 0.5).astype(np.int)
    mask_image = (np.arange(2) == mask_image[..., None]).astype(np.int)

    infer_image = np.squeeze(infer_image, axis=0)
    inter = np.dot(infer_image.flatten(), mask_image.flatten())
    union = np.dot(infer_image.flatten(), infer_image.flatten()) + \
        np.dot(mask_image.flatten(), mask_image.flatten())

    single_dice = 2 * float(inter) / float(union + 1e-6)
    single_iou = single_dice / (2 - single_dice)
    return single_dice, single_iou


def _save_output(output_dir, image_id, infer_image, mask_image):
    if not output_dir:
        return

    infer_image = np.squeeze(infer_image, axis=0)
    infer_image = infer_image.transpose(2, 0, 1)[1]
    infer_image = cv2.resize(infer_image, mask_image.shape[::-1],
                             interpolation=cv2.INTER_CUBIC)

    os.makedirs(output_dir, exist_ok=True)
    infer_image_path = os.path.join(output_dir, f"{image_id}_infer.png")
    mask_image_path = os.path.join(output_dir, f"{image_id}_mask.png")
    cv2.imwrite(infer_image_path,
                np.uint8(infer_image > 0.5).astype(np.uint8) * 255)
    cv2.imwrite(mask_image_path, np.uint8(mask_image))


def main(_args):
    dice_sum = 0.0
    iou_sum = 0.0
    cnt = 0
    stream_mgr_api = _get_stream_manager(_args.pipeline)
    for image_id, image_path, mask_dir in _get_dataset(_args.dataset):
        infer_image = _do_infer_image(stream_mgr_api, image_path)
        mask_image = _process_mask(mask_dir)
        _save_output(_args.output, image_id, infer_image, mask_image)
        dice, iou = _calculate_accuracy(infer_image, mask_image)
        dice_sum += dice
        iou_sum += iou
        cnt += 1
        print(f"image: {image_id}, dice: {dice}, iou: {iou}")
    print(f"========== Cross Valid dice coeff is: {dice_sum / cnt}")
    print(f"========== Cross Valid IOU is: {iou_sum / cnt}")
    stream_mgr_api.DestroyAllStreams()


if __name__ == "__main__":
    args = _parse_arg()
    main(args)
