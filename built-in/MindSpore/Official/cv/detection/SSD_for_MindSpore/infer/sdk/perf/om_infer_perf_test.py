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

import json
import os
import threading
import time
from datetime import datetime

import cv2
from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi
from absl import app
from absl import flags

BOXED_IMG_DIR = None
TXT_DIR = None
PERF_REPORT_TXT = None
DET_RESULT_RESIZED_JSON = None
DET_RESULT_JSON = None

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name="img_dir", default=None, help="Directory of images to infer"
)

flags.DEFINE_string(
    name="pipeline_config",
    default=None,
    help="Path name of pipeline configuration file of " "mxManufacture.",
)

flags.DEFINE_string(
    name="infer_stream_name",
    default=None,
    help="Infer stream name configured in pipeline "
    "configuration file of mxManufacture",
)

flags.DEFINE_boolean(
    name="draw_box",
    default=True,
    help="Whether out put the inferred image with bounding box",
)

flags.DEFINE_float(
    name="score_thresh_for_draw",
    default=0.5,
    help="Draw bounding box if the confidence greater than.",
)

flags.DEFINE_string(
    name="output_dir",
    default=None,
    help="Where to out put the inferred image with bounding box, if the "
    "draw_box is set, this parameter must be set.",
)

flags.DEFINE_integer(
    name="display_step",
    default=100,
    help="Every how many images to print the inference real speed and "
    "progress.",
)

flags.mark_flag_as_required("img_dir")
flags.mark_flag_as_required("pipeline_config")
flags.mark_flag_as_required("infer_stream_name")
flags.mark_flag_as_required("output_dir")


def draw_image(input_image, bboxes, output_img):
    # 原图
    image = cv2.imread(input_image)

    # 模型推理输出数据，需要往后处理代码中增加几行输出文档的代码
    color_index_dict = {
        0: (0, 0, 255),
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (255, 255, 0),
        4: (255, 0, 255),
        5: (0, 255, 255),
        6: (255, 128, 0),
        7: (128, 128, 255),
        8: (0, 255, 128),
        9: (128, 128, 0),
    }
    for index, bbox in enumerate(bboxes):
        color_key = index % 10
        color = color_index_dict.get(color_key)
        # Coordinate must be integer.
        bbox = list(map(lambda cor: int(cor), bbox))
        # pdb.set_trace()
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # 新的图片
    cv2.imwrite(output_img, image)


def draw_img_fun(img_id, bboxes):
    img_name = "%012d.jpg" % img_id
    input_img_dir = FLAGS.img_dir
    input_img = os.path.join(input_img_dir, img_name)

    boxed_img = os.path.join(BOXED_IMG_DIR, img_name)
    draw_image(input_img, bboxes, boxed_img)

def trans_class_id(k):
    if k >= 1 and k <= 11:
        return k
    elif k >= 12 and k <= 24:
        return k + 1
    elif k >= 25 and k <= 26:
        return k + 2
    elif k >= 27 and k <= 40:
        return k + 4
    elif k >= 41 and k <= 60:
        return k + 5
    elif k == 61:
        return k + 6
    elif k == 62:
        return k + 8
    elif k >= 63 and k <= 73:
        return k + 9
    elif k >= 74 and k <= 80:
        return k + 10

def parse_result(img_id, json_content):
    obj_list = json.loads(json_content).get("MxpiObject", [])
    pic_infer_dict_list = []
    bboxes_for_drawing = []
    txt_lines_list = []
    for o in obj_list:
        x0, y0, x1, y1 = (
            round(o.get("x0"), 4),
            round(o.get("y0"), 4),
            round(o.get("x1"), 4),
            round(o.get("y1"), 4),
        )
        # For MAP
        bbox_for_map = [x0, y0, round(x1 - x0, 4), round(y1 - y0, 4)]
        # For drawing bounding box.
        bbox_for_drawing = [int(x0), int(y0), int(x1), int(y1)]
        # calculation
        tmp_list = [
            o.get("classVec")[0].get("classId"),
            o.get("classVec")[0].get("confidence"),
            x0,
            y0,
            x1,
            y1,
        ]
        tmp_list = map(str, tmp_list)
        txt_lines_list.append(" ".join(tmp_list))
        category_id = o.get("classVec")[0].get("classId") # 1-80, GT:1-90
        category_id = trans_class_id(category_id)
        score = o.get("classVec")[0].get("confidence")

        pic_infer_dict_list.append(
            dict(
                image_id=img_id,
                bbox=bbox_for_map,
                category_id=category_id,
                score=score,
            )
        )

        if FLAGS.draw_box and score > FLAGS.score_thresh_for_draw:
            bboxes_for_drawing.append(bbox_for_drawing[:])

    txt_name = "%012d.txt" % img_id
    txt_full_name = os.path.join(TXT_DIR, txt_name)
    with open(txt_full_name, "w") as fw:
        fw.write("\n".join(txt_lines_list))
        fw.write("\n")

    if FLAGS.draw_box:
        draw_img_fun(img_id, bboxes_for_drawing)

    return pic_infer_dict_list


def send_many_images(stream_manager_api):
    data_input = MxDataInput()
    input_dir = FLAGS.img_dir

    imgs = os.listdir(input_dir)
    img_ids = map(lambda img_name: int(img_name.split(".")[0]), imgs)
    img_file_names = [
        os.path.join(input_dir, img_name)
        for img_name in imgs
        if "boxed" not in img_name
    ]
    start = time.time()
    uuid_list = []
    for img_file_name in img_file_names:
        with open(img_file_name, "rb") as f:
            data_input.data = f.read()

        in_plugin_id = 0
        unique_id = stream_manager_api.SendDataWithUniqueId(
            FLAGS.infer_stream_name.encode("utf8"), in_plugin_id, data_input
        )
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()

        uuid_list.append(unique_id)

    end = time.time()
    time_str = (
        f"\nSend all images data took: {round((end-start)*1000, 2)} ms\n"
    )

    with open(PERF_REPORT_TXT, "a+") as fw:
        fw.write(time_str)

    return zip(uuid_list, img_ids)


def get_all_images_result(uuid_img_id_zip, stream_manager_api):
    start_secs = time.time()
    all_infer_dict_list = []
    report_file = open(PERF_REPORT_TXT, "a+")
    img_num = len(
        [
            img_name
            for img_name in os.listdir(FLAGS.img_dir)
            if "boxed" not in img_name
        ]
    )
    for index, (uuid, img_id) in enumerate(uuid_img_id_zip):
        infer_result = stream_manager_api.GetResultWithUniqueId(
            FLAGS.infer_stream_name.encode("utf8"), uuid, 3000
        )
        if (index + 1) % FLAGS.display_step == 0:
            cur_secs = time.time()
            acc_secs = round(cur_secs - start_secs, 4)
            real_speed = round((cur_secs - start_secs) * 1000 / (index + 1), 4)
            perf_detail = (
                f"Inferred: {index + 1}/{img_num} images; "
                f"took: {acc_secs} seconds; "
                f"average inference speed at: {real_speed} ms/image\n"
            )
            threading.Thread(
                target=write_speed_detail, args=(perf_detail, report_file)
            ).start()

        threading.Thread(
            target=parse_infer_result,
            args=(all_infer_dict_list, img_id, infer_result),
        ).start()

    finish_secs = time.time()
    avg_infer_speed = round((finish_secs - start_secs) * 1000 / img_num, 4)
    final_perf = (
        f"Infer finished, average speed:{avg_infer_speed} "
        f"ms/image for {img_num} images.\n\n"
    )
    report_file.write(final_perf)
    report_file.close()

    return all_infer_dict_list


def write_speed_detail(perf_detail, report_file):
    report_file.write(perf_detail)
    report_file.flush()


def parse_infer_result(all_infer_dict_list, img_id, infer_result):
    if infer_result.errorCode != 0:
        print(
            "GetResultWithUniqueId error. errorCode=%d, errorMsg=%s"
            % (infer_result.errorCode, infer_result.data.decode())
        )
        exit()

    info_json_str = infer_result.data.decode()
    all_infer_dict_list.extend(parse_result(img_id, info_json_str))


def infer_img(stream_manager_api, input_image, infer_stream_name):
    """Infer one input image with specified stream name configured in
    mxManufacture pipeline config file.

    :param stream_manager_api:
    :param input_image: file name the image to be inferred.
    :param infer_stream_name:
    :return:
    """
    data_input = MxDataInput()
    with open(input_image, "rb") as f:
        data_input.data = f.read()

    in_plugin_id = 0
    unique_id = stream_manager_api.SendDataWithUniqueId(
        infer_stream_name.encode("utf8"), in_plugin_id, data_input
    )
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and unique_id.
    infer_result = stream_manager_api.GetResultWithUniqueId(
        infer_stream_name.encode("utf8"), unique_id, 3000
    )
    end = time.time()
    print(f"Infer time: {end} s.")
    if infer_result.errorCode != 0:
        print(
            "GetResultWithUniqueId error. errorCode=%d, errorMsg=%s"
            % (infer_result.errorCode, infer_result.data.decode())
        )
        exit()

    info_json_str = infer_result.data.decode()
    img_id = int(os.path.basename(input_image).split(".")[0])
    return parse_result(img_id, info_json_str)


def prepare_infer_stream():
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    # create streams by pipeline config file
    with open(FLAGS.pipeline_config, "rb") as f:
        pipelineStr = f.read()

    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    return stream_manager_api


def infer_imgs():
    stream_manager_api = prepare_infer_stream()
    uuid_img_id_zip = send_many_images(stream_manager_api)
    all_infer_dict_list = get_all_images_result(
        uuid_img_id_zip, stream_manager_api
    )
    with open(DET_RESULT_JSON, "w") as fw:
        fw.write(json.dumps(all_infer_dict_list))

    stream_manager_api.DestroyAllStreams()


def main(unused_arg):
    global BOXED_IMG_DIR
    global TXT_DIR
    global PERF_REPORT_TXT
    global DET_RESULT_JSON
    """
    output_dir
    |_boxed_imgs
    |_txts
    |_per_report_npu.txt
    |_det_result_npu.json

    """

    BOXED_IMG_DIR = os.path.join(FLAGS.output_dir, "boxed_imgs")
    TXT_DIR = os.path.join(FLAGS.output_dir, "txts")
    PERF_REPORT_TXT = os.path.join(FLAGS.output_dir, "om_perf_report.txt")
    DET_RESULT_JSON = os.path.join(FLAGS.output_dir, "om_det_result.json")

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    if not os.path.exists(TXT_DIR):
        os.makedirs(TXT_DIR)

    if FLAGS.draw_box and not os.path.exists(BOXED_IMG_DIR):
        os.makedirs(BOXED_IMG_DIR)

    now_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head_info = f"{'-'*50}Perf Test On NPU starts @ {now_time_str}{'-'*50}\n"
    with open(PERF_REPORT_TXT, "a+") as fw:
        fw.write(head_info)

    infer_imgs()

    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tail_info = f"{'-'*50}Perf Test On NPU ends @ {end_time_str}{'-'*50}\n"
    with open(PERF_REPORT_TXT, "a+") as fw:
        fw.write(tail_info)


if __name__ == "__main__":
    app.run(main)
