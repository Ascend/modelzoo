# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
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

from npu_bridge.npu_init import *
from networks import backbone
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from utils import generate_anchors, draw_bbox, recover_ImgAndBbox_scale
from config import IMG_H, IMG_W, CLASSES, K
from ops import top_k_score_bbox, sigmoid, offset2bbox
anchors_p3 = generate_anchors(area=32, stride=8)
anchors_p4 = generate_anchors(area=64, stride=16)
anchors_p5 = generate_anchors(area=128, stride=32)
anchors_p6 = generate_anchors(area=256, stride=64)
anchors_p7 = generate_anchors(area=512, stride=128)
anchors = np.concatenate((anchors_p3, anchors_p4, anchors_p5, anchors_p6, anchors_p7), axis=0)

class Inference():

    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 3])
        self.is_training = tf.placeholder(tf.bool)
        (_, _, class_logits_dict, box_logits_dict) = backbone(self.inputs, self.is_training)
        (class_logits_dict['P3'], class_logits_dict['P4'], class_logits_dict['P5'], class_logits_dict['P6'], class_logits_dict['P7']) = (tf.reshape(class_logits_dict['P3'], [(- 1), K]), tf.reshape(class_logits_dict['P4'], [(- 1), K]), tf.reshape(class_logits_dict['P5'], [(- 1), K]), tf.reshape(class_logits_dict['P6'], [(- 1), K]), tf.reshape(class_logits_dict['P7'], [(- 1), K]))
        (box_logits_dict['P3'], box_logits_dict['P4'], box_logits_dict['P5'], box_logits_dict['P6'], box_logits_dict['P7']) = (tf.reshape(box_logits_dict['P3'], [(- 1), 4]), tf.reshape(box_logits_dict['P4'], [(- 1), 4]), tf.reshape(box_logits_dict['P5'], [(- 1), 4]), tf.reshape(box_logits_dict['P6'], [(- 1), 4]), tf.reshape(box_logits_dict['P7'], [(- 1), 4]))
        (P3_class_pred, P4_class_pred, P5_class_pred, P6_class_pred, P7_class_pred) = (sigmoid(class_logits_dict['P3']), sigmoid(class_logits_dict['P4']), sigmoid(class_logits_dict['P5']), sigmoid(class_logits_dict['P6']), sigmoid(class_logits_dict['P7']))
        (P3_bbox_pred, P4_bbox_pred, P5_bbox_pred, P6_bbox_pred, P7_bbox_pred) = (box_logits_dict['P3'], box_logits_dict['P4'], box_logits_dict['P5'], box_logits_dict['P6'], box_logits_dict['P7'])
        (P3_topK_score, P3_topK_bbox, P3_topK_anchors, P3_topK_class) = top_k_score_bbox(P3_class_pred, P3_bbox_pred, anchors_p3, threshold=0.05, k=1000)
        (P4_topK_score, P4_topK_bbox, P4_topK_anchors, P4_topK_class) = top_k_score_bbox(P4_class_pred, P4_bbox_pred, anchors_p4, threshold=0.05, k=1000)
        (P5_topK_score, P5_topK_bbox, P5_topK_anchors, P5_topK_class) = top_k_score_bbox(P5_class_pred, P5_bbox_pred, anchors_p5, threshold=0.05, k=1000)
        (P6_topK_score, P6_topK_bbox, P6_topK_anchors, P6_topK_class) = top_k_score_bbox(P6_class_pred, P6_bbox_pred, anchors_p6, threshold=0.05, k=1000)
        (P7_topK_score, P7_topK_bbox, P7_topK_anchors, P7_topK_class) = top_k_score_bbox(P7_class_pred, P7_bbox_pred, anchors_p7, threshold=0.05, k=1000)
        self.topK_score = tf.concat([P3_topK_score, P4_topK_score, P5_topK_score, P6_topK_score, P7_topK_score], axis=0)
        self.topK_bbox = tf.concat([P3_topK_bbox, P4_topK_bbox, P5_topK_bbox, P6_topK_bbox, P7_topK_bbox], axis=0)
        self.topK_anchors = tf.concat([P3_topK_anchors, P4_topK_anchors, P5_topK_anchors, P6_topK_anchors, P7_topK_anchors], axis=0)
        self.topK_class = tf.concat([P3_topK_class, P4_topK_class, P5_topK_class, P6_topK_class, P7_topK_class], axis=0)
        self.bbox = offset2bbox(self.topK_anchors, self.topK_bbox)
        self.nms_idx = tf.image.non_max_suppression(self.bbox, self.topK_score, max_output_size=300)
        self.sess = tf.Session(config=npu_session_config_init())
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, './model/model.ckpt')

    def __call__(self, IMG):
        IMG_ = np.array(Image.fromarray(IMG).resize([IMG_W, IMG_H]))
        [NMS_IDX, BBOX, TOPK_CLASS, TOPK_SCORE] = self.sess.run([self.nms_idx, self.bbox, self.topK_class, self.topK_score], feed_dict={self.inputs: ((IMG_[np.newaxis] / 127.5) - 1.0), self.is_training: True})
        for i in NMS_IDX:
            if (TOPK_SCORE[i] > 0.5):
                IMG = draw_bbox(IMG, recover_ImgAndBbox_scale(IMG, BBOX[i]), CLASSES[TOPK_CLASS[i]])
        return IMG

def detect_video(vid_path, inference):
    cap = cv2.VideoCapture(vid_path)
    while cap.isOpened():
        (ret, frame) = cap.read()
        if (ret == True):
            frame = np.array(frame)
            frame = np.array(Image.fromarray(frame).rotate(270))
            frame = inference(frame)
            cv2.imshow('Frame', np.uint8(frame))
            if ((cv2.waitKey(25) & 255) == ord('q')):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
if (__name__ == '__main__'):
    IMG_PATH = 'C:/Users/gmt/Desktop/cats/65.jpg'
    inference = Inference()
    IMG = np.array(Image.open(IMG_PATH))
    IMG = inference(IMG)
    Image.fromarray(IMG).show()
    Image.fromarray(IMG).save('1.jpg')
