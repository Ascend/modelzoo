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

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from model_darknet19 import darknet
from decode import decode
from utils import preprocess_image, postprocess, draw_detection
from config import anchors, class_names
import os 

def main():
    labels_to_names={
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 6: 'train', 7: 'truck',8: 'boat', 9:'traffic light', 10:'fire hydrant', 11:'stop sign',12:'parking meter', 13:'bench', 14:'bird',
    15:'cat',16: 'dog',17:'horse', 18:'sheep',19:'cow',20:'elephant',21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'bandbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',31:'snowboard',32:'sports ball',
    33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',
    49:'orange',50:'broccoli',51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'couch',58:'pottedplant',59:'bed',60:'diningtable',61:'toilet',62:'tv',63:'laptop',64:'mouse',65:'remote',66:'keyboard',
    67:'cellphone',68:'microwave',69:'oven',70:'toaster',71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair direr',79:'toothbrush'}
    img_dir = "./data/pascal_voc/VOCdevkit/VOC2007_test/JPEGImages"
    for filename in os.listdir(img_dir):
        input_size = (416,416)
        
        image = cv2.imread(img_dir + '/' + filename)
        image_shape = image.shape[:2] #只取wh，channel=3不取

        # copy、resize416*416、归一化、在第0维增加存放batchsize维度
        image_cp = preprocess_image(image,input_size)
        tf.reset_default_graph()  #运行到第2张就报错，需要加上这句，清除默认图形堆栈并充值默认图形
        
        # 【1】输入图片进入darknet19网络得到特征图，并进行解码得到：xmin xmax表示的边界框、置信度、类别概率
        tf_image = tf.placeholder(tf.float32,[1,input_size[0],input_size[1],3])
        model_output = darknet(tf_image) # darknet19网络输出的特征图
        output_sizes = input_size[0]//32, input_size[1]//32 # 特征图尺寸是图片下采样32倍
        output_decoded = decode(model_output=model_output,output_sizes=output_sizes,
                                   num_class=len(class_names),anchors=anchors)  # 解码

        model_path = "./yolov2_model/checkpoint_dir/yolo2_coco.ckpt"
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,model_path)
            bboxes,obj_probs,class_probs = sess.run(output_decoded,feed_dict={tf_image:image_cp})

        # 【2】筛选解码后的回归边界框——NMS(post process后期处理)
        bboxes,scores,class_max_index = postprocess(bboxes,obj_probs,class_probs,image_shape=image_shape)
        label_path_txt = "./map_mul/detections_npu/"
        with open(os.path.join(label_path_txt + filename.split('.')[0] + '.txt'), 'a+') as f:
            for i in range(len(scores)):
                if " " in labels_to_names[class_max_index[i]]:
                    labels_to_name = labels_to_names[class_max_index[i]].split(' ')[0] + labels_to_names[class_max_index[i]].split(' ')[1]
                    f.write(labels_to_name + " " + str(scores[i]) + " " + str(bboxes[i][0])+ " " + str(bboxes[i][1])+ " " + str(bboxes[i][2])+ " " + str(bboxes[i][3]) + '\n')
                else:
                    f.write(labels_to_names[class_max_index[i]] + " " + str(scores[i]) + " " + str(bboxes[i][0])+ " " + str(bboxes[i][1])+ " " + str(bboxes[i][2])+ " " + str(bboxes[i][3]) + '\n')
        
        
        # 【3】绘制筛选后的边界框
        #print('-----',filename)
        #img_detection = draw_detection(image, bboxes, scores, class_max_index, class_names)
        #cv2.imwrite(f"./VOC2007_jpeg_demo/" + filename.split('.')[0]+'_' + "detection.jpg", img_detection)
        print('YOLO_v2 detection has done!')
        #cv2.imshow("detection_results", img_detection)
        #cv2.waitKey(0)

if __name__ == '__main__':
    main()
