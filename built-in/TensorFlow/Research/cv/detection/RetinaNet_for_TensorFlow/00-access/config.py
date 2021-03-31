CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle",
         "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person",
         "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

A = 9  #number of anchor
K = 20  #number of class

area = [32, 64, 128, 256, 512]
aspect_ratio = [0.5, 2.0, 1.0]
scales = [1.0, 1.26, 1.59]

BATCH_SIZE = 2
IMG_H = 512
IMG_W = 512
WEIGHT_DECAY = 0.0001
LEARNING_RATE = 0.001

XML_PATH = "./VOCdevkit/VOC2007/Annotations/"
IMG_PATH = "./VOCdevkit/VOC2007/JPEGImages/"
