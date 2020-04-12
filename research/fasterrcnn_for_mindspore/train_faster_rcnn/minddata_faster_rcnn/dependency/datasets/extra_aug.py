import mmcv
import numpy as np
from numpy import random

from .bbox_overlaps import bbox_overlaps

class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        img = img.astype('float32')

        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


class RandomCrop(object):

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels, gt_num):
        h, w, c = img.shape
        loop = 0
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1 or loop > 0:
                len = gt_num.shape[0]

                assert boxes.shape[0] == labels.shape[0]
                assert boxes.shape[0] <= len

                extr = np.zeros((len, 4), dtype=np.float32)
                for i in range(boxes.shape[0]):
                    extr[i] = boxes[i]
                boxes = extr

                extr = np.zeros(len, dtype=np.bool)
                for i in range(labels.shape[0]):
                    extr[i] = True
                gt_num = extr

                extr = np.ones(len, dtype=np.int32)*(-1)
                for i in range(labels.shape[0]):
                    extr[i] = labels[i]
                labels = extr

                return img, boxes, labels, gt_num

            if boxes.shape[0] == gt_num.shape[0]:
                boxes = boxes[gt_num]
                labels = labels[gt_num]

            min_iou = mode
            loop += 1
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)

                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue

                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                len = gt_num.shape[0]

                assert boxes.shape[0] == labels.shape[0]
                assert boxes.shape[0] <= len
                assert boxes.shape[0] > 0

                extr = np.zeros((len, 4), dtype=np.float32)
                for i in range(boxes.shape[0]):
                    extr[i] = boxes[i]
                boxes = extr

                extr = np.zeros(len, dtype=np.bool)
                for i in range(labels.shape[0]):
                    extr[i] = True
                gt_num = extr

                extr = np.ones(len, dtype=np.int32)*(-1)
                for i in range(labels.shape[0]):
                    extr[i] = labels[i]
                labels = extr

                return img, boxes, labels, gt_num


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels

class MixupDetection(object):
    def __init__(self, mixup=None, *args):
        self._mixup = mixup
        self._mixup_args = args

    def set_mixup(self, mixup=None, *args):
        self._mixup = mixup
        self._mixup_args = args

    def __call__(self, img, boxes, labels):
        # first image
        img1, label1 = self._dataset[idx]
        lambd = 1

        # draw a random lambda ratio from distribution
        if self._mixup is not None:
            lambd = max(0, min(1, self._mixup(*self._mixup_args)))

        if lambd >= 1:
            weights1 = np.ones((label1.shape[0], 1))
            label1 = np.hstack((label1, weights1))
            return img1, label1

        # second image
        idx2 = np.random.choice(np.delete(np.arange(len(self)), idx))
        img2, label2 = self._dataset[idx2]

        # mixup two images
        height = max(img1.shape[0], img2.shape[0])
        width = max(img1.shape[1], img2.shape[1])
        mix_img = mx.nd.zeros(shape=(height, width, 3), dtype='float32')
        mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * lambd
        mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - lambd)
        mix_img = mix_img.astype('uint8')
        y1 = np.hstack((label1, np.full((label1.shape[0], 1), lambd)))
        y2 = np.hstack((label2, np.full((label2.shape[0], 1), 1. - lambd)))
        mix_label = np.vstack((y1, y2))
        return mix_img, mix_label

