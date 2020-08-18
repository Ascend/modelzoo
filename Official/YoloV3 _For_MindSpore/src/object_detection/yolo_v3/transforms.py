import io
import numpy as np
from PIL import Image
# from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import copy
import cv2

from src.object_detection.yolo_v3.config import Config_yolov3
from src.object_detection.utils.bbox import bbox_iou
from src.object_detection.utils.data_aug import statistic_normalize_img
from src.object_detection.utils.data_aug import get_interp_method
from src.object_detection.utils.data_aug import PIL_image_reshape


def _rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


def _preprocess_true_boxes(true_boxes, anchors, in_shape, num_classes, max_boxes, label_smooth, label_smooth_factor=0.1):
        """
        Introduction
        ------------
            对训练数据的ground truth box进行预处理
        Parameters
        ----------
            true_boxes: ground truth box 形状为[boxes, 5], x_min, y_min, x_max, y_max, class_id
        """
        anchors = np.array(anchors)
        num_layers = anchors.shape[0] // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        true_boxes = np.array(true_boxes, dtype='float32')
        # input_shape = np.array([in_shape, in_shape], dtype='int32')
        input_shape = np.array(in_shape, dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2.
        # trans to box center point
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        # input_shape is [h, w]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
        # true_boxes = [xywh]

        grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
        # grid_shape [h, w]
        y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]),
                            5 + num_classes), dtype='float32') for l in range(num_layers)]
        # y_true [gridy, gridx]
        # 这里扩充维度是为了后面应用广播计算每个图中所有box的anchor互相之间的iou
        anchors = np.expand_dims(anchors, 0)
        anchors_max = anchors / 2.
        anchors_min = -anchors_max
        # 因为之前对box做了padding, 因此需要去除全0行
        valid_mask = boxes_wh[..., 0] > 0

        wh = boxes_wh[valid_mask]
        if len(wh) != 0:
            # 为了应用广播扩充维度
            wh = np.expand_dims(wh, -2)
            # wh 的shape为[box_num, 1, 2]
            # move to original point to compare, and choose the best layer-anchor to set
            boxes_max = wh / 2.
            boxes_min = -boxes_max

            intersect_min = np.maximum(boxes_min, anchors_min)
            intersect_max = np.minimum(boxes_max, anchors_max)
            intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # 找出和ground truth box的iou最大的anchor box,
            # 然后将对应不同比例的负责该ground turth box 的位置置为ground truth box坐标
            best_anchor = np.argmax(iou, axis=-1)
            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')  # grid_y
                        j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')  # grid_x

                        k = anchor_mask[l].index(n)
                        c = true_boxes[t, 4].astype('int32')
                        y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                        y_true[l][j, i, k, 4] = 1.

                        # lable-smooth
                        if label_smooth:
                            sigma = label_smooth_factor/(num_classes-1)
                            y_true[l][j, i, k, 5:] = sigma
                            y_true[l][j, i, k, 5+c] = 1-label_smooth_factor
                        else:
                            y_true[l][j, i, k, 5 + c] = 1.

        # pad_gt_boxes for avoiding dynamic shape
        pad_gt_box0 = np.zeros(shape=[max_boxes, 4], dtype=np.float32)
        pad_gt_box1 = np.zeros(shape=[max_boxes, 4], dtype=np.float32)
        pad_gt_box2 = np.zeros(shape=[max_boxes, 4], dtype=np.float32)

        mask0 = np.reshape(y_true[0][..., 4:5], [-1])
        gt_box0 = np.reshape(y_true[0][..., 0:4], [-1, 4])
        # gt_box [boxes, [x,y,w,h]]
        gt_box0 = gt_box0[mask0 == 1]
        # gt_box0: get all boxes which have object
        pad_gt_box0[:gt_box0.shape[0]] = gt_box0
        # gt_box0.shape[0]: total number of boxes in gt_box0
        # top N of pad_gt_box0 is real box, and after are pad by zero

        mask1 = np.reshape(y_true[1][..., 4:5], [-1])
        gt_box1 = np.reshape(y_true[1][..., 0:4], [-1, 4])
        gt_box1 = gt_box1[mask1 == 1]
        pad_gt_box1[:gt_box1.shape[0]] = gt_box1

        mask2 = np.reshape(y_true[2][..., 4:5], [-1])
        gt_box2 = np.reshape(y_true[2][..., 0:4], [-1, 4])

        gt_box2 = gt_box2[mask2 == 1]
        pad_gt_box2[:gt_box2.shape[0]] = gt_box2
        return y_true[0], y_true[1], y_true[2], pad_gt_box0, pad_gt_box1, pad_gt_box2


def _reshape_data(image, image_size):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    ori_w, ori_h = image.size
    ori_image_shape = np.array([ori_w, ori_h], np.int32)
    # original image shape fir:H sec:W
    h, w = image_size
    interp = get_interp_method(interp=9, sizes=(ori_h, ori_w, h, w))
    image = image.resize((w, h), PIL_image_reshape(interp))
    image_data = statistic_normalize_img(image, statistic_norm=True)
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, axis=-1)
        image_data = np.concatenate([image_data, image_data, image_data], axis=-1)
    image_data = image_data.astype(np.float32)
    return image_data, ori_image_shape


def color_distortion(img, hue, sat, val, device_num):
    hue = _rand(-hue, hue)
    sat = _rand(1, sat) if _rand() < .5 else 1 / _rand(1, sat)
    val = _rand(1, val) if _rand() < .5 else 1 / _rand(1, val)
    if device_num != 1:
        cv2.setNumThreads(1)
    x = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    x = x / 255.
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    # image_data = hsv_to_rgb(x)
    x = x * 255.
    x = x.astype(np.uint8)
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB_FULL)
    return image_data



def filp_PIL_image(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def convert_gray_to_color(img):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
        img = np.concatenate([img, img, img], axis=-1)
    return img


def _is_iou_satisfied_constraint(min_iou, max_iou, box, crop_box):
    iou = bbox_iou(box, crop_box)
    if min_iou <= iou.min() and max_iou >= iou.max():
        return True
    else:
        return False


def _choose_candidate_by_constraints(max_trial, input_w, input_h, image_w, image_h, jitter, box, use_constraints):
    if use_constraints:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )
    else:
        constraints = (
            (None, None),
        )
    # add default candidate
    candidates = [(0, 0, input_w, input_h)]
    for constraint in constraints:
        min_iou, max_iou = constraint
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):
            # box_data should have at least one box
            new_ar = float(input_w) / float(input_h) * _rand(1 - jitter, 1 + jitter) / _rand(1 - jitter, 1 + jitter)
            scale = _rand(0.25, 2)

            if new_ar < 1:
                nh = int(scale * input_h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * input_w)
                nh = int(nw / new_ar)

            dx = int(_rand(0, input_w - nw))
            dy = int(_rand(0, input_h - nh))

            if len(box) > 0:
                t_box = copy.deepcopy(box)
                t_box[:, [0, 2]] = t_box[:, [0, 2]] * float(nw) / float(image_w) + dx
                t_box[:, [1, 3]] = t_box[:, [1, 3]] * float(nh) / float(image_h) + dy
                # is it satisfied constrained? not satisfied will redo the crop

                crop_box = np.array((0, 0, input_w, input_h))
                if not _is_iou_satisfied_constraint(min_iou, max_iou, t_box, crop_box[np.newaxis]):
                    continue
                else:
                    candidates.append((dx, dy, nw, nh))
            else:
                raise Exception("!!! annotation box is less than 1")
    return candidates


def _correct_bbox_by_candidates(candidates, input_w, input_h, image_w, image_h, flip, box, box_data, allow_outside_center):
    while candidates:
        if len(candidates) > 1:
            # ignore default candidate which do not crop
            candidate = candidates.pop(np.random.randint(1, len(candidates)))
        else:
            candidate = candidates.pop(np.random.randint(0, len(candidates)))
        dx, dy, nw, nh = candidate
        t_box = copy.deepcopy(box)
        t_box[:, [0, 2]] = t_box[:, [0, 2]] * float(nw) / float(image_w) + dx
        t_box[:, [1, 3]] = t_box[:, [1, 3]] * float(nh) / float(image_h) + dy
        if flip:
            t_box[:, [0, 2]] = input_w - t_box[:, [2, 0]]

        if allow_outside_center:
            pass
        else:
            t_box = t_box[np.logical_and((t_box[:, 0] + t_box[:, 2])/2. >= 0., (t_box[:, 1] + t_box[:, 3])/2. >= 0.)]
            t_box = t_box[np.logical_and((t_box[:, 0] + t_box[:, 2]) / 2. <= input_w, (t_box[:, 1] + t_box[:, 3]) / 2. <= input_h)]

        # recorrect x, y for case x,y < 0 reset to zero, after dx and dy, some box can smaller than zero
        t_box[:, 0:2][t_box[:, 0:2] < 0] = 0
        # recorrect w,h not higher than input size
        t_box[:, 2][t_box[:, 2] > input_w] = input_w
        t_box[:, 3][t_box[:, 3] > input_h] = input_h
        box_w = t_box[:, 2] - t_box[:, 0]
        box_h = t_box[:, 3] - t_box[:, 1]
        # discard invalid box: w or h smaller than 1 pixel
        t_box = t_box[np.logical_and(box_w > 1, box_h > 1)]

        if t_box.shape[0] > 0:
            # break if number of find t_box
            box_data[: len(t_box)] = t_box
            return box_data, candidate
    raise Exception('all candidates can not satisfied re-correct bbox')


def _data_aug(image, box, jitter, hue, sat, val, image_input_size, max_boxes, anchors, num_classes, max_trial=10, device_num=1):
    """Crop an image randomly with bounding box constraints.

        This data augmentation is used in training of
        Single Shot Multibox Detector [#]_. More details can be found in
        data augmentation section of the original paper.
        .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
           Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
           SSD: Single Shot MultiBox Detector. ECCV 2016."""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image_w, image_h = image.size
    input_h, input_w = image_input_size

    np.random.shuffle(box)
    if len(box) > max_boxes:
        box = box[:max_boxes]
    flip = _rand() < .5
    box_data = np.zeros((max_boxes, 5))

    candidates = _choose_candidate_by_constraints(use_constraints=False,
                                                  max_trial=max_trial,
                                                  input_w=input_w,
                                                  input_h=input_h,
                                                  image_w=image_w,
                                                  image_h=image_h,
                                                  jitter=jitter,
                                                  box=box)
    box_data, candidate = _correct_bbox_by_candidates(candidates=candidates,
                                                      input_w=input_w,
                                                      input_h=input_h,
                                                      image_w=image_w,
                                                      image_h=image_h,
                                                      flip=flip,
                                                      box=box,
                                                      box_data=box_data,
                                                      allow_outside_center=True)
    dx, dy, nw, nh = candidate
    interp = get_interp_method(interp=10)
    image = image.resize((nw, nh), PIL_image_reshape(interp))
    # place image, gray color as back graoud
    new_image = Image.new('RGB', (input_w, input_h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    if flip:
        image = filp_PIL_image(image)

    image = np.array(image)

    image = convert_gray_to_color(image)

    image_data = color_distortion(image, hue, sat, val, device_num)
    image_data = statistic_normalize_img(image_data, statistic_norm=True)

    image_data = image_data.astype(np.float32)

    return image_data, box_data


def preprocess_fn(image, box, config, input_size, device_num):
    config_anchors = config.anchor_scales
    anchors = np.array([list(x) for x in config_anchors])
    max_boxes = config.max_box
    num_classes = config.num_classes
    jitter = config.jitter
    hue = config.hue
    sat = config.saturation
    val = config.value
    # input_size = config.get_input_size()
    # print('input_size {}'.format(input_size))
    image, anno =\
        _data_aug(image, box, jitter=jitter, hue=hue, sat=sat, val=val,
                  image_input_size=input_size, max_boxes=max_boxes, num_classes=num_classes, anchors=anchors, device_num=device_num)
    return image, anno


def reshape_fn(image, img_id, config):
    input_size = config.test_img_shape
    image, ori_image_shape = _reshape_data(image, image_size=input_size)
    return image, ori_image_shape, img_id
