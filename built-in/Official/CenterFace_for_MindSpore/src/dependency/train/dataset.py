###modified based on centernet###
#MIT License
#Copyright (c) 2019 Xingyi Zhou
#All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import datetime

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import cv2
import torch
import torch.utils.data as data
import matplotlib.image as mpimg

import random
import math
from PIL import Image
import re
from torch._six import container_abcs, string_classes, int_classes

def get_dataLoader(opt, args):
    torch.manual_seed(opt.seed)
    Dataset = get_dataset()

    train_dataset = Dataset(opt, 'train')
    # use ddp sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.group_size, rank=args.rank)
    train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=opt.master_batch_size,
      shuffle=False,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True,
      collate_fn=Multiposebatch,
      sampler=train_sampler
    )
    return train_loader, train_sampler

def Data_anchor_sample(image, anns):
    maxSize = 12000
    infDistance = 9999999

    boxes = []
    for ann in anns:
        boxes.append([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]])
    boxes = np.asarray(boxes, dtype=np.float32)

    height, width, _ = image.shape

    random_counter = 0

    boxArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    rand_idx = random.randint(0, len(boxArea) - 1)
    rand_Side = boxArea[rand_idx] ** 0.5

    #anchors = [16, 32, 48, 64, 96, 128, 256, 512] can get what we want sometime, but unstable
    anchors = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 128, 256, 512]
    distance = infDistance
    anchor_idx = 5
    for i, anchor in enumerate(anchors):
        if abs(anchor - rand_Side) < distance:
            distance = abs(anchor - rand_Side)
            anchor_idx = i

    target_anchor = random.choice(anchors[0:min(anchor_idx + 1, 11)])#5)])
    ratio = float(target_anchor) / rand_Side
    ratio = ratio * (2 ** random.uniform(-1, 1))

    if int(height * ratio * width * ratio) > maxSize * maxSize:
        ratio = (maxSize * maxSize / (height * width)) ** 0.5

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = random.choice(interp_methods)
    image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)

    boxes[:, 0] *= ratio
    boxes[:, 1] *= ratio
    boxes[:, 2] *= ratio
    boxes[:, 3] *= ratio

    boxes = boxes.tolist()
    for i in range(len(anns)):
        anns[i]['bbox'] = [boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]]
        for j in range(5):
            anns[i]['keypoints'][j * 3] *= ratio  # [0, 3, 6, ...]
            anns[i]['keypoints'][j * 3 + 1] *= ratio  # [1, 4, 7, ...]

    return image, anns

def get_border(border, size):
    i = 1
    while size - border // i <= border // i:  # size > 2 * (border // i)
        i *= 2
    return border // i


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad) # (0, 1)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

# affine_transform(bbox[:2], trans_output)
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)

def coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
    return bbox

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def get_dataset():
    class Dataset(FACEHP, MultiPoseDataset):
        pass
    return Dataset

class FACEHP(data.Dataset):
  num_classes = 1
  num_joints = 5
  default_resolution = [800, 800]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  flip_idx = [[0, 1], [3, 4]]

  def __init__(self, opt, split):
    super(FACEHP, self).__init__()
    self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                  [4, 6], [3, 5], [5, 6],
                  [5, 7], [7, 9], [6, 8], [8, 10],
                  [6, 12], [5, 11], [11, 12],
                  [12, 14], [14, 16], [11, 13], [13, 15]]

    self.acc_idxs = [1, 2, 3, 4]
    self.data_dir = opt.data_dir
    self.img_dir = os.path.join(self.data_dir, 'images')

    _ann_name = {'train': 'train', 'val': 'val'}

    self.img_dir = opt.img_dir
    self.annot_path = opt.annot_path

    self.max_objs = 64
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    print('==> initializing centerface key point {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    image_ids = self.coco.getImgIds()

    if split == 'train':
      self.images = []
      for img_id in image_ids:
        idxs = self.coco.getAnnIds(imgIds=[img_id])

        if len(idxs) > 0:
          self.images.append(img_id)
    else:
      self.images = image_ids
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples)) # Loaded train 12671 samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def __len__(self):
    return self.num_samples

np_str_obj_array_pattern = re.compile(r'[SaUO]')
#cv2.setNumThreads(0) # need in some system to speed training

class MultiPoseDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:  # size > 2 * (border // i)
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = len(anns)
    if num_objs > self.max_objs:
        num_objs = self.max_objs
        anns = np.random.choice(anns, num_objs)

    img = cv2.imread(img_path)

    img, anns = Data_anchor_sample(img, anns)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.0, 0.05)) # for 512 * 512
        #s = s * np.random.choice(np.arange(0.8, 1.3, 0.05)) #for 768 * 768
        # s = s * np.random.choice(np.arange(0.3, 0.5, 0.1)) # for larger image, but speed down
        s = s
        _border = s * np.random.choice([0.1, 0.2, 0.25]) #
        w_border = self._get_border(_border, img.shape[1]) # w > 2 * w_border
        h_border = self._get_border(_border, img.shape[0]) # h > 2 * h_border
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      if np.random.random() < self.opt.aug_rot:
        rf = self.opt.rotate
        rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)

      if np.random.random() < self.opt.flip: # opt.flip = 0.5
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1

    trans_input = get_affine_transform(
      c, s, rot, [self.opt.input_res, self.opt.input_res])
    inp = cv2.warpAffine(img, trans_input,
                         (self.opt.input_res, self.opt.input_res),
                         flags=cv2.INTER_LINEAR)

    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_res = self.opt.output_res
    num_joints = self.num_joints
    trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
    trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

    hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
    hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
    dense_kps = np.zeros((num_joints, 2, output_res, output_res),
                          dtype=np.float32)
    dense_kps_mask = np.zeros((num_joints, output_res, output_res),
                               dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)

    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    wight_mask = np.ones((self.max_objs), dtype=np.float32)
    kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
    hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
    hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
    hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

    draw_gaussian = draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(ann['category_id']) - 1
      pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3) # (x,y,0/1)
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        pts[:, 0] = width - pts[:, 0] - 1
        for e in self.flip_idx:
          pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox = np.clip(bbox, 0, output_res - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0) or (rot != 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        #radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
        radius = max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        # wh[k] = 1. * w, 1. * h
        wh[k] = np.log(1. * w / 4), np.log(1. * h / 4)
        ind[k] = ct_int[1] * output_res + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        # if w*h <= 20: # can get what we want sometime, but unstable
        #     wight_mask[k] = 15
        if w*h <= 40:
            wight_mask[k] = 5
        if w*h <= 20:
            wight_mask[k] = 10
        if w*h <= 10:
            wight_mask[k] = 15
        if w*h <= 4:
            wight_mask[k] = 0.1

        num_kpts = pts[:, 2].sum()
        if num_kpts == 0:
          hm[cls_id, ct_int[1], ct_int[0]] = 0.9999

        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = max(0, int(hp_radius))
        for j in range(num_joints):
          if pts[j, 2] > 0:
            pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
            if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
               pts[j, 1] >= 0 and pts[j, 1] < output_res:
              kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
              kps[k, j * 2: j * 2 + 1] =  kps[k, j * 2: j * 2 + 1] / w
              kps[k, j * 2 + 1: j * 2 + 2] =  kps[k, j * 2 + 1: j * 2 + 2] / h
              kps_mask[k, j * 2: j * 2 + 2] = 1
              pt_int = pts[j, :2].astype(np.int32)
              hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
              hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
              hp_mask[k * num_joints + j] = 1
              if self.opt.dense_hp:
                # must be before draw center hm gaussian
                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                               pts[j, :2] - ct_int, radius, is_offset=True)
                draw_gaussian(dense_kps_mask[j], ct_int, radius)
              draw_gaussian(hm_hp[j], pt_int, hp_radius)
              if ann['bbox'][2]*ann['bbox'][3] <= 8.0:
                kps_mask[k, j * 2: j * 2 + 2] = 0
        draw_gaussian(hm[cls_id], ct_int, radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                       ct[0] + w / 2, ct[1] + h / 2, 1] +
                       pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
    if rot != 0:
      hm = hm * 0 + 0.9999
      reg_mask *= 0
      kps_mask *= 0
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
           'landmarks': kps, 'hps_mask': kps_mask, 'wight_mask': wight_mask}
    if self.opt.dense_hp:
      dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints, 1, output_res, output_res)
      dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints * 2, output_res, output_res)
      ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
      del ret['hps'], ret['hps_mask']
    if self.opt.reg_offset:
      ret.update({'hm_offset': reg})
    if self.opt.hm_hp:
      ret.update({'hm_hp': hm_hp})
    if self.opt.reg_hp_offset:
      ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
    return ret


_use_shared_memory = False

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return default_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))


def multipose_collate(batch):
  objects_dims = [d.shape[0] for d in batch]
  index = objects_dims.index(max(objects_dims))

  # one_dim = True if len(batch[0].shape) == 1 else False
  res = []
  for i in range(len(batch)):
      tres = np.zeros_like(batch[index], dtype=batch[index].dtype)
      tres[:batch[i].shape[0]] = batch[i]
      res.append(tres)

  return res


def Multiposebatch(batch):
  sample_batch = {}
  for key in batch[0]:
    if key in ['hm', 'input']:
      sample_batch[key] = default_collate([d[key] for d in batch])
    else:
      align_batch = multipose_collate([d[key] for d in batch])
      sample_batch[key] = default_collate(align_batch)

  return sample_batch


