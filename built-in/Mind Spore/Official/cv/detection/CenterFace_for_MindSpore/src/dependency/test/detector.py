###modified based on centernet###
#MIT License
#Copyright (c) 2019 Xingyi Zhou
#All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import time
import torch
import torch.nn as nn

from mindspore import Tensor

from src.dependency.test.nms import soft_nms
try:
  from src.dataset import get_affine_transform, affine_transform
except:
  from src.dependency.train.dataset import get_affine_transform, affine_transform

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind) # 关键点这边跟ct的index,因为有ct,才有对应的lankmark的预测
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()  # 这是 ‘[b,c,h,w]-->[b,h,w,c]’
    feat = feat.view(feat.size(0), -1, feat.size(3))  # 把中间的w*h拉起来了,我们可以自己对应
    feat = _gather_feat(feat, ind)
    return feat

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def multi_pose_post_process(dets, c, s, h, w):
  # dets：box: 4 + score:1 + kpoints: 10 +  class: 1 = 16
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:15].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5],
       pts.reshape(-1, 10)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def _max_pooling_forward(z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    :param z: size (N,C,H,W)
    :param pooling: (k1,k2)
    :param strides
    :param padding
    :return:
    """
    N, C, H, W = z.shape
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)

    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.max(padding_z[n, c,
                                                          strides[0] * i:strides[0] * i + pooling[0],
                                                          strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    #hmax = _max_pooling_forward(
    #    heat, (kernel, kernel), strides=(1, 1), padding=pad)
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()

      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)             # 前100个点

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


class CenterFaceDetector(object):
  def __init__(self, opt, model):

    self.flip_idx = opt.flip_idx

    print('Creating model...')
    self.model = model

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = False
    #self.device = torch.device('cpu')

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res: # True
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      # inp_height = (new_height | self.opt.pad) + 1
      # inp_width = (new_width | self.opt.pad) + 1
      inp_height = int(np.ceil(new_height / 32) * 32)
      inp_width = int(np.ceil(new_width / 32) * 32)
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)

    meta = {'c': c, 's': s,
            'out_height': inp_height // self.opt.down_ratio,
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def process(self, images, return_time=False):
    images = Tensor(images)
    # test with mindspore model
    output_hm, output_wh, output_off, output_kps = self.model(images)
    # Tensor to numpy
    output_hm = output_hm.asnumpy()
    output_wh = output_wh.asnumpy()
    output_off = output_off.asnumpy()
    output_kps = output_kps.asnumpy()

    output_hm = output_hm.astype(np.float32)
    output_wh = output_wh.astype(np.float32)
    output_off = output_off.astype(np.float32)
    output_kps = output_kps.astype(np.float32)
    #numpy to torch
    output_hm = torch.from_numpy(output_hm)
    output_wh = torch.from_numpy(output_wh)
    output_off = torch.from_numpy(output_off)
    output_kps = torch.from_numpy(output_kps)

    reg = output_off if self.opt.reg_offset else None

    dets = self.centerface_decode(
      output_hm, output_wh, output_kps,
      reg=reg, K=self.opt.K)

    return dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 15)             # 关键点数+5=15
      # import pdb; pdb.set_trace()
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    if self.opt.nms or len(self.opt.test_scales) > 1:
      soft_nms(results[1], Nt=0.5, method=2)
    results[1] = results[1].tolist()
    return results

  def run(self, image_or_path_or_tensor, meta=None): # image_path
    #torch.set_num_threads(192)
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''):  # 走这个分支, pre_processed = False
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True

    detections = []
    for scale in self.scales: # [1]
      scale_start_time = time.time()
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta) # --1: pre_process
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}

      dets = self.process(images) # --2: process

      dets = self.post_process(dets, meta, scale)   # box:4+score:1+kpoints:10+class:1=16   ## --3: post_process

      detections.append(dets)

    results = self.merge_outputs(detections) # --4: merge_outputs
    return {'results': results}

  def centerface_decode(self, heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys_int, xs_int = _topk(heat, K=K)
    # scores, inds, clses, ys_int, xs_int, K = threshold_choose(heat, threshold=0.05)

    if reg is not None:  # 回归的中心点偏移量
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs_int.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys_int.view(batch, K, 1) + reg[:, :, 1:2]
        # xs = (xs_int.view(batch, K, 1) + reg[:, :, 0:1] + 0.5)
        # ys = (ys_int.view(batch, K, 1) + reg[:, :, 1:2] + 0.5)
    else:
        xs = xs_int.view(batch, K, 1) + 0.5
        ys = ys_int.view(batch, K, 1) + 0.5

    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    wh = wh.exp() * 4.
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)

    kps = _tranpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)

    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if hp_offset is not None:
            hp_offset = _tranpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > thresh).float()  #choose hm_score > 0.1
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)

        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)  # box:4+score:1+kpoints:10+class:1=16

    return detections
