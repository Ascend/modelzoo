# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Anchor Generator."""
import torch


class AnchorGenerator(object):
    """Anchor generator."""

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        """Init anchor generator.

        :param base_size: base size
        :param scales: anchor scales
        :param ratios: anchor ratios
        :param scale_major: if scale major
        :param ctr: center
        """
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """Num of base anchors.

        :return: size of base anchors
        """
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        """Generate base anchors.

        :return: base anchors
        """
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr
        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)
        base_anchors = torch.stack([x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                                    x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)],
                                   dim=-1).round()
        return base_anchors

    def meshgrid(self, x, y, row_major=True):
        """Mesh grid.

        :param x: x
        :param y: y
        :param row_major: raw major
        :return: xx and yy
        """
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        """Grid anchor.

        :param featmap_size: size of feature map
        :param stride: stride of feature map
        :param device: device
        :return: anchors
        """
        base_anchors = self.base_anchors.to(device)
        feat_h, feat_w = featmap_size
        x_shift = torch.arange(0, feat_w, device=device) * stride
        y_shift = torch.arange(0, feat_h, device=device) * stride
        xx_shift, yy_shift = self.meshgrid(x_shift, y_shift)
        shifts = torch.stack([xx_shift, yy_shift, xx_shift, yy_shift], dim=-1)
        shifts = shifts.type_as(base_anchors)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        """Flag of valid anchors.

        :param featmap_size: size of feature map
        :param valid_size: size of valid anchor
        :param device: device
        :return: valid flags
        """
        h_feat, w_feat = featmap_size
        h_valid, w_valid = valid_size
        assert h_valid <= h_feat and w_valid <= w_feat
        x_valid = torch.zeros(w_feat, dtype=torch.uint8, device=device)
        y_valid = torch.zeros(h_feat, dtype=torch.uint8, device=device)
        x_valid[:w_valid] = 1
        y_valid[:h_valid] = 1
        xx_valid, yy_valid = self.meshgrid(x_valid, y_valid)
        valid = xx_valid & yy_valid
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid
