# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Efficient models for joint denoise and demosaicing."""
import torch
import torch.nn as nn
import logging
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory


class CRB(nn.Module):
    """Construct the residual block for JDDNET."""

    def __init__(self, InChannel, InterChannel, OutChannel, kSize=3):
        """Initialize Block.

        :param InChannel: the channel number of input
        :type InChannel: int
        :param InterChannel: the channel number of interlayer
        :type InterChannel: int
        :param OutChannel: the channel number of output
        :type OutChannel: int
        :param kSize: the kernel size of convolution
        :type kSize: int
        """
        super(CRB, self).__init__()
        self.InChan = InChannel
        self.InterCh = InterChannel
        self.OutChan = OutChannel
        self.ConvB = nn.ModuleList()
        if self.InChan != self.OutChan:
            self.trans = nn.Sequential(
                *[nn.Conv2d(self.InChan, self.OutChan, kSize, padding=(kSize - 1) // 2, stride=1), nn.ReLU()])
        self.ConvB = nn.ModuleList()
        self.ConvB.append(nn.Sequential(
            *[nn.Conv2d(self.OutChan, self.InterCh, kSize, padding=(kSize - 1) // 2, stride=1), nn.ReLU()]))
        self.ConvB.append(nn.Sequential(
            *[nn.Conv2d(self.InterCh, self.OutChan, kSize, padding=(kSize - 1) // 2, stride=1), nn.ReLU()]))

    def forward(self, x):
        """Forward function.

        :return: output of block
        :rtype: tensor
        """
        if self.InChan != self.OutChan:
            x = self.trans(x)
        x_inter = self.ConvB[0](x)
        x_inter = self.ConvB[1](x_inter)
        x_return = x + x_inter
        return x_return


class SpaceToDepth(nn.Module):
    """Construct the space-to-depth opreation for JDDNET."""

    def __init__(self, block_size):
        """Initialize Block.

        :param block_size: interval
        :type block_size: int
        """
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        """Forward function.

        :return: output of block
        :rtype: tensor
        """
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)
        return x


@NetworkFactory.register(NetTypes.JDDBODY)
class JDDNet(Network):
    """Construct the JDD networks."""

    def __init__(self, net_desc):
        """Initialize the JDD network class.

        :param net_desc: config of the searched structure
        :type net_desc: dictionary
        """
        super(JDDNet, self).__init__()
        logging.info("start init JDDNet")
        self.desc = net_desc
        self.arch = net_desc.architecture
        self.D = len(self.arch)
        Channel_num = net_desc.basic_channel
        kSize = 3
        n_colors = 5

        self.Conc_loc = []
        self.Bone_blocks = nn.ModuleList()
        self.Bone_blocks.append(nn.Sequential(*[nn.Conv2d(n_colors, Channel_num, kSize,
                                                          padding=(kSize - 1) // 2, stride=1), nn.ReLU()]))
        self.Conc_loc.append(0)

        out_ch = Channel_num
        conc_state = 0
        for i in range(self.D):
            name = self.arch[i]
            key = name.split('_')
            b_type = key[0]
            if b_type == 'Space-to-Depth':
                self.Bone_blocks.append(SpaceToDepth(2))
                self.Conc_loc[-1] = 1
                self.Conc_loc.append(0)
            elif b_type == 'Depth-to-Space':
                self.Bone_blocks.append(nn.Sequential(*[nn.Conv2d(out_ch, out_ch, kSize, padding=(kSize - 1) // 2,
                                                                  stride=1), nn.ReLU()]))
                self.Conc_loc.append(0)
                self.Bone_blocks.append(nn.PixelShuffle(2))
                self.Conc_loc.append(0)
                conc_state = 1
            elif b_type == 'R':
                in_ch = int(key[1])
                inter_ch = int(key[2])
                out_ch = int(key[3])
                self.Bone_blocks.append(CRB(InChannel=in_ch, InterChannel=inter_ch, OutChannel=out_ch))
                if conc_state == 0:
                    self.Conc_loc.append(0)
                elif conc_state == 1:
                    self.Conc_loc.append(2)
                    conc_state = 0

        self.pre_ch = out_ch
        self.TransConv = nn.Sequential(*[nn.Conv2d(out_ch * 2, Channel_num * 2, kSize,
                                                   padding=(kSize - 1) // 2, stride=1), nn.ReLU()])
        self.Up = nn.PixelShuffle(2)
        self.RecConv = nn.Conv2d(Channel_num // 2, 3, kSize, padding=(kSize - 1) // 2, stride=1)

    def forward(self, x):
        """Calculate the output of the model.

        :param x: input images
        :type x: tensor
        :return: output tensor of the model
        :rtype: tensor
        """
        raw_frames = torch.split(x, 1, 4)
        demosaic_frames = []
        feat_aux = torch.zeros([x.shape[0], self.pre_ch, x.shape[2], x.shape[3]]).cuda()
        for frame in raw_frames:
            frame = torch.squeeze(frame, 4)
            branch_feature = []
            for ind in range(len(self.Conc_loc)):
                if self.Conc_loc[ind] == 0:
                    frame = self.Bone_blocks[ind](frame)
                elif self.Conc_loc[ind] == 1:
                    frame = self.Bone_blocks[ind](frame)
                    branch_feature.append(frame)
                elif self.Conc_loc[ind] == 2:
                    frame = self.Bone_blocks[ind](torch.cat((frame, branch_feature.pop()), 1))

            f_trans = self.TransConv(torch.cat((frame, feat_aux), 1))
            demosaic = self.RecConv(self.Up(f_trans))
            demosaic_frames.append(demosaic)
            feat_aux = frame * 1.0
        demosaic_all = torch.stack(demosaic_frames, 4)
        return demosaic_all
