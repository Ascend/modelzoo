# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .nms.cpu_nms import cpu_nms, cpu_soft_nms


# Original NMS implementation
def nms(dets, thresh):
    """CPU NMS implementations."""
    if dets.shape[0] == 0:
        return []

    return cpu_nms(dets, thresh)
