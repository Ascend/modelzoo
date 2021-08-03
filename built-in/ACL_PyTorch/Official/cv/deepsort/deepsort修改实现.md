# deepsort源码适配Ascend NPU修改

## darknet.py文件修改

将darknet.py中Darknet类的forward函数返回值修改为。

~~~python
return out_boxes[0]['x'], out_boxes[0]['a'], out_boxes[1]['x'], out_boxes[1]['a'], out_boxes[2]['x'], out_boxes[2]['a']
~~~

## detector.py文件修改：

```
from .nms import boxes_nms
from .acl_net_dynamic import AclModel
```

YOLOv3类__init__函数中添加：

~~~python
self.dims = {'dimCount':4, 'name':'', 'dims':[1,3,416,416]}
self.model = AclModel(device_id = 0, model_path = "yolov3-sim.om")
self.anchors = [torch.tensor([3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875]),
                torch.tensor([1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375]),
                torch.tensor([1.25, 1.625, 2.0, 3.75, 4.125, 2.875])]
~~~

__call__函数中with.no_grad()修改：

```
        with torch.no_grad():
            #img = img.to(self.device)
            #out_boxes = self.net(img)
            '''
            # export onnx code
            input_names = ["actual_input_1"]
            dummpy_input = img
            import onnx
            torch.onnx.export(self.net, dummpy_input, "yolov3.onnx", input_names = input_names, opset_version = 11)
            return 
            '''

            img = np.array(img, np.float32)
            img = np.ascontiguousarray(img, dtype = np.float32)
            out_boxes = self.model(img, self.dims)
            boxes = get_all_boxes(out_boxes, self.anchors, self.conf_thresh, self.num_classes,
                                  use_cuda=self.use_cuda)  # batch size is 1
```

## feature_extractor.py文件修改：

```
from .acl_net_dynamic import AclModel
```

类Extractor中__init__函数添加：

```
self.model = AclModel(device_id = 0, model_path = "deep_dims.om")
```

__call__函数修改为：

~~~python
def __call__(self, im_crops):
    im_batch = self._preprocess(im_crops)
    with torch.no_grad():
        im_batch = im_batch.to(self.device)
        dynamic_dim = im_batch.shape[0]
        dims = {'dimCount':4, 'name': '', 'dims': [dynamic_dim, 3, 128, 64]}
        im_batch = im_batch.cpu().numpy()
        #features = self.model([im_batch], dims)
        features = self.net(im_batch)
        return features[0]
~~~

## yolo_utils.py文件修改

get_all_boxes方法修改为。

~~~python
def get_all_boxes(output, output_anchors, conf_thresh, num_classes, only_objectness=1, validation=False, use_cuda=True):
    # total number of inputs (batch size)
    # first element (x) for first tuple (x, anchor_mask, num_anchor)
    batchsize = output[0]['x'].data.size(0)

    all_boxes = []
    for i in range(len(output)):
        #pred, anchors, num_anchors = output[i]['x'].data, output[i]['a'], output[i]['n'].item()
        pred, anchors, num_anchors = torch.from_numpy(output[i]), output_anchors[i], 3
        boxes = get_region_boxes(pred, conf_thresh, num_classes, anchors, num_anchors, \
                                 only_objectness=only_objectness, validation=validation, use_cuda=use_cuda)

        all_boxes.append(boxes)
    return torch.cat(all_boxes, dim=1)
~~~

导出onnx时注释掉原方法，调用BNMS方法等价替换。

~~~python
def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
   ...
   
    # npu return
        if torch.onnx.is_in_onnx_export():
            dets, labels = batch_nms_op(proposals, 
            							scores, 0.0, 
            							nms_cfg.get("iou_threshold"), 
            							cfg.nms_post, 
            							cfg.nms_post)
            return dets
        # cpu and gpu return
        else:
            dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
            return dets[:cfg.nms_post]
~~~

## single_level_roi_extractor.py文件修改

注册NPU RoiExtractor算子。

在源文件中添加如下代码，这里的forward只是为了推导shape写的伪实现，只需要成功导出到onnx即可。

~~~python
class RoiExtractor(torch.autograd.Function):
    @staticmethod
    def forward(self, f0, f1, f2, f3, rois, aligned=1, finest_scale=56, pooled_height=7, pooled_width=7,
                         pool_mode='avg', roi_scale_factor=0, sample_num=0, spatial_scale=[0.25, 0.125, 0.0625, 0.03125]):
        """
        feats (torch.Tensor): feats in shape (batch, 256, H, W).
        rois (torch.Tensor): rois in shape (k, 5).
        return:
            roi_feats (torch.Tensor): (k, 256, pooled_width, pooled_width)
        """

        # phony implementation for shape inference
        k = rois.size()[0]
        roi_feats = torch.ones(k, 256, pooled_height, pooled_width)
        return roi_feats

    @staticmethod
    def symbolic(g, f0, f1, f2, f3, rois):
        # TODO: support tensor list type for feats
        roi_feats = g.op('RoiExtractor', f0, f1, f2, f3, rois, aligned_i=1, finest_scale_i=56, pooled_height_i=7, pooled_width_i=7,
                         pool_mode_s='avg', roi_scale_factor_i=0, sample_num_i=0, spatial_scale_f=[0.25, 0.125, 0.0625, 0.03125], outputs=1)
        return roi_feats
~~~

在原forward方法中插入分支，导出onnx时使用RoiExtractor算子等价替换。

~~~python
@force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        # Work around to export onnx for npu
        if torch.onnx.is_in_onnx_export():
            roi_feats = RoiExtractor.apply(feats[0], feats[1], feats[2], feats[3], rois)
            # roi_feats = RoiExtractor.apply(list(feats), rois)
            return roi_feats

        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(
                -1, self.out_channels * out_size[0] * out_size[1])
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)
	...  
~~~

## delta_xywh_bbox_coder.py文件修改

添加onnx export分支，利用numpy()将means和std的shape固定下来。

修改坐标的轴顺序，使切片操作在NPU上效率更高，整网性能提升约7%。

~~~python
# fix shape for means and stds when exporting onnx
def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True):
    if torch.onnx.is_in_onnx_export():
        means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1).numpy() // 4)
        stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1).numpy() // 4)
    else:
        means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 4)
        stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 4)
        
...
        
    # improve gather performance on NPU
    if torch.onnx.is_in_onnx_export():
        rois_perf = rois.permute(1, 0)
        # Compute center of each roi
        px = ((rois_perf[0, :] + rois_perf[2, :]) * 0.5).unsqueeze(1).expand_as(dx)
        py = ((rois_perf[1, :] + rois_perf[3, :]) * 0.5).unsqueeze(1).expand_as(dy)
        # Compute width/height of each roi
        pw = (rois_perf[2, :] - rois_perf[0, :]).unsqueeze(1).expand_as(dw)
        ph = (rois_perf[3, :] - rois_perf[1, :]).unsqueeze(1).expand_as(dh)
    else:
        # Compute center of each roi
        px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
        py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
        # Compute width/height of each roi
        pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
        ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
        
    ...
~~~

## deform_conv.py文件修改

添加onnx export分支，利用numpy()将means和std的shape固定下来。

修改坐标的轴顺序，使切片操作在NPU上效率更高，整网性能提升约7%。

~~~python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

from mmcv.utils import deprecated_api_warning
from ..cnn import CONV_LAYERS
from ..utils import ext_loader, print_log

ext_module = ext_loader.load_ext('_ext', [
    'deform_conv_forward', 'deform_conv_backward_input',
    'deform_conv_backward_parameters'
])


class DeformConv2dFunction(Function):

    @staticmethod
    def symbolic(g,
                 input,
                 weight,
                 offset,
                 stride,
                 padding,
                 dilation,
                 groups,
                 deform_groups,
                 bias=False,
                 im2col_step=32):
        return g.op(
            'DeformableConv2D',
            input,
            weight,
            offset,
            strides_i=stride,
            pads_i=padding,
            dilations_i=dilation,
            groups_i=groups,
            deformable_groups_i=deform_groups,
            bias_i=bias,
            data_format_s="NCHW",
            im2col_step_i=im2col_step)

    @staticmethod
    def forward(ctx,
                input,
                weight,
                offset,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deform_groups=1,
                bias=False,
                im2col_step=32):
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        assert bias is False, 'Only support bias is False.'
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, offset, weight)

        output = input.new_empty(
            DeformConv2dFunction._output_size(ctx, input, weight))

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        cur_im2col_step = min(ctx.im2col_step, input.size(0))
        assert (input.size(0) %
                cur_im2col_step) == 0, 'im2col step must divide batchsize'
        if torch.onnx.is_in_onnx_export():
            return torch.rand(output.shape)
        ext_module.deform_conv_forward(
            input,
            weight,
            offset,
            output,
            ctx.bufs_[0],
            ctx.bufs_[1],
            kW=weight.size(3),
            kH=weight.size(2),
            dW=ctx.stride[1],
            dH=ctx.stride[0],
            padW=ctx.padding[1],
            padH=ctx.padding[0],
            dilationW=ctx.dilation[1],
            dilationH=ctx.dilation[0],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            im2col_step=cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = None

        cur_im2col_step = min(ctx.im2col_step, input.size(0))
        assert (input.size(0) %
                cur_im2col_step) == 0, 'im2col step must divide batchsize'

        grad_output = grad_output.contiguous()
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_input = torch.zeros_like(input)
            grad_offset = torch.zeros_like(offset)
            ext_module.deform_conv_backward_input(
                input,
                offset,
                grad_output,
                grad_input,
                grad_offset,
                weight,
                ctx.bufs_[0],
                kW=weight.size(3),
                kH=weight.size(2),
                dW=ctx.stride[1],
                dH=ctx.stride[0],
                padW=ctx.padding[1],
                padH=ctx.padding[0],
                dilationW=ctx.dilation[1],
                dilationH=ctx.dilation[0],
                group=ctx.groups,
                deformable_group=ctx.deform_groups,
                im2col_step=cur_im2col_step)

        if ctx.needs_input_grad[2]:
            grad_weight = torch.zeros_like(weight)
            ext_module.deform_conv_backward_parameters(
                input,
                offset,
                grad_output,
                grad_weight,
                ctx.bufs_[0],
                ctx.bufs_[1],
                kW=weight.size(3),
                kH=weight.size(2),
                dW=ctx.stride[1],
                dH=ctx.stride[0],
                padW=ctx.padding[1],
                padH=ctx.padding[0],
                dilationW=ctx.dilation[1],
                dilationH=ctx.dilation[0],
                group=ctx.groups,
                deformable_group=ctx.deform_groups,
                scale=1,
                im2col_step=cur_im2col_step)

        return grad_input, grad_offset, grad_weight, \
            None, None, None, None, None, None, None

    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be ' +
                'x'.join(map(str, output_size)) + ')')
        return output_size


deform_conv2d = DeformConv2dFunction.apply


class DeformConv2d(nn.Module):

    @deprecated_api_warning({'deformable_groups': 'deform_groups'},
                            cls_name='DeformConv2d')
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=False):
        super(DeformConv2d, self).__init__()

        assert not bias, \
            f'bias={bias} is not supported in DeformConv2d.'
        assert in_channels % groups == 0, \
            f'in_channels {in_channels} cannot be divisible by groups {groups}'
        assert out_channels % groups == 0, \
            f'out_channels {out_channels} cannot be divisible by groups \
              {groups}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        # only weight, no bias
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        # To fix an assert error in deform_conv_cuda.cpp:128
        # input image is smaller than kernel
        input_pad = (x.size(2) < self.kernel_size[0]) or (x.size(3) <
                                                          self.kernel_size[1])
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0)
            offset = offset.contiguous()
        out = deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                            self.dilation, self.groups, self.deform_groups)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()
        return out


@CONV_LAYERS.register_module('DCN')
class DeformConv2dPack(DeformConv2d):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    The offset tensor is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
    The spatial arrangement is like:

    .. code:: text

        (x0, y0) (x1, y1) (x2, y2)
        (x3, y3) (x4, y4) (x5, y5)
        (x6, y6) (x7, y7) (x8, y8)

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(DeformConv2dPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        if torch.onnx.is_in_onnx_export():
            offset_y = offset.reshape(1, -1, 2, offset.shape[2].numpy(), offset.shape[3].numpy())[:, :, 0, ...].reshape(
                1, offset.shape[1].numpy() // 2, offset.shape[2].numpy(), offset.shape[3].numpy())
            offset_x = offset.reshape(1, -1, 2, offset.shape[2].numpy(), offset.shape[3].numpy())[:, :, 1, ...].reshape(
                1, offset.shape[1].numpy() // 2, offset.shape[2].numpy(), offset.shape[3].numpy())
            mask = torch.ones(offset.shape[0].numpy(), offset.shape[1].numpy() // 2, offset.shape[2].numpy(), offset.shape[3].numpy())
            offset = torch.cat((offset_x, offset_y, mask), 1)
        return deform_conv2d(x, self.weight, offset, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, DeformConvPack loads previous benchmark models.
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                f'DeformConv2dPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

~~~



# 去掉pytorch2onnx的算子检查

由于NPU自定义算子在 onnx 中未定义，需要去掉pytorch2onnx的检查，否则会报错。

1. 通过pip show torch找到pytorch安装位置，比如/home/mmdet/lib/python3.7/site-packages。
2. 打开文件/home/mmdet/lib/python3.7/site-packages/torch/onnx/utils.py。
3. 搜索_check_onnx_proto(proto)，并注释该行。



