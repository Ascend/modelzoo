"""Profile the model info, parameters, flops etc."""

import logging

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmdet.models.utils import ConvWS2d
from torch.nn.modules.conv import _ConvNd

from .count_hooks import (count_adap_avgpool, count_adap_maxpool,
                          count_avgpool, count_bn, count_convNd,
                          count_convtranspose2d, count_gn, count_linear,
                          count_maxpool, count_relu)

register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose2d: count_convtranspose2d,
    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,
    nn.ReLU: count_relu,
    nn.ReLU6: count_relu,
    nn.LeakyReLU: count_relu,
    nn.MaxPool1d: count_maxpool,
    nn.MaxPool2d: count_maxpool,
    nn.MaxPool3d: count_maxpool,
    nn.AdaptiveMaxPool1d: count_adap_maxpool,
    nn.AdaptiveMaxPool2d: count_adap_maxpool,
    nn.AdaptiveMaxPool3d: count_adap_maxpool,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: None,
}
custom_ops = {ConvWS2d: count_convNd,
              nn.GroupNorm: count_gn}


def clever_format(num, format="%.2f"):
    """Format num into scientific notation."""
    if num > 1e12:
        return format % (num / 1e12) + "T"
    if num > 1e9:
        return format % (num / 1e9) + "G"
    if num > 1e6:
        return format % (num / 1e6) + "M"
    if num > 1e3:
        return format % (num / 1e3) + "K"
    else:
        return format % (num)


class counter(int):
    """Handle int num."""

    def __new__(cls, value, *args, **kwargs):
        """Instantiate."""
        return super().__new__(cls, value, *args, **kwargs)

    def __add__(self, other):
        """Add."""
        res = super().__add__(other)
        return self.__class__(res)

    def __sub__(self, other):
        """Sub."""
        res = super().__sub__(other)
        return self.__class__(res)

    def __format__(self, format_spec):
        """Format."""
        return format(str(self), format_spec)

    def __repr__(self):
        """Repr."""
        return self.__str__()

    def __str__(self):
        """Str."""
        format = "%.2f"
        if self > 1e12:
            return format % (self / 1e12) + "T"
        if self > 1e9:
            return format % (self / 1e9) + "G"
        if self > 1e6:
            return format % (self / 1e6) + "M"
        if self > 1e3:
            return format % (self / 1e3) + "K"
        else:
            return format % (self)


def gen_input(input_size, size_divisor, style, **kwargs):
    """Generate input tensor."""
    input = np.zeros((input_size[0], input_size[1], 3))
    if size_divisor:
        input = mmcv.impad_to_multiple(input, size_divisor)
    input = torch.from_numpy(
        input.transpose(
            2, 0, 1)).unsqueeze(0).type(
        torch.FloatTensor).cuda()
    input.requires_grad = False
    if style == 'mmdet':
        img_meta = [[
            dict(ori_shape=(input_size[1], input_size[0], 3),
                 img_shape=(input_size[1], input_size[0], 3),
                 pad_shape=(input_size[1], input_size[0], 3),
                 scale_factor=1, flip=False)]]
        kwargs.update(dict(img_meta=img_meta))
        kwargs.update(dict(return_loss=False))
        input = [input]
    return input


def get_model_info(model):
    """Get model info for mmdet model."""
    all_ops = 0
    all_mac = 0
    all_params = 0
    check = []
    results = dict()
    for name, part in model.named_children():
        total_ops = 0
        total_params = 0
        total_mac = 0
        for m in part.modules():
            if len(list(m.children())) > 0:  # skip for non-leaf module
                continue
            total_ops += m.total_ops
            total_params += m.total_params
            total_mac += m.total_mac
            # for check information
            if name == 'bbox_head':
                if isinstance(m, nn.Conv2d):
                    op_name = 'CONV{}x{}'.format(
                        m.kernel_size[0], m.kernel_size[1])
                    check.append(
                        [op_name, m.total_ops.item(), m.total_params.item()])
                elif isinstance(m, nn.Linear):
                    check.append(
                        ['FC', m.total_ops.item(), m.total_params.item()])
                elif isinstance(m, nn.AdaptiveAvgPool2d):
                    check.append(
                        ['POOL', m.total_ops.item(), m.total_params.item()])
        total_ops = total_ops.item()
        total_params = total_params.item()
        total_mac = total_mac.item()
        all_ops += total_ops
        all_params += total_params
        all_mac += total_mac
        results[name] = dict(
            FLOPs=counter(total_ops),
            params=counter(total_params),
            MAC=counter(total_mac))
    results['total'] = dict(
        FLOPs=counter(all_ops),
        params=counter(all_params),
        MAC=counter(all_mac))

    return results


def profile(model,
            input_size,
            size_divisor=32,
            device="cpu",
            style='mmdet',
            show_result=True,
            *args,
            **kwargs):
    """Compute params and flops."""
    handler_collection = []
    assert style in ['mmdet', 'normal']

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))
        m.register_buffer('total_mac', torch.zeros(1))
        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])
        m_type = type(m)
        fn = None
        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            logging.info("Not implemented for ", m)
        if fn is not None:
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.apply(add_hooks)
    input = gen_input(input_size, size_divisor, style, **kwargs)
    with torch.no_grad():
        model(input, *args, **kwargs)
    if style == 'mmdet':
        results = get_model_info(model)
    else:
        all_ops = 0
        all_mac = 0
        all_params = 0
        for m in model.modules():
            if len(list(m.children())) > 0:  # skip for non-leaf module
                continue
            all_ops += m.total_ops
            all_params += m.total_params
            all_mac += m.total_mac
        all_ops = all_ops.item()
        all_params = all_params.item()
        all_mac = all_mac.item()
        results = dict(
            FLOPs=counter(all_ops),
            params=counter(all_params),
            MAC=counter(all_mac))
    if show_result:
        print_results(results, style=style)
    for handler in handler_collection:
        handler.remove()
    del input, model
    torch.cuda.empty_cache()
    return results


def print_results(results, style='mmdet'):
    """Show result."""
    info = []
    if style == 'mmdet':
        template = '{:<20}|{FLOPs:<10}|{params:<10}|{MAC:<10}'
        header = template.format(
            'Parts',
            FLOPs='FLOPs',
            params='Params',
            MAC='MAC')
        divider = len(template.format('', FLOPs='', params='', MAC='')) * '-'
        info.append(divider)
        info.append(header)
        info.append(divider)
        for key, value in results.items():
            info.append(template.format(key, **value))
        info.append(divider)
        logging.info('\n'.join(info))
    else:
        template = '{FLOPs:<10}|{params:<10}|{MAC:<10}'
        divider = len(template.format(FLOPs='', params='', MAC='')) * '-'
        header = template.format(FLOPs='FLOPs', params='Params', MAC='MAC')
        logging.info(header)
        logging.info(divider)
        logging.info(template.format(**results))
