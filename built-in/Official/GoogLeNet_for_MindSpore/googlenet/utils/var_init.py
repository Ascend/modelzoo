import math
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import initializer as init

def _calculate_gain(nonlinearity, param=None):
    r"""
    Return the recommended gain value for the given nonlinearity function.

    The values are as follows:
    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _select_fan(array, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = init._calculate_in_and_out(array)
    return fan_in if mode == 'fan_in' else fan_out

class KaimingInit(init.Initializer):
    r"""
    Base Class. Initialize the array with He kaiming algorithm.

    Args:
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function, recommended to use only with
            ``'relu'`` or ``'leaky_relu'`` (default).
    """
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingInit, self).__init__()
        self.mode = mode
        self.gain = _calculate_gain(nonlinearity, a)


class KaimingUniform(KaimingInit):
    r"""
    Initialize the array with He kaiming uniform algorithm. The resulting tensor will
    have values sampled from :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Input:
        arr (Array): The array to be assigned.

    Returns:
        Array, assigned array.

    Examples:
        >>> w = np.empty(3, 5)
        >>> KaimingUniform(w, mode='fan_in', nonlinearity='relu')
    """
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingUniform, self).__init__(a, mode, nonlinearity)

    def _initialize(self, arr):
        fan = _select_fan(arr, self.mode)
        bound = math.sqrt(3.0) * self.gain / math.sqrt(fan)
        data = np.random.uniform(-bound, bound, arr.shape)

        init._assignment(arr, data)


class KaimingNormal(KaimingInit):
    r"""
    Initialize the array with He kaiming normal algorithm. The resulting tensor will
    have values sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Input:
        arr (Array): The array to be assigned.

    Returns:
        Array, assigned array.

    Examples:
        >>> w = np.empty(3, 5)
        >>> KaimingNormal(w, mode='fan_out', nonlinearity='relu')
    """
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingNormal, self).__init__(a, mode, nonlinearity)

    def _initialize(self, arr):
        fan = _select_fan(arr, self.mode)
        std = self.gain / math.sqrt(fan)
        data = np.random.normal(0, std, arr.shape)

        init._assignment(arr, data)


def default_recurisive_init(custom_cell):
    for _, cell in custom_cell.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.default_input = init.initializer(KaimingUniform(a=math.sqrt(5)), cell.weight.default_input.shape(), cell.weight.default_input.dtype()).to_tensor()
            if cell.bias is not None:
                fan_in, _ = init._calculate_in_and_out(cell.weight.default_input.asnumpy())
                bound = 1 / math.sqrt(fan_in)
                cell.bias.default_input = Tensor(np.random.uniform(-bound, bound, cell.bias.default_input.shape()), cell.bias.default_input.dtype())
        elif isinstance(cell, nn.Dense):
            cell.weight.default_input = init.initializer(KaimingUniform(a=math.sqrt(5)), cell.weight.default_input.shape(), cell.weight.default_input.dtype()).to_tensor()
            if cell.bias is not None:
                fan_in, _ = init._calculate_in_and_out(cell.weight.default_input.asnumpy())
                bound = 1 / math.sqrt(fan_in)
                cell.bias.default_input = Tensor(np.random.uniform(-bound, bound, cell.bias.default_input.shape()), cell.bias.default_input.dtype())
        elif isinstance(cell, nn.BatchNorm2d) or isinstance(cell, nn.BatchNorm1d):
            pass
