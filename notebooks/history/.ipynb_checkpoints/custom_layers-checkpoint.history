#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np

class BReLU(nn.Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{BReLU}(x) = \min(\max(0,x), cutoff)

    Args:
        cutoff: can optionally set the maximum accepted output
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    modified from ReLU6

    Examples::

        >>> m = nn.BReLU(cutoff, inpalce)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, cutoff=24., inplace=False):
        super(BReLU, self).__init__(0., cutoff, inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str



class Mask(nn.Module):
    '''
    Args:
        x: input 4d tensor [batch, channel, height, width]. only support batch first
        length: true height of each data. It's length should be the same as batch size
    '''
    def __init__(self):
        super(Mask, self).__init__()

    def forward(self,x, length):
        assert x.shape[0] == len(length)

        mask = torch.zeros_like(x, dtype=torch.float32)
        for index, length in enumerate(length):
            mask[index, :, :length, :] = 1
        return x * mask



class Conv_bn_mask(nn.Module):
    def __init__(self, ichannel, ochannel, kernel_size, padding, stride, bias=False, track_running_stats=False):
        super(Conv_bn_mask, self).__init__()
        self.conv = nn.Conv2d(ichannel, ochannel, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(ochannel, track_running_stats=track_running_stats)
        self.activation = BReLU(cutoff=24)
        self.mask = Mask()
    def forward(self, x, length):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.mask(x, length)
        return x
