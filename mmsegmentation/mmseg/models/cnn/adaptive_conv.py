# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.registry import MODELS
from torch import nn


@MODELS.register_module()
class Conv2d_tip(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode)

        # weight_shape = self.weight.shape
        self.register_parameter('disturbance', nn.Parameter(self.weight.clone()))
        self.indicator = nn.Parameter(torch.randn(1).cuda())
        self.weight.requires_grad = False
        self.weight.grad = None

    def forward(self, input):

        score = binarizer_fn(self.indicator)
        weight = (1 - score) * self.weight + score * self.disturbance
        return self._conv_forward(input, weight, None)


class BinarizerFn(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold=0.5):
        outputs = (inputs>=threshold).float()
        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None

binarizer_fn = BinarizerFn.apply