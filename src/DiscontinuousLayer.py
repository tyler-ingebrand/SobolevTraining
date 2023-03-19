import math
from typing import Any

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.modules.lazy import LazyModuleMixin


class DifferentiableHeaviside(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        outs = torch.zeros_like(input)
        outs[input == 0.0] = float('nan')
        outs[input > 0.0] = 1.0
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return torch.zeros_like(input)
heaviside = DifferentiableHeaviside.apply

class DiscontinuousLayer(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features']
    in_features: int

    def __init__(self, in_features: int, activation_function=torch.nn.Sigmoid(),
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DiscontinuousLayer, self).__init__()
        self.in_features = in_features
        self.epsilon = Parameter(torch.empty(in_features, **factory_kwargs))
        self.activation_function = activation_function
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        init.uniform_(self.epsilon, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return self.activation_function(input) + self.epsilon * heaviside(input)

    def extra_repr(self) -> str:
        return 'in_features={}'.format(self.in_features)