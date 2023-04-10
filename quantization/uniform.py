import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))



def quantize_uniform(data, n_bits, clip, device='cuda'):
    clip = .5
    w_c = data.clamp(-clip, clip)
    b = torch.pow(torch.tensor(2.0), 1 - n_bits).to(device)
    w_q = clip * torch.min(b * torch.round(w_c / (b * clip)), 1 - b)
    return w_q


class QConv2d(nn.Conv2d):
    def __init__(self, quant_args=None, init_args=None, *kargs, **kwargs):
        super(QConv2d, self).__init__(*kargs, **kwargs)
        self.weight = init_args
        self.quant_args = quant_args

    def forward(self, inputs):
        self.quantize_params()
        return F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def quantize_params(self):
        unquantized_weights = rgetattr(self,'weight')
        quantized_weights = torch.nn.Parameter(quantize_uniform(unquantized_weights,self.quant_args,1,'cpu'))
        # print(quantized_weights)
        rsetattr(self, 'weight', quantized_weights)
        return


class QLinear(nn.Linear):
    def __init__(self, quant_args=None, init_args=None, *kargs, **kwargs):
        super(QLinear, self).__init__(*kargs, **kwargs)
        self.weight = init_args
        self.quant_args = quant_args

    def forward(self, inputs):
        self.quantize_params()
        return F.linear(inputs, self.weight, self.bias)

    def quantize_params(self):
        unquantized_weights = rgetattr(self,'weight')
        quantized_weights = torch.nn.Parameter(quantize_uniform(unquantized_weights,self.quant_args,1,'cpu'))
        rsetattr(self, 'weight', quantized_weights)
        return