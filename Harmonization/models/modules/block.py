from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


"""
Helper function, for selecting activation
- neg_slope: for leakyrelu and init of prelu
- n_prelu: for p_relu num_parameters
"""
def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    # intialize activation based on type
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


"""
Helper function, for selecting normalization layer
"""
def norm(norm_type, nc):
    print('norm type:', norm_type)
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm3d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm3d(nc, affine=False)
    elif norm_type =='layer':
        layer = Layernorm(nc,affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


"""
Note: If dilation is 1, it is a normal convolution operation. If the dilation rate is 2, then there is skip 
of one pixel per input. In general, if there is a dilation rate of n, skip of n-1 pixels per input. 
- Receptive field will increase as dilation rate is increased.
- Number of elements of filter remains the same but with the increase in dilation rate, they will cover 
more coverage.
If dilation is not 1, we calculate the padding size such that input image and output feature map is same
"""
def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


"""
Helper function to create nn.Sequential module give a tuple data created from 'conv_block' data
"""
def sequential(*args):
    # raise error if input tuple has length 1 It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        # if 'module' is instance of 'nn.Sequential', we unpack the childrens of 'module' since it will have more than one layer.
        # if 'module' is instance of 'nn.Module', it will only have one layer so no need to unpack
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


"""
Returns one colvolutions layer -> ConvLayer, Activation (if specified), BactchNorm (if specified)
"""
def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, bias=True, \
               norm_type=None, act_type='relu'):
    # get padding size, in case dilation is not 1, so that input image and feature map size is same
    padding = get_valid_padding(kernel_size, dilation) # padding = 1
    c = nn.Conv3d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias)
    a = act(act_type) if act_type else None # get activation type if provided
    n = norm(norm_type, out_nc) if norm_type else None # get normalization type if provided
    return sequential(c, n, a)


def s_conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, bias=True, \
               norm_type=None, act_type='relu'):
    '''
    Conv layer with padding, normalization, activation
    Conv -> Norm -> Activation
    '''
    padding = get_valid_padding(kernel_size, dilation)

    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)
        
    spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
    spatial_stride =  [1, stride[1], stride[2]]
    spatial_padding =  [0, padding[1], padding[2]]

    c = nn.Conv3d(in_nc, out_nc, spatial_kernel_size, stride=spatial_stride, padding=spatial_padding, \
            dilation=dilation, bias=bias)
    a = act(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None

    return sequential(c, n, a)


def t_conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, bias=True, \
               norm_type=None, act_type='relu'):
    '''
    Conv layer with padding, normalization, activation
    Conv -> Norm -> Activation
    '''
    padding = get_valid_padding(kernel_size, dilation)

    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)

    temporal_kernel_size = [kernel_size[0], 1, 1]
    temporal_stride =  [stride[0], 1, 1]
    temporal_padding =  [padding[0], 0, 0]

    c = nn.Conv3d(in_nc, out_nc, temporal_kernel_size, stride=temporal_stride, padding=temporal_padding, \
            dilation=dilation, bias=bias)
    a = act(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None

    return sequential(c, n, a)


class Layernorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(Layernorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


"""
Element-wise sum the output of a residual to its input
"""
class ShortcutBlock(nn.Module):
    # residual will be 'resnet_blocks' with 'lr_conv' combined together in nn.Sequential
    def __init__(self, residual):
        super(ShortcutBlock, self).__init__()
        self.res = residual
        self.skip_op = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_op.add(x, self.res(x))

    # '__repr__' is a special method used to represent a class’s objects as a string 
    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.res.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

