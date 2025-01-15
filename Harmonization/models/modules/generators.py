import torch.nn as nn
import torch
import torch.nn.functional as F
# from einops import rearrange,repeat
from . import block as B
import sys
import functools


"""
SR-ResNet Architecture
"""
class SRResNet(nn.Module):
    # in_nc=1, out_nc=1, nf=64, nb=8, upscale=1, norm_type=None, act_type='relu' 
    def __init__(self, in_nc, out_nc, nf, nb, upscale=1, norm_type='batch', act_type='relu'):
        super(SRResNet, self).__init__()
        fea_conv = nn.Conv3d(in_nc, nf, 3, 1, 1, bias=True)
        resnet_blocks = [ResNetBlock(nf, nf, 3, act_type=act_type) for _ in range(nb)]
        lr_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None)
        up = nn.Upsample(scale_factor=(upscale, 1., 1.,), mode='nearest')
        up_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        hr_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        last_conv = nn.Conv3d(nf, out_nc, 3, 1, 1, bias=True)
        if upscale == 1:# denoising
            self.net = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, lr_conv)),
                                    hr_conv, last_conv)
        else: # super-res
            self.net = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, lr_conv)),
                                    up, up_conv, hr_conv, last_conv)

    def forward(self, x):
        return self.net(x)


class QuantizedModel(nn.Module):
    # https://leimao.github.io/blog/PyTorch-Static-Quantization/
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized. This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x


# WGAN paper  https://arxiv.org/abs/1708.00961 only for denoising
# Low Dose CT Image Denoising Using a Generative Adversarial Network with Wasserstein Distance
# and Perceptual Loss
class VanillaNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb):
        super(VanillaNet, self).__init__()
        layers = [nn.Conv3d(in_nc, nf, 3, 1, 1), nn.ReLU()]
        for _ in range(2, nb):
            layers.extend([nn.Conv3d(nf, nf, 3, 1, 1), nn.ReLU()])
        layers.extend([nn.Conv3d(nf, out_nc, 3, 1, 1)]) # removed ReLU layer for stability
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''
    # in_nc=64, out_nc=64, act_type='relu', norm_type=None 
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, dilation=1, bias=True, \
                 norm_type=None, act_type='relu', res_scale=1):
        super(ResNetBlock, self).__init__()
        # has activation layer
        conv0 = B.conv_block(in_nc, out_nc, kernel_size, stride, dilation, bias, norm_type, act_type)
        # no activation layer
        conv1 = B.conv_block(out_nc, out_nc, kernel_size, stride, dilation, bias, norm_type, None)
        self.res = B.sequential(conv0, conv1)
        self.res_scale = res_scale
        self.skip_op = nn.quantized.FloatFunctional()

    def forward(self, x):
        res = self.skip_op.mul_scalar(self.res(x), self.res_scale)
        return self.skip_op.add(x, res)


"""
Dense block for RRBD MODEL
"""
class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """
    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels + growth_channels * 0, growth_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.conv2 = nn.Conv3d(channels + growth_channels * 1, growth_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.conv3 = nn.Conv3d(channels + growth_channels * 2, growth_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.conv4 = nn.Conv3d(channels + growth_channels * 3, growth_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.conv5 = nn.Conv3d(channels + growth_channels * 4, channels, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)
        return out


class ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """
    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)
        return out


"""
RRBD MODEL
"""
class RRDB_Gen(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=1):
        super(RRDB_Gen, self).__init__()
        self.conv1 = nn.Conv3d(in_nc, nf, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        # Feature extraction backbone network
        trunk = []
        for _ in range(nb):
            trunk.append(ResidualResidualDenseBlock(nf, 32))
        self.trunk = nn.Sequential(*trunk)
        self.conv2 = nn.Conv3d(nf, nf, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        # Reconnect a layer of convolution block after upsampling
        self.conv3 = nn.Sequential(
            nn.Conv3d(nf, nf, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        # Output layer
        self.conv4 = nn.Conv3d(nf, out_nc, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.conv3(out)
        out = self.conv4(out)
        out = torch.clamp_(out, 0.0, 1.0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)