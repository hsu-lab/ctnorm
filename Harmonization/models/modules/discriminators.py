import torch.nn as nn
import torch.nn.functional as F
from . import block as B
import torch.nn.utils as utils
import functools
import torch
import sys


"""
Standard VGG style Discriminator with Spectral Normalization
"""
class Discriminator_VGG_64_SN(nn.Module):
    def __init__(self, in_nc, base_nf):
        super(Discriminator_VGG_64_SN, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, True)
        """
        Apply spectral normalization to a pramaters in a given module.
        - Spectral normalization stabilizes the training of discriminators (critics) 
        in Generative Adversarial Networks (GANs) by rescaling the weight tensor with spectral
        norm \sigmaσ of the weight matrix calculated using power iteration method.
        """
        self.conv0 = utils.spectral_norm(nn.Conv3d(in_nc, base_nf, kernel_size=3, stride=1, padding=1))
        self.conv1 = utils.spectral_norm(nn.Conv3d(base_nf, base_nf, kernel_size=3, stride=(1, 2, 2), padding=1))
        # 32, 64
        self.conv2 = utils.spectral_norm(nn.Conv3d(base_nf, base_nf*2, kernel_size=3, stride=1, padding=1))
        self.conv3 = utils.spectral_norm(nn.Conv3d(base_nf*2, base_nf*2, kernel_size=3, stride=2, padding=1))
        # 16, 128
        self.conv4 = utils.spectral_norm(nn.Conv3d(base_nf*2, base_nf*4, kernel_size=3, stride=1, padding=1))
        self.conv5 = utils.spectral_norm(nn.Conv3d(base_nf*4, base_nf*4, kernel_size=3, stride=2, padding=1))
        # 8, 256
        self.conv6 = utils.spectral_norm(nn.Conv3d(base_nf*4, base_nf*8, kernel_size=3, stride=1, padding=1))
        self.conv7 = utils.spectral_norm(nn.Conv3d(base_nf*8, base_nf*8, kernel_size=3, stride=2, padding=1))
        # 4, 512
        # classifier
        self.linear0 = utils.spectral_norm(nn.Linear(512 * 4 * 4 * 4, 512)) # for 1.0mm scale
        # self.linear0 = utils.spectral_norm(nn.Linear(512 * 2 * 2 * 2, 512)) # for 2.0mm scale
        self.linear1 = utils.spectral_norm(nn.Linear(512, 1))

    def forward(self, x):   # input size is 64x64
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


# WGAN paper  https://arxiv.org/abs/1708.00961
# VGG style Discriminator with input size 64*64
class WGAN_Discriminator_VGG_64(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu'):
        super(WGAN_Discriminator_VGG_64, self).__init__()
        # features
        # batch x channel x t  x  h  x  w [batch x    1    x 32 x  64 x  64]
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=3, stride=(1, 2, 2), norm_type=norm_type, act_type=act_type)
        # batch x    64   x 32 x  32 x  32
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=3, stride=2, norm_type=norm_type, act_type=act_type)
        # batch x    128  x 16 x  16 x  16
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, act_type=act_type)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=3, stride=2, norm_type=norm_type, act_type=act_type)
        # batch x    256  x 8 x   8 x  8
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5 )
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8 * 8, 1024), nn.LeakyReLU(0.2, True), nn.Linear(1024, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

