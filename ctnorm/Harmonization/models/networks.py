import logging
import torch
import torch.nn as nn
import ctnorm.Harmonization.models.modules.discriminators as D_arch
import ctnorm.Harmonization.models.modules.generators as G_arch
import torch.nn.init as init
import math
import sys
from .build_unet import *

logger = logging.getLogger('base')


"""
Define Generator Network and Initialize weights
"""
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    # Initialze model
    if which_model == 'sr_resnet':
        netG = G_arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu')
    elif which_model == 'unet3D': # Pix2Pix
        netG = UNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'])
    elif which_model == 'vanilla':  # WGAN
        netG = G_arch.VanillaNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
                nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDB':
        netG = G_arch.RRDB_Gen(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], upscale=opt_net['scale'])

    # initialize models if in training mode. for 'fsrcnn', do an initialization different than the rest of other models
    if opt['is_train']: 
        if which_model != 'fsrcnn': # fsrcnn has its own initialization
            initialize_weights(netG, scale=0.1)
        else:
            initialize_FSRCNN_weights(netG)
    return netG


"""
Define Discriminator network
"""
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']
    # Initialize model
    if which_model == 'discriminator_vgg_64_SN': # SNGAN | Pix2Pix Disc
        netD = D_arch.Discriminator_VGG_64_SN(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'])
    elif which_model == 'wgan_discriminator_vgg_64': # WGAN Disc
        netD = D_arch.WGAN_Discriminator_VGG_64(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], act_type=opt_net['act_type'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    initialize_weights(netD, scale=1)
    return netD


"""
8bit Quantization network
"""
def define_Quant(model_fp32):
    return G_arch.QuantizedModel(model_fp32)


"""
Initialize layers for all models other than FSRCNN
"""
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            # if isinstance(m, nn.Conv2d):
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


"""
Initialize FSRCNN model specifically
"""
def initialize_FSRCNN_weights(net_l):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.first_part:
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in net.mid_part:
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(net.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(net.last_part.bias.data)
