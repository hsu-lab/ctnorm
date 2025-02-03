import os
import glob
import torch
import torch.nn as nn
import math
import Harmonization.models.networks as networks
from collections import OrderedDict
import copy
import numpy as np


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt['gpu_id']) if opt.get('gpu_id') is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.netG = networks.define_G(opt).to(self.device)

    """
    Get train data
    """
    def feed_train_data(self, data, need_HR=True):
        # get LR and HR data
        self.var_L = data['dataroot_LR'].to(self.device, non_blocking=True)
        if need_HR:
            self.real_H = data['dataroot_HR'].to(self.device, non_blocking=True)

    """
    Get test data
    """
    def feed_test_data(self, data, need_HR=False):
        self.var_L = data['dataroot_LR'].to(self.device, non_blocking=True)
        if need_HR:
            self.real_H = data['dataroot_HR'].to(self.device, non_blocking=True)
        self.pt = self.opt['dataset_opt']['tile_z'] # 32
        self.ot = self.opt['dataset_opt']['z_overlap'] # 4
        self.nt = 1 + math.ceil((self.var_L.size(2) - self.pt) / (self.pt - self.ot))

    """
    Copies fp32 model and convert to fp16
    """
    def half(self):
        if self.opt['precision'] == 'fp16':
            self.netG_eval = copy.deepcopy(self.netG).half()
        else:
            self.netG_eval = self.netG
            
    def prepare_quant(self, loader, not_loader=False):
        # PyTorch Static Quantization 
        # https://leimao.github.io/blog/PyTorch-Static-Quantization/
        fused_model = copy.deepcopy(self.netG)
        fused_model.eval()
        # fused conv3d + relu
        fused_model = torch.quantization.fuse_modules(fused_model.net, [["3", "4"],["5","6"]], inplace=True)
        for i in range(8):
            torch.quantization.fuse_modules(fused_model[1].res[i].res, [["0", "1"]], inplace=True)
        quantized_model = networks.define_Quant(model_fp32=fused_model)
        # config
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        quantized_model.qconfig = quantization_config
        torch.quantization.prepare(quantized_model, inplace=True)
        # calibration
        if not_loader:
            data = loader
        else:
            data = next(iter(loader))  
        # get a small chunk for calibration
        _ = quantized_model(data['LR'][:,:,:32 ,:,:])
        quantized_model = torch.quantization.convert(quantized_model, inplace=True)
        self.netG_eval = quantized_model

    def optimize_parameters(self):
        pass

    def get_current_visuals(self, data, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0, 0].float() # [1, 512, 512]
        out_dict['SR'] = self.fake_H.detach()[0, 0].float() # [1, 512, 512]
        if need_HR:
            out_dict['HR'] = self.real_H.detach().float()[0, 0, :]
        return out_dict

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    """
    For each schedulers, we step up the learning rate
    """
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    """
    Returns the current learning rate of scheduler for generator only
    """
    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    """
    Get string representation of a network and its parameters
    """
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    """
    Save the model 'state_dict' given the iteration step
    """
    def save_network(self, network, network_label, iter_step):
        save_filename = '{}_{}.pth'.format(iter_step, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    """
    Loads the model with weights
    """
    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)

    """
    Saves training state during training
    """
    def save_training_state(self, epoch, iter_step):
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        old_files = os.listdir(self.opt['path']['training_state'])
        for f in old_files:
            os.remove(os.path.join(self.opt['path']['training_state'], f))
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

