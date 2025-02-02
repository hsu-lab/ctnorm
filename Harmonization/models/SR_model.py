from collections import OrderedDict
import torch
import torch.nn as nn
from Harmonization.models import lr_scheduler as lr_scheduler
from Harmonization.models import networks as networks
from Harmonization.models.base_model import BaseModel
from apex import amp
import apex
import torch.nn.functional as F
import sys
import numpy as np
import logging
logger = logging.getLogger('base')


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        train_opt = opt['train']
        if self.is_train:
            self.netG.train()
            loss_type = train_opt.get('pixel_criterion', 'l1')
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'huber':
                self.cri_pix = nn.HuberLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))

            self.l_pix_w = train_opt.get('pixel_weight', 1)
            # set-up optimizers
            wd_G = train_opt.get('weight_decay_G', 0)
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            
            if opt["network_G"]["which_model_G"] == "fsrcnn":
                self.optimizer_G = torch.optim.Adam([
                        {'params': self.netG.first_part.parameters()},
                        {'params': self.netG.mid_part.parameters()},
                        {'params': self.netG.last_part.parameters(), 'lr':float(train_opt.get('lr_G', 1e-05)) * 0.1}
                        ], lr=float(train_opt.get('lr_G', 1e-05)))
            else:
                self.optimizer_G = torch.optim.Adam(
                    optim_params, lr=float(train_opt.get('lr_G', 1e-05)), weight_decay=wd_G, betas=(float(train_opt.get('beta1_G', 0.9)), float(train_opt.get('beta2_G', 0.99))))
            self.optimizers.append(self.optimizer_G)

            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt.get('lr_steps', [20e3, 40e3, 60e3]),
                                                         restarts=train_opt.get('restarts', None), # null
                                                         weights=train_opt.get('restart_weights', None), # null
                                                         gamma=train_opt.get('lr_gamma', 0.5),
                                                         clear_state=train_opt.get('clear_state', None))) # None
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(optimizer, train_opt['T_period'],
                                                               eta_min=train_opt['eta_min'],
                                                               restarts=train_opt['restarts'],
                                                               weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('Choose MultiStepLR or CosineAnnealingLR')
            self.log_dict = OrderedDict()
        self.print_network()
        self.load()

    def initialize_amp(self):
        self.netG, self.optimizer_G = amp.initialize(self.netG, self.optimizer_G, opt_level=self.opt['opt_level'])
        if self.opt['gpu_id']:
            assert torch.cuda.is_available()
            self.netG = nn.DataParallel(self.netG)

    def test(self, data):
        self.netG_eval.eval()
        if self.opt['precision'] == 'fp16':
            var_L_eval = self.var_L.half()
        else:
            var_L_eval = self.var_L

        HR_ot = int(self.opt['dataset_opt']['scale'] * self.ot)
        num_HR, H, W = int(var_L_eval.size(2) * self.opt['dataset_opt']['scale']), var_L_eval.size(3), var_L_eval.size(4)
        pt = int(self.opt['dataset_opt']['tile_z'] * self.opt['dataset_opt']['scale'])
        self.fake_H = torch.empty(1, 1,  num_HR, H, W, device=self.device)
        # ---------------------------- #
        #    Sliding Tile Inference    #
        # ---------------------------- #
        with torch.no_grad():
            for row in range(0, var_L_eval.size(3), self.opt['dataset_opt']['tile_xy']):
                for column in range(0, var_L_eval.size(4), self.opt['dataset_opt']['tile_xy']):
                    LR_chunked = var_L_eval[:, :, :, row:row+self.opt['dataset_opt']['tile_xy'], column:column+self.opt['dataset_opt']['tile_xy']]
                    if self.opt['precision'] == 'fp16':
                        tmp_chunk_along_z = torch.empty(self.nt, 1, pt, self.opt['dataset_opt']['tile_xy'], self.opt['dataset_opt']['tile_xy'],
                                            dtype=torch.half, device=self.device)
                    else:
                        tmp_chunk_along_z = torch.empty(self.nt, 1, pt, self.opt['dataset_opt']['tile_xy'], self.opt['dataset_opt']['tile_xy'],
                                            device=self.device)
                    # iterate over chunks
                    for i in range(0, self.nt - 1):
                        tmp_chunk_along_z[i, :, :, :, :] = self.netG_eval(LR_chunked[:, :, i*(pt-self.ot):i*(pt-self.ot)+pt, :, :])
                    # add the last chunk
                    tmp_chunk_along_z[-1, :, :, :, :] = self.netG_eval(LR_chunked[:, :, -pt:, :, :])

                    reconstructed_z = torch.empty(1, 1, num_HR, self.opt['dataset_opt']['tile_xy'],
                                                self.opt['dataset_opt']['tile_xy'], device=self.device)
                    # stitch volume along z direction with overlap
                    stitch_mask = torch.zeros_like(reconstructed_z, device=self.device)
                    for i in range(0, self.nt - 1):
                        ts, te = i * (pt - HR_ot), i * (pt - HR_ot) + pt
                        reconstructed_z[0, 0, ts:te, :, :] = (reconstructed_z[0, 0, ts:te, :, :] * stitch_mask[0, 0, ts:te, :, :] + 
                                                                tmp_chunk_along_z[i,...].float() * (2 - stitch_mask[0, 0, ts:te, :, :])) / 2
                        stitch_mask[0, 0, ts:te, :, :] = 1.
                    reconstructed_z[0, 0, -pt:, :, :] = \
                        (reconstructed_z[0, 0, -pt:, :, :] * stitch_mask[0, 0, -pt:, :, :] +
                        tmp_chunk_along_z[-1,...].float() * (2 - stitch_mask[0, 0, -pt:, :, :])) / 2
                    self.fake_H[0, 0, :, row:row+self.opt['dataset_opt']['tile_xy'], column:column+self.opt['dataset_opt']['tile_xy']] = reconstructed_z

        if self.opt['is_train']:
            self.netG.train()

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        with amp.scale_loss(l_pix, self.optimizer_G) as scale_loss:
            scale_loss.backward()
        self.optimizer_G.step()
        self.log_dict['l_pix'] = l_pix.item()

    def get_current_log(self):
        return self.log_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_G = self.opt['path'].get('pretrained_G', None)
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
