import logging
import torch
import torch.nn as nn
import sys
from Harmonization.models import lr_scheduler as lr_scheduler
from Harmonization.models import networks as networks
from Harmonization.models.base_model import BaseModel
from Harmonization.models.modules.loss import GANLoss, GradientPenaltyLoss
"""
Amp allows to experiment with different pure and mixed precision modes. 
Commonly-used default modes are chosen by selecting an “optimization level” or opt_level; each opt_level 
establishes a set of properties that govern Amp’s implementation of pure or mixed precision training.
- opt_level: 01 = Mixed Precision (recommended for typical use)
"""
from apex import amp
import apex
logger = logging.getLogger('base')


"""
Instantiates both generator and discriminator, put models in training, define losses, optimizers and schedulers
"""
class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        self.opt = opt
        if self.opt['is_train']:
            # Define discriminator 
            self.netD = networks.define_D(opt).to(self.device)  # D
            self.netG.train()
            self.netD.train()
            raise NotImplementedError('Train mode not implemented yet!')
        self.print_network(use_logger=True)
        self.load()  # load G and D if needed

    """
    Initializes model for mixed precision training, depending on 'opt_level'. Below is sample implementation
    -----------------------------------------------------------------------
    # Declare model and optimizer as usual, with default (FP32) precision
    model = torch.nn.Linear(D_in, D_out).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Allow Amp to perform casts as required by the opt_level
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    ...
    # loss.backward() becomes:
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    """
    def initialize_amp(self):
        [self.netG, self.netD], [self.optimizer_G, self.optimizer_D] = \
        amp.initialize([self.netG, self.netD], [self.optimizer_G, self.optimizer_D],
                       opt_level=self.opt['opt_level'], num_losses = 2)
        if self.opt['gpu_id']: 
            """
            Implements data parallelism at the module level. This container parallelizes 
            the application of the given module by splitting the input across the specified
            """
            assert torch.cuda.is_available()
            self.netG = nn.DataParallel(self.netG)
            self.netD = nn.DataParallel(self.netD)

    def test(self, data):
        self.netG_eval.eval()
        if self.opt['precision'] == 'fp16':
            var_L_eval = self.var_L.half()
        else:
            var_L_eval = self.var_L

        HR_ot = int(self.opt['dataset_opt']['scale'] * self.ot)
        num_HR, H, W = int(var_L_eval.size(2) * self.opt['dataset_opt']['scale']), var_L_eval.size(3), var_L_eval.size(4)
        pt_HR = int(self.opt['dataset_opt']['tile_z'] * self.opt['dataset_opt']['scale'])
        self.fake_H = torch.empty(1, 1,  num_HR, H, W, device=self.device)
        # ---------------------------- #
        #    Sliding Tile Inference    #
        # ---------------------------- #
        with torch.no_grad():
            for row in range(0, var_L_eval.size(3), self.opt['dataset_opt']['tile_xy']):
                for column in range(0, var_L_eval.size(4), self.opt['dataset_opt']['tile_xy']):
                    LR_chunked = var_L_eval[:, :, :, row:row+self.opt['dataset_opt']['tile_xy'], column:column+self.opt['dataset_opt']['tile_xy']]
                    if self.opt['precision'] == 'fp16':
                        tmp_chunk_along_z = torch.empty(self.nt, 1, pt_HR, self.opt['dataset_opt']['tile_xy'], self.opt['dataset_opt']['tile_xy'],
                                            dtype=torch.half, device=self.device)
                    else:
                        tmp_chunk_along_z = torch.empty(self.nt, 1, pt_HR, self.opt['dataset_opt']['tile_xy'], self.opt['dataset_opt']['tile_xy'],
                                            device=self.device)

                    # iterate over chunks
                    for i in range(0, self.nt - 1):
                        tmp_chunk_along_z[i, :, :, :, :] = self.netG_eval(LR_chunked[:, :, i*(self.pt-self.ot):i*(self.pt-self.ot)+self.pt, :, :])
                    # add the last chunk
                    tmp_chunk_along_z[-1, :, :, :, :] = self.netG_eval(LR_chunked[:, :, -self.pt:, :, :])

                    reconstructed_z = torch.empty(1, 1, num_HR, self.opt['dataset_opt']['tile_xy'],
                                                self.opt['dataset_opt']['tile_xy'], device=self.device)
                    # stitch volume along z direction with overlap
                    stitch_mask = torch.zeros_like(reconstructed_z, device=self.device)
                    for i in range(0, self.nt - 1):
                        ts, te = i * (pt_HR - HR_ot), i * (pt_HR - HR_ot) + pt_HR
                        reconstructed_z[0, 0, ts:te, :, :] = (reconstructed_z[0, 0, ts:te, :, :] * stitch_mask[0, 0, ts:te, :, :] + 
                                                                tmp_chunk_along_z[i,...].float() * (2 - stitch_mask[0, 0, ts:te, :, :])) / 2
                        stitch_mask[0, 0, ts:te, :, :] = 1.
                    reconstructed_z[0, 0, -pt_HR:, :, :] = \
                        (reconstructed_z[0, 0, -pt_HR:, :, :] * stitch_mask[0, 0, -pt_HR:, :, :] +
                        tmp_chunk_along_z[-1,...].float() * (2 - stitch_mask[0, 0, -pt_HR:, :, :])) / 2
                    self.fake_H[0, 0, :, row:row+self.opt['dataset_opt']['tile_xy'], column:column+self.opt['dataset_opt']['tile_xy']] = reconstructed_z

        if self.opt['is_train']:
            self.netG.train()


    """
    Feed in LR and HR data to models, calculate loss and optimize gradients
    """
    def optimize_parameters(self, step):
        pass

    """
    Returns the log dictionary that contains the gan (generator & discriminator) losses
    """
    def get_current_log(self):
        return self.log_dict


    """
    Prints out both the generator and discriminator network structure
    """
    def print_network(self, use_logger=False):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if use_logger:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if use_logger:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)
            if self.cri_fea:  # F, Perceptual Network (not currently used)
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                    self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)


    """
    Load the generator and discriminator weights if provided
    """
    def load(self, use_logger=True):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None: # load, model weights, if path is not None
            self.load_network(load_path_G, self.netG)
            if use_logger:
                logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
        load_path_D = self.opt['path'].get('pretrain_model_D')
        # load if opt['is_train'] is 'Train' and discriminator weight path is not None
        if self.opt['is_train'] and load_path_D is not None:
            self.load_network(load_path_D, self.netD)
            if use_logger:
                logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))


    """
    Save model weights, both generator and discriminator, at a particular iteration
    """
    def save(self, iter_step):
        # 'self.save_network()' is implemented in base class
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
