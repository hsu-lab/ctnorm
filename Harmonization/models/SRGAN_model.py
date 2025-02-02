import logging
import torch
import torch.nn as nn
import sys
from collections import OrderedDict
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
        train_opt = self.opt['train']
        if self.opt['is_train']:
            # Define discriminator 
            self.netD = networks.define_D(opt).to(self.device)  # D
            self.netG.train()
            self.netD.train()

            # Define losses, optimizer and scheduler for training mode
            if train_opt.get('pixel_weight', 1) > 0:
                l_pix_type = train_opt.get('pixel_criterion', 'l1')
                if l_pix_type == 'l1':
                    # MAE
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    # MSE
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt.get('pixel_weight', 1) # get pixel weight
            else:
                logger.info('Skipping pixel loss ...')
                self.cri_pix = None

            # Parmater `gan_type` already set depending on GAN model
            self.cri_gan = GANLoss(train_opt['gan_type'], real_label_val=1.0, fake_label_val=0.0).to(self.device)
            self.l_gan_w = float(train_opt.get('gan_weight', 5e-3))

            self.D_update_ratio = train_opt.get('D_update_ratio', 1)
            self.D_init_iters = train_opt.get('D_init_iters', 0)

            # Initialize gradient penality
            if "wgan" in train_opt['gan_type']:
                if train_opt['gan_type'] == "wgan-gp":                    
                    self.cri_gp = GradientPenaltyLoss(center=1.).to(self.device)
                elif train_opt['gan_type'] == "wgan-gp0":
                    self.cri_gp = GradientPenaltyLoss(center=0.).to(self.device)
                else:
                    raise NotImplementedError("{:s} not found".format(train_opt['gan_type']))
                self.l_gp_w = 10. # Weight for gradient penality

            # Optimizers for generator
            wd_G = train_opt.get('weight_decay_G', 0)
            optim_params = []
            for k, v in self.netG.named_parameters():  # Optimizer for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))

            # Initialize generator optimizer with parameters that only require gradients
            self.optimizer_G = torch.optim.Adam(optim_params, lr=float(train_opt.get('lr_G', 1e-5)), \
                weight_decay=wd_G, betas=(float(train_opt.get('beta1_G', 0.5)), float(train_opt.get('beta2_G', 0.999))))

            self.optimizers.append(self.optimizer_G) # add optimizer to list defined in base class
            # Weight decay for discriminator
            wd_D = train_opt.get('weight_decay_D', 0)
            # Optimizer for discriminator
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=float(train_opt.get('lr_D', 1e-5)), \
                weight_decay=wd_D, betas=(float(train_opt.get('beta1_D', 0.5)), float(train_opt.get('beta2_D', 0.999))))
            self.optimizers.append(self.optimizer_D) # add optimizer to list defined in base class
            # Configure schedulers
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
        self.optimizer_G.zero_grad() # zero-gradient
        self.fake_H = self.netG(self.var_L)
        l_g_total = 0
        # ------------------------ #
        # Calculate Generator Loss #
        # ------------------------ #
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                # Calculate loss between output from generator (given 'LR' data) & real 'HR'
                l_g_pix = self.cri_pix(self.fake_H, self.real_H)
                l_g_pix = self.l_pix_w * l_g_pix # all pixels weighted equally with 1
                l_g_total += l_g_pix

            """
            # Perceptual loss if defined
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.real_H).detach()
                fake_fea = self.netF(self.fake_H) 
                # calculate feature loss, either 'L1' or 'MSE' loss
                l_g_fea = self.cri_fea(fake_fea, real_fea)
                l_g_total += self.l_fea_w * l_g_fea
            """

            # Fake prediction 
            pred_g_fake = self.netD(self.fake_H)
            # If 'gan_type' is 'hinge', no need to call the 'self.cri_gan()' function, we can 
            # calculate the gan generator loss
            if self.opt['train']['gan_type'] == 'hinge':
                l_g_gan = -pred_g_fake.mean()
            else:
                l_g_gan = self.cri_gan(pred_g_fake, True)
            
            l_g_total += self.l_gan_w * l_g_gan
            # Step-up optimizer
            with amp.scale_loss(l_g_total , self.optimizer_G, loss_id=0) as errG_scaled:
                errG_scaled.backward()
            self.optimizer_G.step()

        # Optimize discriminator
        self.optimizer_D.zero_grad()
        l_d_total = 0
        # For 'wgan-gp0', we do need gradient on real data
        if self.opt['train']['gan_type'] == 'wgan-gp0':
            self.real_H.requires_grad_()

        # Get prediction from discriminator based on real 'HR' data
        pred_d_real = self.netD(self.real_H)
        # Get predictions form discriminator based on 'fake_HR' data
        pred_d_fake = self.netD(self.fake_H.detach())  # Detach to avoid back propogation to G

        # ---------------------------- #
        # Calcualte Discriminator Loss #
        # ---------------------------- #
        l_d_real = self.cri_gan(pred_d_real, True)
        l_d_fake = self.cri_gan(pred_d_fake, False)
        l_d_total = l_d_real + l_d_fake

        # If 'wgan' in 'gan_tpye', we calculate gradient penality, multiply with gradient penalty weight
        if 'wgan' in self.opt['train']['gan_type']:
            if self.opt['train']['gan_type'] == 'wgan-gp0':
                l_d_gp = self.cri_gp(self.real_H, pred_d_real)
            elif self.opt['train']['gan_type'] == 'wgan-gp':
                batch_size = self.real_H.size(0)
                eps = torch.rand(batch_size, device=self.device).view(batch_size, 1, 1, 1, 1)
                x_interp = (1 - eps) * self.real_H + eps * self.fake_H.detach()
                x_interp.requires_grad_()
                pred_d_x_interp = self.netD(x_interp)
                l_d_gp = self.cri_gp(x_interp, pred_d_x_interp)
            else:
                raise NotImplementedError('Gan type [{:s}] not recognized'.format(self.opt['train']['gan_type']))
            l_d_total += self.l_gp_w * l_d_gp

        # backpropogate loss, step-up optimizer
        with amp.scale_loss(l_d_total , self.optimizer_D, loss_id=1) as errD_scaled:
            errD_scaled.backward()
        self.optimizer_D.step()

        # Set logs - Log Generator  
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # Log Losses
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            """
            # Append perceptual loss/feature loss, if specified
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            """
            self.log_dict['l_g_gan'] = l_g_gan.item()
            # Append total loss (pixel loss (if specified) + feature loss (if specified) + gan loss/hinge loss)
            self.log_dict['l_g_total'] = l_g_total.item()

        # Log Discriminator
        self.log_dict['l_d_total'] = l_d_total.item()
        if 'wgan' in self.opt['train']['gan_type']:
            # Append gradient penalty
            self.log_dict['l_d_gp'] = l_d_gp.item()
            self.log_dict['w_dist'] = - ( l_d_real.item() + l_d_fake.item() )
        
        # D outputs (mean of output from real HR and mean of output from fake HR)
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

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
            """
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

    """
    Load the generator and discriminator weights if provided
    """
    def load(self, use_logger=True):
        load_path_G = self.opt['path'].get('pretrained_G', None)
        if load_path_G is not None: # load, model weights, if path is not None
            self.load_network(load_path_G, self.netG)
            if use_logger:
                logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
        load_path_D = self.opt['path'].get('pretrained_D', None)
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
