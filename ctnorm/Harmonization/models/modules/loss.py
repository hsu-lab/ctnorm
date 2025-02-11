import torch
import torch.nn as nn


# Define GAN loss: [gan | lsgan | wgan-gp | wgan-gp0 | hinge ]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif 'wgan' in self.gan_type:
            def _wgan_loss(input, target):
                return -1 * input.mean() if target else input.mean()
            self.loss = _wgan_loss

        # if 'gan_type' is 'hinge', we define the '_hing_loss' function and store the function as an attribute
        elif self.gan_type =='hinge':
            def _hinge_loss(input, target):
                if target:
                    return nn.ReLU()(1.0 - input).mean()
                else: 
                    return nn.ReLU()(1.0 + input).mean()
            self.loss = _hinge_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        # if 'self.gan_type' is 'wgan' or 'hinge', we do not need target label 
        if 'wgan' in self.gan_type or self.gan_type == 'hinge':
            return target_is_real
        # otherwise, return tensors with 1 if real label or 0 if fake label
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


# gradient penalty regularization
# wgan-gp or zero centered penalty https://github.com/LMescheder/GAN_stability
class GradientPenaltyLoss(nn.Module):
    def __init__(self, center):
        super(GradientPenaltyLoss, self).__init__()
        self.center = center

    @staticmethod
    def compute_grad2(x_in, d_out):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(dim=1)
        return reg

    def gp_reg(self, x_in, d_out):
        if self.center == 0.:
            return self.compute_grad2(x_in, d_out).mean()
        else:
            return (self.compute_grad2(x_in, d_out).sqrt() - self.center).pow(2).mean()

    def forward(self, x_in, d_out):
        return self.gp_reg(x_in, d_out)
