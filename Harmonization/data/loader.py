import torch.utils.data as data
from .utils import *
import os


class Loader(data.Dataset):
    def __init__(self, opt):
        super(Loader, self).__init__()
        self.FILL_RATIO_THRESHOLD = 0.8
        self.opt = opt
        # Set up path to data I/O
        self.in_folder = opt.get('input_path')
        self.tar_folder = opt.get('target_path')
        self.mask_folder = opt.get('maskroot_path')
        self.data_type = opt.get('type')
        if self.opt['mode'] == 'train':
            self.ps = (opt['LR_slice_z'], opt['LR_size_xy'], opt['LR_size_xy'])      
        self.uids = opt['uids']
        self.mode = opt['mode']
        self.scale = opt.get('scale') # Scale by slice thickness
        self.ToTensor = ImgToTensor()


    def __getitem__(self, index):
        uid = self.uids[index]
        affine_in, in_data = read_data(os.path.join(self.in_folder, uid+'.{}'.format(self.data_type)), self.data_type)
        in_data = self.ToTensor(in_data, min_HU_clip=0., max_HU_clip=1500., move_HU=0.)
        out_dict = {'uid':uid, 'affine_info': torch.from_numpy(affine_in), 'dataroot_LR':in_data}
        if self.tar_folder:
            affine_out, tar_data = read_data(os.path.join(self.tar_folder, uid+'.{}'.format(self.data_type)), self.data_type)
            out_dict['dataroot_HR'] = tar_data
        return out_dict


    def __len__(self):
        return len(self.uids)