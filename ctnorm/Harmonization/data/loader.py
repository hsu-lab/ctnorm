import torch.utils.data as data
from .utils import *
import os
import pickle
import random


class Loader(data.Dataset):
    def __init__(self, opt):
        super(Loader, self).__init__()
        self.FILL_RATIO_THRESHOLD = 0.8
        self.opt = opt
        # Set up path to data I/O
        self.data_type = opt.get('in_dtype')
        self.in_uids = opt['in_uids_names']
        self.tar_uids = opt.get('tar_uids_names', None)
        self.mask_folder = opt.get('maskroot_path') # Currently not used!
        self.out_dir = opt.get('out')
        self.is_train = opt['is_train']
        # Define 3D patch size for training
        if self.is_train:
            self.ps = (opt['tile_z'], opt['tile_xy'], opt['tile_xy'])      
        self.scale = opt['scale'] # Scale by slice thickness
        self.ToTensor = ImgToTensor()


    def __getitem__(self, index):
        in_uid = self.in_uids[index]
        affine_in, header_in, in_data = read_data(in_uid, self.data_type)
        # Save metadata used during test
        if header_in and not self.is_train:
            metadata_out = os.path.join(self.out_dir, '--'.join(in_uid.lstrip('/').split('/')))
            os.makedirs(metadata_out, exist_ok=True)
            metadata_out_f = os.path.join(metadata_out, 'metadata.pkl')
            with open(metadata_out_f, 'wb') as f:
                pickle.dump(header_in, f)

        if self.tar_uids:
            tar_uid = self.tar_uids[index]
            affine_tar, header_tar, tar_data = read_data(tar_uid, self.data_type)

        if self.is_train and self.tar_uids:
            t, w, h = self.ps
            IMG_THICKNESS, IMG_WIDTH, IMG_HEIGHT = tar_data.shape
            # Calculate the required HR crop depth
            hr_crop_depth = int(t * self.scale)
            lr_thickness = in_data.shape[0]

            valid_crop = False
            while not valid_crop:
                # Randomly choose HR indices
                rnd_t_HR = random.randint(0, IMG_THICKNESS - hr_crop_depth)
                rnd_w = random.randint(0, IMG_WIDTH - w)
                rnd_h = random.randint(0, IMG_HEIGHT - h)
                # Compute corresponding LR starting index
                start_lr = round(rnd_t_HR / self.scale)
                # Check if the LR crop will fit
                if start_lr + t <= lr_thickness:
                    valid_crop = True
            in_data = in_data[start_lr:start_lr+t, rnd_h:rnd_h+h, rnd_w:rnd_w+w]
            tar_data = tar_data[rnd_t_HR:rnd_t_HR+hr_crop_depth, rnd_h:rnd_h+h, rnd_w:rnd_w+w]

        in_data = self.ToTensor(in_data, min_HU_clip=-1000., max_HU_clip=500., move_HU=1000.)
        out_dict = {'uid':'--'.join(in_uid.lstrip('/').split('/')), 'affine_info': torch.from_numpy(affine_in), 'dataroot_LR':in_data}
        if self.is_train:
            tar_data = self.ToTensor(tar_data, min_HU_clip=-1000., max_HU_clip=500., move_HU=1000.)
            out_dict['dataroot_HR'] = tar_data
        return out_dict


    def __len__(self):
        return len(self.in_uids)