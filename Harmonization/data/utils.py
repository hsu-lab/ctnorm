import os
import numpy as np
import torch
import logging
import h5py
import nibabel as nib


def _check_transpose(vol):
    # Determine the order and transpose to D, H, W
    if vol.shape[0] < vol.shape[1] and vol.shape[0] < vol.shape[2]:
        # Case: Z, X, Y (already correct)
        pass
    elif vol.shape[2] < vol.shape[0] and vol.shape[2] < vol.shape[1]:
        # Case: X, Y, Z
        vol = vol.transpose((2, 0, 1))
    else:
        raise ValueError(f"Unexpected data shape: {vol.shape}. Unable to determine the correct transpose.")
    return vol


def read_data(vol, ext):
    if ext == 'nii' or ext == 'nii.gz':
        data = nib.load(vol)
        affine_info = data.affine
        data = _check_transpose(data.get_fdata())
    elif ext == 'h5':
        affine_info = np.eye(4)
        with h5py.File(vol, 'r') as file:
            data = _check_transpose(file['data'][:,:,:])
    elif ext == 'dcm':
        # DICOM handling code ...
        raise NotImplementedError("DICOM format handling not implemented yet.")
    else:
        raise ValueError('Unknown file format!... Expects only `nii`, `nii.gz`, or `dcm`')
    o = {'affine':torch.from_numpy(affine_info), 'data':data.astype(np.int16)}
    return o


"""
Converts incoming data input (uint16 - D,H,W) to torch tensor float in a given range range [0, 1500].
- For dataloader using, inplace operation
- Clip value to lung window range
"""
class ImgToTensor(object):
    def __call__(self, img, min_HU_clip=-1000., max_HU_clip=500., move_HU=1000.):
        img = np.clip(img.astype(np.float32), min_HU_clip, max_HU_clip)
        img += move_HU
        img = torch.from_numpy(img)
        return torch.unsqueeze(img.float().div_(img.max()), axis=0)