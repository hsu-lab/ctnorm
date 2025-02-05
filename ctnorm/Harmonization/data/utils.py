import os
import numpy as np
import torch
import logging
import h5py
import nibabel as nib
import pydicom


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

def _get_pixels_hu(scans, apply_lut):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    if apply_lut:
        image[image == -2000] = 0
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
        return np.array(image, dtype=np.int16)
    else:
        return image

def read_data(vol_pth, ext, apply_lut_for_dcm=True):
    if ext == 'nii' or ext == 'nii.gz':
        vol_pth = vol_path + '.nii.gz'
        data = nib.load(vol_pth)
        affine_info, header_info = data.affine, {}
        data = _check_transpose(data.get_fdata())
    elif ext == 'h5':
        with h5py.File(vol, 'r') as file:
            data = _check_transpose(file['data'][:,:,:])
            affine_info, header_info = np.eye(4), {}
    elif ext == 'dcm':
        dicom_files = [f for f in os.listdir(vol_pth) if f.endswith('.dcm')]
        dicom_files.sort()
        slices = []
        z_start = None
        for index, s in enumerate(dicom_files):
            dcm = pydicom.dcmread(os.path.join(vol_pth, s))
            slices.append(dcm)
            if index == 0:
                metadata_only = pydicom.dcmread(os.path.join(vol_pth, s)) # stop_before_pixels=True
                """
                metadata = pydicom.Dataset()
                # Copy all non-pixel data elements
                for elem in dcm.iterall():
                    if elem.tag != (0x7FE0, 0x0010):  # Skip PixelData
                        metadata.add(elem)
                z_start = float(metadata_only.ImagePositionPatient[2])  # Starting Z position
                z_sign = -1 if z_start < 0 else 1  # Determine the sign of the Z position (top-bottom/bottom-top)
                """
        data = _get_pixels_hu(slices, apply_lut_for_dcm)
        z_sign = 1 if slices[-1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2] else -1
        z_start = slices[0].ImagePositionPatient[2]
        affine_info, header_info = np.eye(4), {'z_start':z_start, 'z_sign':z_sign, 'meta_data':metadata_only}
    else:
        raise ValueError('Unknown file format!... Expects only `nii`, `nii.gz`, or `dcm`')
    return affine_info, header_info, data.astype(np.int16)


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