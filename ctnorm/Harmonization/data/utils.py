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


def _sort_dicoms(dicoms):
    """
    Sort the dicoms based om the image possition patient

    :param dicoms: list of dicoms
    """
    # find most significant axis to use during sorting
    # the original way of sorting (first x than y than z) does not work in certain border situations
    # where for exampe the X will only slightly change causing the values to remain equal on multiple slices
    # messing up the sorting completely)
    dicom_input_sorted_x = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[0]))
    dicom_input_sorted_y = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[1]))
    dicom_input_sorted_z = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[2]))
    diff_x = abs(dicom_input_sorted_x[-1].ImagePositionPatient[0] - dicom_input_sorted_x[0].ImagePositionPatient[0])
    diff_y = abs(dicom_input_sorted_y[-1].ImagePositionPatient[1] - dicom_input_sorted_y[0].ImagePositionPatient[1])
    diff_z = abs(dicom_input_sorted_z[-1].ImagePositionPatient[2] - dicom_input_sorted_z[0].ImagePositionPatient[2])
    if diff_x >= diff_y and diff_x >= diff_z:
        return dicom_input_sorted_x
    if diff_y >= diff_x and diff_y >= diff_z:
        return dicom_input_sorted_y
    if diff_z >= diff_x and diff_z >= diff_y:
        return dicom_input_sorted_z


def get_affine_matrix(sorted_dicoms):
    """
    Function to generate the affine matrix for a dicom series
    This method was based on (http://nipy.org/nibabel/dicom/dicom_orientation.html)

    :param sorted_dicoms: list with sorted dicom files
    """
    # Create affine matrix (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
    image_orient1 = np.array(sorted_dicoms[0].ImageOrientationPatient)[0:3]
    image_orient2 = np.array(sorted_dicoms[0].ImageOrientationPatient)[3:6]
    delta_r = float(sorted_dicoms[0].PixelSpacing[0])
    delta_c = float(sorted_dicoms[0].PixelSpacing[1])
    image_pos = np.array(sorted_dicoms[0].ImagePositionPatient)
    last_image_pos = np.array(sorted_dicoms[-1].ImagePositionPatient)

    if len(sorted_dicoms) == 1:
        # Single slice
        slice_thickness = 1
        if "SliceThickness" in sorted_dicoms[0]:
            slice_thickness = sorted_dicoms[0].SliceThickness
        step = - np.cross(image_orient1, image_orient2) * slice_thickness
    else:
        step = (image_pos - last_image_pos) / (1 - len(sorted_dicoms))
    # check if this is actually a volume and not all slices on the same location
    if np.linalg.norm(step) == 0.0:
        raise ConversionError("NOT_A_VOLUME")
    affine = np.array(
        [[-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -step[0], -image_pos[0]],
         [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -step[1], -image_pos[1]],
         [image_orient1[2] * delta_c, image_orient2[2] * delta_r, step[2], image_pos[2]],
         [0, 0, 0, 1]]
    )
    return affine # , np.linalg.norm(step)


def _get_pixels_hu(scans, apply_lut=True):
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
        
        # print(slices[0].ImagePositionPatient[2], slices[-1].ImagePositionPatient[2])
        # for s in slices:
        #     print(s.InstanceNumber, s.SliceLocation, s.ImagePositionPatient[2])
        sorted_slices = _sort_dicoms(slices)
        # print(sorted_slices[0].ImagePositionPatient[2], sorted_slices[-1].ImagePositionPatient[2])

        data = _get_pixels_hu(sorted_slices, apply_lut_for_dcm)
        # z_sign = 1 if slices[-1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2] else -1
        # z_start = slices[0].ImagePositionPatient[2]
        z_sign = 1 if sorted_slices[-1].ImagePositionPatient[2] > sorted_slices[0].ImagePositionPatient[2] else -1
        z_start = sorted_slices[0].ImagePositionPatient[2]
        # print('='*40)
        # for s in sorted_slices:
        #     print(s.InstanceNumber, s.SliceLocation, s.ImagePositionPatient[2])
        # print('-'*40)
        affine_info, header_info = get_affine_matrix(sorted_slices), {'z_start':z_start, 'z_sign':z_sign, 'meta_data':metadata_only}
    else:
        raise ValueError('Unknown file format!... Expects only `.nii`, `.nii.gz`, or `.dcm`')
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

