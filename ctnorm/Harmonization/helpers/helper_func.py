import os
import pydicom
import nibabel as nib
import pickle
import numpy as np
import torch
import random
from skimage import filters


def create_minimum_dicom_header(pixel_array, slice_number, metadata, output_path, z_position, thickness=1.0):
    """
    Modify the provided metadata, update PixelData and SliceThickness, and save the image as a DICOM file.
    Ensures geometry information is consistent across slices.
    """
    # Copy the metadata to avoid modifying the original dataset
    ds = metadata
    # Ensure pixel_array is int16
    pixel_array = pixel_array.astype(np.int16)
    # Update PixelData with the new pixel array
    ds.PixelData = pixel_array.tobytes()
    ds.Rows, ds.Columns = pixel_array.shape
    # Update SliceThickness
    ds.SliceThickness = thickness
    # Set RescaleIntercept and RescaleSlope for correct intensity interpretation
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1.0
    # Ensure PixelRepresentation is set for signed integers
    ds.PixelRepresentation = 1  # 1 = signed integer; 0 = unsigned integer
    # Update Window Center and Width for visualization
    ds.WindowCenter = "-250"  # Midpoint of [-1000, 500]
    ds.WindowWidth = "1500"   # Width of the range [-1000, 500]
    # Update Image Position (Patient) for the z-coordinate
    ds.ImagePositionPatient = [float(metadata.ImagePositionPatient[0]), 
                                float(metadata.ImagePositionPatient[1]), 
                                float(z_position)]
    """
    if ds.ImageOrientationPatient != [1, 0, 0, 0, 1, 0]:
        print('Orientation not in RAS')
    """
    # Ensure InstanceNumber and generate a unique SOPInstanceUID for each slice
    ds.InstanceNumber = slice_number
    ds.SliceLocation = abs(z_position) if z_position is not None else None
    # Save the updated DICOM file
    ds.save_as(output_path)


"""
SUPPORTS '.nii.gz' and '.dcm' as fileout
"""
def save_volume(data, out_type, out_dir, m_type, f_name, meta=None, affine_in=None, target_scale=None):
    if out_type == '.nii.gz':
        if isinstance(affine_in, torch.Tensor):
            if affine_in.dim() == 3 and affine_in.shape[0] == 1:
                affine_in = affine_in.squeeze(0)
            affine_in = affine_in.cpu().numpy()
        if target_scale is not None:
            affine_in[2,2] = 1.0
        
        data = np.transpose(data, (2,1,0)) # REQUIRED FOR RAS ALIGNMENT IN ITK
        output_folder = os.path.join(out_dir, m_type)
        os.makedirs(output_folder, exist_ok=True)
        out_f = os.path.join(output_folder, f_name+out_type)
        nii_to_save = nib.Nifti1Image(data, affine=affine_in)
        nib.save(nii_to_save, out_f)

    elif out_type == '.dcm':
        output_folder = os.path.join(out_dir, m_type, f_name)
        os.makedirs(output_folder, exist_ok=True)

        # Access directly
        if isinstance(meta, dict):
            dcm_metadata = meta['meta_data']
            z_start = meta['z_start']
            z_sign = meta['z_sign']
        else:
            # Access using prior saved metadata
            f_metadata = os.path.join(out_dir, 'metadata.pkl')
            with open(f_metadata, 'rb') as f:
                loaded_metadata = pickle.load(f)
                dcm_metadata = loaded_metadata['meta_data']
            z_start = loaded_metadata['z_start']
            z_sign = loaded_metadata['z_sign']

        if target_scale is None:
            z_positions = [z_start + z_sign * i * dcm_metadata.SliceThickness for i in range(data.shape[0])]
            out_thickness = float(dcm_metadata.SliceThickness)
        else:
            z_positions = [z_start + z_sign * i * 1.0 for i in range(data.shape[0])]
            out_thickness = 1.0
        
        for i in range(data.shape[0]):
            slice_path = os.path.join(output_folder, f"slice_{i + 1:03d}.dcm")
            create_minimum_dicom_header(data[i,:,:], i + 1, dcm_metadata, slice_path, z_positions[i], thickness=out_thickness)
    else:
        raise NotImplementedError('{} extension not yet supported!'.format(out_type))


def save_metric(data, out_type, out_dir, metrics_to_c, f_name, affine_in, target_scale=None):
    """Converts a PyTorch tensor to a NumPy array if needed."""
    if isinstance(affine_in, torch.Tensor):  # Check if it's a tensor
        if affine_in.dim() == 3 and affine_in.shape[0] == 1:  # Shape [1, 4, 4]
            affine_in = affine_in.squeeze(0)  # Remove batch dimension → [4, 4]
        affine_in = affine_in.cpu().numpy()  # Convert to NumPy (ensures CPU conversion)
    if target_scale is not None:
        affine_in[2,2] = 1.0
  
    for metric in metrics_to_c:
        if metric.lower() == 'sobel':
            output_folder = os.path.join(out_dir, metric)
            os.makedirs(output_folder, exist_ok=True)
            out_fname = os.path.join(output_folder, f_name+'.{}'.format(out_type))
            vol_map = _apply_filters(data)
            vol_map = vol_map.transpose(1,2,0)
            nii_to_save = nib.Nifti1Image(vol_map , affine=affine_in)
            nib.save(nii_to_save, out_fname)
        elif metric.lower() == 'emphysema':
            continue
        else:
            continue


def _apply_filters(image_3d):
    sobel_map = np.zeros_like(image_3d)
    for z in range(image_3d.shape[0]):
        slice_image = image_3d[z,:,:]
        slice_min = slice_image.min()
        slice_max = slice_image.max()
        slice_normalized = (slice_image - slice_min) / (slice_max - slice_min)
        sobel_map[z,:,:] = filters.sobel(slice_normalized)
    return sobel_map


"""
Make directories; input is an instance of string
"""
def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#############################
# image-specific operation #
# ###########################
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), intercept=-1000.):
    tensor = tensor.cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img_np = tensor.numpy()
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    if out_type == np.uint16:
        img_np = (img_np * 1500.0).round()
    if out_type == np.int16:
        img_np = (img_np * 1500.0).round() + intercept
    return img_np.astype(out_type)

