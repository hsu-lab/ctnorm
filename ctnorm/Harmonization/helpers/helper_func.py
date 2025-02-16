import os
import pydicom
import nibabel as nib
import pickle
import numpy as np
import torch
import pandas as pd
import random
from .image_reorientation import reorient_image
import SimpleITK as sitk
from radiomics import featureextractor
import six
from skimage import filters
from scipy.ndimage import sobel
import pickle

# Set logging for radimioc rather than printing on screen
import radiomics
logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)


def _diagonal_matches(affine1, affine2, atol=1.0):
    """
    Check if the diagonal elements [0,0], [1,1], [2,2] of affine1 
    match those of affine2 within the given tolerance.
    Returns True if all three diagonal elements match within tolerance, 
    False otherwise.
    """
    diag1 = np.array([affine1[0,0], affine1[1,1], affine1[2,2]], dtype=float)
    diag2 = np.array([affine2[0,0], affine2[1,1], affine2[2,2]], dtype=float)
    return np.allclose(diag1, diag2, rtol=0, atol=atol)
    

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
def save_volume(data, out_type, out_dir, m_type, f_name, meta=None, affine_in=None, target_scale=None, reorient=True):
    if out_type == '.nii.gz':
        if isinstance(affine_in, torch.Tensor):
            if affine_in.dim() == 3 and affine_in.shape[0] == 1:
                affine_in = affine_in.squeeze(0)
            affine_in = affine_in.cpu().numpy()
        if target_scale is not None:
            affine_in[2,2] = 1.0 # Currently slice thickness harmonization assumes the CT is always mapped to 1mm
        
        data = np.transpose(data, (2,1,0)) # Required for ras alingment in ITK
        output_folder = os.path.join(out_dir, m_type)
        os.makedirs(output_folder, exist_ok=True)
        out_f = os.path.join(output_folder, f_name+out_type)
        nii_to_save = nib.Nifti1Image(data, affine=affine_in)
        # By Default: We reorient image as done in dicom2nifti library
        if reorient:
            nii_to_save = reorient_image(nii_to_save)
        nib.save(nii_to_save, out_f)

    elif out_type == '.dcm':
        output_folder = os.path.join(out_dir, m_type, f_name)
        os.makedirs(output_folder, exist_ok=True)

        # Access metada directly - used for nonDL methods; i.e. BM3D
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
        
        num_slices = data.shape[0]
        for i in range(num_slices):
            instance_number = num_slices - i
            slice_path = os.path.join(output_folder, f"slice_{i + 1:03d}.dcm")
            create_minimum_dicom_header(data[i,:,:], instance_number, dcm_metadata, slice_path, z_positions[i], thickness=out_thickness)
    else:
        raise NotImplementedError('{} extension not yet supported!'.format(out_type))


def save_metric(cases, metric, mod_name):
    if metric.lower() == 'sobel':
        in_cases = cases['img_pth']
        for case_idx in range(len(in_cases)):
            vol_pth = in_cases[case_idx]
            if os.path.isdir(vol_pth):
                raise ValueError("Harmonization output is in '.dcm' directory format! Please ensure it follows '.nii.gz' directory format")
            assert vol_pth.endswith('.nii.gz'), "Computing metrics only expects input to be .nii.gz"

            output_metric_to = os.path.join(vol_pth.split('Volume')[0], metric.title()) #  
            os.makedirs(output_metric_to, exist_ok=True)
            out_metric_to_name = os.path.join(output_metric_to, vol_pth.split('/')[-1])
            vol = nib.load(vol_pth)
            sobel_map = _compute_3d_sobel_map(vol.get_fdata())
            nii_to_save = nib.Nifti1Image(sobel_map, affine=vol.affine)
            nib.save(nii_to_save, out_metric_to_name)

    elif metric.lower() == 'snr':
        in_cases = cases['img_pth']
        snrs = []
        for case_idx in range(len(in_cases)):
            if case_idx == 0:
                output_metric_to = os.path.join(in_cases[case_idx].split('test')[0], metric)
                os.makedirs(output_metric_to, exist_ok=True)

            vol_pth = in_cases[case_idx]
            vol = nib.load(vol_pth)
            mean_signal = np.mean(vol.get_fdata())
            std_noise = np.std(vol.get_fdata())
            snr = snr_value = 10 * np.log10((mean_signal ** 2) / (std_noise ** 2)) if std_noise > 0 else 0
            snrs.append(snr)
        out_name = os.path.join(output_metric_to, '{}.pkl'.format(mod_name))
        with open(out_name, 'wb') as p_file:
            pickle.dump(snrs, p_file)
        
    elif metric.lower() == 'emphysema':
        pass

    elif metric.lower() == 'radiomic':
        in_cases = cases['img_pth']
        in_masks = cases['mask_pth']
        if len(in_masks) == 0:
            raise ValueError("Mask must be provided for computing 'Radiomic' features!")

        # Only define once
        if "extractor" not in globals():
            global extractor
            paramPath = './CT.yaml'
            extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)

        df_array = {'id':[]}
        for case_idx in range(len(in_cases)):
            if case_idx == 0:
                output_metric_to = os.path.join(in_cases[case_idx].split('test')[0], metric)
                os.makedirs(output_metric_to, exist_ok=True)

            in_case = sitk.ReadImage(in_cases[case_idx])
            in_mask = sitk.ReadImage(in_masks[case_idx])
            feats = _get_radiomics(extractor, in_case, in_mask)
            for feat in feats:
                if feat.startswith('original'):
                    if feat not in df_array:
                        df_array[feat] = []
                    df_array[feat].append(feats[feat].item())
            df_array['id'].append(in_cases[case_idx].split('/')[-1])
        df = pd.DataFrame(df_array)
        csv_name = os.path.join(output_metric_to, '{}.csv'.format(mod_name))
        df.to_csv(csv_name, index=False)
    else:
        raise NotImplementedError('Metric {} not implemented!'.format(metric))


def _compute_3d_sobel_map(volume):
    """
    Compute the Sobel map for each slice along the depth axis of a 3D volume.
    Parameters:
    volume (numpy.ndarray): 3D numpy array with shape (d, w, h).
    Returns:
    numpy.ndarray: 3D sobel map with the same shape as the input volume.
    """
    if len(volume.shape) != 3:
        raise ValueError("Input volume must be a 3D array")
    
    w, h, d = volume.shape
    sobel_map = np.zeros((w, h, d), dtype=np.float32)
    for i in range(d):
        slice_ = volume[:, :, i]
        # Normalize slice to 0-255 only if needed
        if slice_.max() != slice_.min():
            slice_ = (slice_ - slice_.min()) / (slice_.max() - slice_.min()) * 255
        sobel_x = sobel(slice_, axis=0, mode='constant')
        sobel_y = sobel(slice_, axis=1, mode='constant')
        # Compute the Sobel magnitude
        sobel_map[:, :, i] = np.hypot(sobel_x, sobel_y)

    # Normalize the Sobel map to 0-255
    sobel_map = (sobel_map - sobel_map.min()) / (sobel_map.max() - sobel_map.min()) * 255
    return sobel_map


def _get_radiomics(extractor, image, label):
    feats = {}
    featureVector = extractor.execute(image, label)
    for (key, val) in six.iteritems(featureVector):
        if key.startswith("original") or key.startswith("log") or key.startswith("wavelet") :
            feats[key] = val
    return feats


"""
def _nifti_volume_to_sitk(nifti_img, affine_in=None, clip=None):
    data = nifti_img.get_fdata()
    if clip:
        data = np.clip(data, clip[0], clip[1]).astype(np.float32)  # Apply clipping
    if affine_in is None:
        affine = nifti_img.affine
    else:
        affine = affine_in
    sitk_image = sitk.GetImageFromArray(data)
    # Extract metadata from affine
    spacing = np.abs(affine[:3, :3].diagonal())  # Extract voxel spacing
    origin = affine[:3, 3]  # Extract image origin
    direction = affine[:3, :3].flatten().tolist()  # Extract direction cosines
    # Apply metadata to SimpleITK image
    sitk_image.SetSpacing(tuple(spacing))
    sitk_image.SetOrigin(tuple(origin))
    sitk_image.SetDirection(direction)
    return sitk_image
"""


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

