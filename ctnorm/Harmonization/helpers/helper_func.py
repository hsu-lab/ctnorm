import os
import pydicom
import nibabel as nib
import pickle
import numpy as np
import torch
import random
from .image_reorientation import reorient_image
import SimpleITK as sitk
from radiomics import featureextractor
import six
from skimage import filters
"""
# Set logging for radimioc rather than printing on screen
logger = radiomics.logging.getLogger("radiomics")
logger.setLevel(radiomics.logging.ERROR)
"""


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
        
        data = np.transpose(data, (2,1,0)) # REQUIRED FOR RAS ALIGNMENT IN ITK
        output_folder = os.path.join(out_dir, m_type)
        os.makedirs(output_folder, exist_ok=True)
        out_f = os.path.join(output_folder, f_name+out_type)
        nii_to_save = nib.Nifti1Image(data, affine=affine_in)
        # BY DEFAULT: WE REORIENT IMAGE AS DONE IN dicom2nifti LIBRARY
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


def save_metric(cases, metric):
    if metric.lower() == 'radiomic':
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
            in_case = sitk.ReadImage(in_cases[case_idx])
            in_mask = sitk.ReadImage(in_masks[case_idx])
            feats = _get_radiomics(extractor, in_case, in_mask)
            for feat in feats:
                if feat.startswith('original'):
                    if feat not in df_array:
                        df_array[feat] = []
                    df_array[feat].append(feats[feat].item())
            df_array['id'].append(in_case)
        df = pd.DataFrame(df_array)
        return df


    # for metric in metrics_to_c:
    #     if metric.lower() == 'sobel':
    #         continue

    #     elif metric.lower() == 'emphysema':
    #         continue

    #     elif metric.lower() == 'radiomic':
    #         if ext_utils is None:
    #             raise ValueError('ROI mask must be specified to extract radiomic features!')

    #         # Only define once
    #         if "extractor" not in globals():
    #             global extractor
    #             paramPath = './CT.yaml'
    #             extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)

    #         # NOTE: This logic can be modified if output volume and mask has been validated to have same orientation
    #         # even though the affine doesn't match
    #         mask_f = nib.load(ext_utils['mask_root'][0]) # Assumes mask is in RAS
    #         if _diagonal_matches(affine_in, mask_f.affine):
    #             # Assumes orientation has been checked and OK!
    #             nii_img = nib.Nifti1Image(data, affine=affine_in)
    #             del data
    #             pass
    #         else:
    #             # Reorients image to RAS; assuming masks is already in RAS
    #             # NOTE: This logic can be skipped if orientation has been verified to match
    #             data = np.transpose(data, (2,1,0))
    #             nii_img = nib.Nifti1Image(data, affine=affine_in)
    #             del data
    #             nii_img = reorient_image(nii_img)

    #         # Extract feature
    #         image_sitk = _nifti_volume_to_sitk(nii_img, None, [-1000., 500.])
    #         mask_sitk = _nifti_volume_to_sitk(mask_f, nii_img.affine, None)
    #         feats = _get_radiomics(extractor, image_sitk, mask_sitk)
    #         print('Features extracted!')
    #         output_folder = os.path.join(out_dir, metric)
    #         os.makedirs(output_folder, exist_ok=True)
    #     else:
    #         continue


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

