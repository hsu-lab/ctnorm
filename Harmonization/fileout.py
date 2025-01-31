import os
import pydicom
import nibabel as nib
import pickle
import numpy as np


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
    # print(f"DICOM file saved at {output_path}")


def save_volume(data, out_type, out_dir, m_type, f_name, target_scale=None):
    if out_type == 'nii.gz':
        raise NotImplementedError('.nii.gz extension not yet supported!')
    elif out_type == 'dcm':
        output_folder = os.path.join(out_dir, m_type, f_name)
        os.makedirs(output_folder, exist_ok=True)
        f_metadata = os.path.join(out_dir, 'metadata.pkl')
        with open(f_metadata, 'rb') as f:
            loaded_metadata = pickle.load(f)
            dcm_metadata = loaded_metadata['meta_data']

        z_start = loaded_metadata['z_start']
        z_sign = loaded_metadata['z_sign']
        if target_scale is None:
            z_positions = [z_start + z_sign * i * dcm_metadata.SliceThickness for i in range(data.shape[2])]
            # z_positions = [loaded_metadata['z_sign'] * (abs(loaded_metadata['z_start']) + i * 1.0) for i in range(data.shape[0])]
            out_thickness = float(dcm_metadata.SliceThickness)
        else:
            z_positions = [z_start + z_sign * i * 1.0 for i in range(data.shape[2])]
            # z_positions = [loaded_metadata['z_sign'] * (abs(loaded_metadata['z_start']) + i * float(dcm_metadata.SliceThickness)) for i in range(data.shape[0])]
            out_thickness = 1.0
        
        for i in range(data.shape[2]):
            slice_path = os.path.join(output_folder, f"slice_{i + 1:03d}.dcm")
            create_minimum_dicom_header(data[:,:,i], i + 1, dcm_metadata, slice_path, z_positions[i], thickness=out_thickness)
    else:
        raise NotImplementedError('{} extension not yet supported!'.format(out_type))