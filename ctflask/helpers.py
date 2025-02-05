import os
import nibabel as nib


# Function to read a NIfTI file and extract the relevant slice
def process_nifti_file(case):
    case_nii = nib.load(case)
    sobel_volume = case_nii.get_fdata()
    # Select the 10 center slices
    num_slices = sobel_volume.shape[2]
    center_slice = num_slices // 2
    start_slice = max(center_slice - 5, 0)
    end_slice = min(center_slice + 5, num_slices)
    sobel_volume = sobel_volume[:, :, start_slice:end_slice]
    return (start_slice, end_slice, sobel_volume)

