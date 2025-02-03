
import os
import numpy as np
import pandas as pd
import pydicom
from scipy.stats import skew, kurtosis
from tqdm import tqdm

def data_char(input_file, out_dir, dataset_name, global_logger, bins=64, voxel=1, metadata=1):
    """
    Characterizes DICOM datasets by computing voxel histograms and extracting metadata.
    Saves results as CSV and histogram files.
    """
    try:
        # Load dataset paths from CSV
        input_df = pd.read_csv(input_file)
        directory_paths = input_df['uids']
    except Exception as e:
        global_logger.error(f"Error reading CSV file '{input_file}': {e}")
        return
    
    aggregate_hist = None  # Stores cumulative histogram data
    folder_stats = []  # Stores metadata and voxel statistics

    for directory_path in tqdm(directory_paths, desc="Processing DICOM directories"):
        directory_path = str(directory_path)  # Ensure it's a string

        # Check if directory exists
        if not os.path.exists(directory_path):
            global_logger.error(f"Directory not found: {directory_path}")
            continue

        dicom_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.dcm')]
        if not dicom_files:
            global_logger.error(f"No DICOM files found in: {directory_path}")
            continue

        all_pixel_values = []
        slice_thickness, convolution_kernel, tube_current, pixel_spacing, manufacturer = [None] * 5

        for dicom_file in dicom_files:
            dicom_path = os.path.join(directory_path, dicom_file)
            try:
                dicom_image = pydicom.dcmread(dicom_path)

                # Process voxel data if enabled
                if voxel:
                    img = dicom_image.pixel_array
                    img = np.clip(img, -1024, 3071)  # Standardize intensity range
                    all_pixel_values.extend(img.flatten())

                    # Compute histogram
                    hist, _ = np.histogram(img.flatten(), bins=bins, range=(-1024, 3071))
                    if aggregate_hist is None:
                        aggregate_hist = np.zeros_like(hist, dtype=np.float64)
                    aggregate_hist += hist  # Accumulate histograms

                # Extract metadata (only from the first file in the directory)
                if metadata and (slice_thickness is None):
                    slice_thickness = dicom_image.get('SliceThickness')
                    convolution_kernel = dicom_image.get('ConvolutionKernel')
                    tube_current = dicom_image.get('XRayTubeCurrent')
                    pixel_spacing = dicom_image.get('PixelSpacing')
                    manufacturer = dicom_image.get('Manufacturer')

            except Exception as e:
                global_logger.error(f"Error reading DICOM file '{dicom_file}': {e}")
                continue

        # Compute voxel statistics
        if voxel and all_pixel_values:
            mean_value = np.mean(all_pixel_values)
            median_value = np.median(all_pixel_values)
            skewness_value = skew(all_pixel_values)
            kurtosis_value = kurtosis(all_pixel_values)
            min_value = np.min(all_pixel_values)
            max_value = np.max(all_pixel_values)
        else:
            mean_value = median_value = skewness_value = kurtosis_value = min_value = max_value = None

        folder_stats.append({
            'uids': directory_path,
            'mean': mean_value,
            'median': median_value,
            'skewness': skewness_value,
            'kurtosis': kurtosis_value,
            'min': min_value,
            'max': max_value,
            'slice_thickness': slice_thickness,
            'convolution_kernel': convolution_kernel,
            'tube_current': tube_current,
            'pixel_spacing': pixel_spacing,
            'manufacturer': manufacturer
        })

    if voxel and aggregate_hist is not None:
        np.save(os.path.join(out_dir, 'histogram.npy'), aggregate_hist)
        global_logger.info(f"Histogram saved at {out_dir}/histogram.npy")

    data_stats = pd.DataFrame(folder_stats)
    csv_path = os.path.join(out_dir, 'data_characterization.csv')
    data_stats.to_csv(csv_path, index=False)
    global_logger.info(f"Data characterization saved at {csv_path}")