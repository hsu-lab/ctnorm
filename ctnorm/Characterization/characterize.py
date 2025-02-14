import os
import numpy as np
import pandas as pd
import pydicom
from scipy.stats import skew, kurtosis, gaussian_kde
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import pickle 
from ctnorm.Harmonization.data.utils import read_data


def data_char(input_file, input_dtype, out_dir, dataset_name, global_logger, params, metrics):
    """
    Characterizes DICOM datasets by computing voxel histograms and extracting metadata.
    Saves results as CSV and histogram files.
    """
    input_df = pd.read_csv(input_file)
    directory_paths = input_df['uids']
    
    clip_range = params.get('clip_range', [-1024, 3071])
    bins = params.get('bins', 64)
    kde_points = params.get('kde_points', None)
    voxel = metrics.get('voxel', {})
    metadata = metrics.get('metadata', {})

    aggregate_hist = None  # Stores cumulative histogram data
    metadata_stats = []  # Stores metadata and voxel statistics
    voxel_stats = []
    x_grid = np.linspace(clip_range[0], clip_range[1], 1000)
    avg_kde = np.zeros_like(x_grid)
    kde_count = 0

    for i, directory_path in enumerate(tqdm(directory_paths, desc="Processing DICOM directories")):
        directory_path = str(directory_path)  # Ensure it's a string
        # Check if directory exists
        if not os.path.exists(directory_path):
            global_logger.error(f"Directory not found: {directory_path}")
            continue

        _, header, data = read_data(directory_path, ext=input_dtype, apply_lut_for_dcm=True, need_affine_for_dcm=False)
        data = np.clip(data, clip_range[0], clip_range[1])

        if voxel:
            voxel_info = {'uids': directory_path}
            if 'histogram' in voxel or 'all' in voxel:
                hist, bin_edges = np.histogram(data.flatten(), bins=bins, range=(clip_range[0], clip_range[1]))
                if aggregate_hist is None:
                    aggregate_hist = np.zeros_like(hist, dtype=np.float64)
                aggregate_hist += hist  # Accumulate histograms
            if 'skewness' in voxel or 'all' in voxel:
                voxel_info['skewness'] = skew(data.flatten())
            if 'kurtosis' in voxel or 'all' in voxel:
                voxel_info['kurtosis'] = kurtosis(data.flatten())
            if 'snr' in voxel or 'all' in voxel:
                mean_signal = np.mean(data)
                std_noise = np.std(data)
                snr_value = 10 * np.log10((mean_signal ** 2) / (std_noise ** 2)) if std_noise > 0 else 0
                voxel_info['snr'] = snr_value
            if 'kde' in voxel or 'all' in voxel:
                # Subsample data to speed up KDE computation
                if kde_points:
                    sample_data = np.random.choice(data.flatten(), kde_points, replace=False)
                    kde = gaussian_kde(sample_data)
                else:
                    kde = gaussian_kde(data.flatten())
                kde_values = kde(x_grid)
                # Incrementally update the average KDE
                avg_kde = (avg_kde * kde_count + kde_values) / (kde_count + 1)
                kde_count += 1
            voxel_stats.append(voxel_info)

        if metadata:
            metadata_info = {
                'uids': directory_path
                }
            if 'all' in metadata or 'SliceThickness' in metadata:
                metadata_info['slice_thickness'] = getattr(header['meta_data'], 'SliceThickness', None)
            if 'all' in metadata or 'ConvolutionKernel' in metadata:
                metadata_info['convolution_kernel'] = getattr(header['meta_data'], 'ConvolutionKernel', None)
            if 'all' in metadata or 'PixelSpacing' in metadata:
                metadata_info['pixel_spacing'] = getattr(header['meta_data'], 'PixelSpacing', None)
            if 'all' in metadata or 'Manufacturer' in metadata:
                metadata_info['manufacturer'] = getattr(header['meta_data'], 'Manufacturer', None)
            metadata_stats.append(metadata_info)

    if ('histogram' in voxel or 'all' in voxel) and aggregate_hist is not None:
        histogram_path = os.path.join(out_dir, 'histogram.pkl')
        with open(histogram_path, "wb") as f:
            pickle.dump({'histogram':aggregate_hist, 'bin_edges':bin_edges}, f)
        global_logger.info(f"Histogram saved at {out_dir}/histogram.pkl")

    if ('kde' in voxel or 'all' in voxel) and kde_count > 0:
        kde_path = os.path.join(out_dir, 'kde.pkl')
        with open(kde_path, "wb") as f:
            pickle.dump(avg_kde, f)
        global_logger.info(f"KDE saved at {kde_path}")

    if voxel:
        voxel_stats_df = pd.DataFrame(voxel_stats)
        voxel_path = os.path.join(out_dir, 'voxel_characterization.csv')
        voxel_stats_df.to_csv(voxel_path, index=False)
        global_logger.info(f"Voxel characterization saved at {voxel_path}")

    if metadata:
        metadata_stats_df = pd.DataFrame(metadata_stats)
        metadata_path = os.path.join(out_dir, 'metadata_characterization.csv')
        metadata_stats_df.to_csv(metadata_path, index=False)
        global_logger.info(f"metadata characterization saved at {metadata_path}")