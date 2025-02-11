import os
import numpy as np
from .data import read_csv
from alive_progress import alive_bar
from .data.utils import read_data
from .models.modules.bm4d import bm4d
from .helpers import helper_func


def estimate_psd_3d(volume):
    """
    Estimate the 3D Power Spectral Density (PSD) for a given CT volume.
    Assumes the noise follows a Gaussian distribution.
    Args:
        volume (numpy.ndarray): 3D NumPy array of shape (D, H, W)
    Returns:
        psd (numpy.ndarray): 3D Power Spectral Density (PSD) of shape (D, H, W)
    """
    # Subtract mean to normalize the noise
    volume = volume - np.mean(volume)
    # Compute the 3D Fourier Transform
    fft_volume = np.fft.fftn(volume)
    fft_shift = np.fft.fftshift(fft_volume)  # Shift zero frequency to center
    # Compute Power Spectral Density (PSD)
    psd = np.abs(fft_shift) ** 2
    psd /= np.prod(volume.shape)  # Normalize by the number of voxels
    return psd


def _main(config, logger, cur_sess_pth, model):
    current_mod = os.path.basename(os.path.dirname(__file__))
    mode = config[current_mod]['mode']
    models_param = config[current_mod]['param']
    datasets = config[current_mod]['input_datasets']
    """
    l_range = models_param.get('l_range', None)
    if isinstance(l_range, list):
        min_HU, max_HU = l_range
    else:
        min_HU, max_HU = None, None
    """

    for dataset in datasets:
        out_d = os.path.join(cur_sess_pth, current_mod, dataset['name'], mode)
        os.makedirs(out_d, exist_ok=True)
        dataset_opt = config['Datasets'][dataset['name']]
        if dataset.get('in_uids', None):
            dataset_opt['in_uids'] = dataset['in_uids']
        df = read_csv(dataset_opt['in_uids'])
        in_uids = df['uids'].to_list()
        in_uids = [in_uid.rstrip() for in_uid in in_uids]

        with alive_bar(len(in_uids), title='Processing files') as bar:
            for in_uid in in_uids:
                affine_in, header_in, in_data = read_data(in_uid, dataset_opt['in_dtype'])
                # Lung analysis window crop: modifying this would require to reset WindowCenter and WindowWidth
                in_data = np.clip(in_data, -1000., 500.).astype(np.int16) 
                if model['name'] == 'BM3D':
                    est_type = models_param.get('noise_type', 'std')
                    if est_type == 'std':
                        sigma = np.std(in_data)
                        in_data = bm4d(in_data, sigma)
                    elif est_type == 'psd':
                        psd = estimate_psd_3d(in_data)
                        in_data = bm4d(in_data, psd)
                    else:
                        raise ValueError("Noise type not recognized! Must be 'std' or 'psd'")

                    if models_param['out_dtype'] == 'dcm':
                        folder_name = '--'.join(in_uid.lstrip('/').split('/'))
                        helper_func.save_volume(in_data.astype(np.int16), out_type=models_param['out_dtype'], out_dir=os.path.join(out_d, folder_name), m_type='Volume',
                        f_name='{}--{}'.format(model['name'], folder_name), meta=header_in,
                        target_scale=None)

                    elif models_param['out_dtype'] == 'nii' or models_param['out_dtype'] == 'nii.gz':
                        raise NotImplementedError('.nii.gz not yet implemented!')
                    else:
                        raise ValueError('File out type not recognized!')
                bar()


