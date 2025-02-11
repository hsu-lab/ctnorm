"""
BM4D denoising of volumetric (& volumetric multichannel) data: (package: "bm4d")

Based on:

Y. Mäkinen, L. Azzari, A. Foi, 2020,
"Collaborative Filtering of Correlated Noise: Exact Transform-Domain Variance for Improved Shrinkage and Patch Matching",
in IEEE Transactions on Image Processing, vol. 29, pp. 8339-8354, and

Y. Mäkinen, S. Marchesini, A. Foi, 2021,
"Ring Artifact and Poisson Noise Attenuation via Volumetric Multiscale Nonlocal Collaborative Filtering of Spatially Correlated Noise", submitted to Journal of Synchrotron Radiation

"""

import copy
import os
from typing import List, Union, Tuple

from scipy.fftpack import *
from scipy.linalg import *
from scipy.ndimage import correlate
from scipy import signal
from scipy.io import loadmat
from scipy.interpolate import interpn
import numpy as np


# PyWavelets is not essential, as we include a few hard-coded transforms
try:
    import pywt
except ImportError:
    pywt = None

from .bm4d_ctypes import bm4d_wie_complex as _bm4d_wie_complex
from .bm4d_ctypes import bm4d_wie as _bm4d_wie
from .bm4d_ctypes import bm4d_ht as _bm4d_ht
from .bm4d_ctypes import bm4d_ht_complex as _bm4d_ht_complex
from .bm4d_ctypes import BlockMatchStorage

from .profiles import BM4DProfile, BM4DProfileRefilter, BM4DProfile2D, BM4DProfile2DRefilter
from .profiles import BM4DProfileBM3D, BM4DProfileBM3DComplex, BM4DProfileComplex, BM4DStages

EPS = 2.2204e-16

def bm4d_multichannel(z: Union[List[np.ndarray], np.ndarray], sigma_psd: Union[np.ndarray, float, list],
                      profile: Union[BM4DProfile, str] = 'np',
                      stage_arg: Union[BM4DStages, np.ndarray] = BM4DStages.ALL_STAGES):
    """
       Perform multichannel BM4D denoising on z: either hard-thresholding, Wiener filtering or both.
       Block-matching will be performed only on the first channel.

       :param z: multichannel 3-D Noisy image as either a 4-D array (channel as axis 0) or a list of 3-D arrays
       :param sigma_psd: Noise PSD same size of z, or
              sigma_psd: Noise standard deviation (float)
              Supply either a single element/3-D array (copy for each channel), or a list / 4-D array similar to z
       :param profile: Settings for BM4D: BM4DProfile object or a string. Default 'np'.
                       'lc' for Low Complexity, 'refilter' for refiltering.
                       'np' and 'refilter' choose block dimensionality based on image size (2-D or 3-D)
                       '8x8' forces 8x8x1 block size
       :param stage_arg: Determines whether to perform hard-thresholding or wiener filtering.
                       either BM4DStages.HARD_THRESHOLDING, BM4DStages.ALL_STAGES or an estimate
                       of the noise-free image.
                       - BM4DStages.ALL_STAGES: Perform both.
                       - BM4DStages.HARD_THRESHOLDING: Perform hard-thresholding only.
                       - ndarray, size of z: Perform Wiener Filtering with stage_arg as pilot.
       :return:
           - denoised array, same format as z
       """

    full_denoi = []

    multiple_sigma = isinstance(sigma_psd, list) or (isinstance(sigma_psd, np.ndarray) and sigma_psd.ndim == 4 and sigma_psd.shape[0] > 1)

    def get_sigma(index: int):
        if not multiple_sigma:
            return np.squeeze(sigma_psd)
        return sigma_psd[index]


    denoi, match_arrs = bm4d(z[0], get_sigma(0), profile, stage_arg, (True, True))
    full_denoi.append(denoi)

    for i in range(1, len(z)):
        denoi = bm4d(z[i], get_sigma(i), profile, stage_arg, match_arrs)
        full_denoi.append(denoi)

    if isinstance(z, list):
        return full_denoi
    return np.array(full_denoi)


def bm4d(z: np.ndarray, sigma_psd: Union[np.ndarray, float],
         profile: Union[BM4DProfile, str] = 'np',
         stage_arg: Union[BM4DStages, np.ndarray] = BM4DStages.ALL_STAGES,
         blockmatches: Tuple[Union[BlockMatchStorage, bool], Union[BlockMatchStorage, bool]] = (False, False)) \
        -> Union[np.ndarray, Tuple[np.ndarray, Tuple[Union[BlockMatchStorage, bool], Union[BlockMatchStorage, bool]]]]:
    """
    Perform BM4D denoising on z: either hard-thresholding, Wiener filtering or both.

    :param z: 3-D Noisy image. 2-D images will cast to 3-D
    :param sigma_psd: Noise PSD same size of z, or
           sigma_psd: Noise standard deviation (float)
    :param profile: Settings for BM4D: BM4DProfile object or a string. Default 'np'.
                    'lc' for Low Complexity, 'refilter' for refiltering.
                    'np' and 'refilter' choose block dimensionality based on image size (2-D or 3-D)
                    '8x8' forces 8x8x1 block size
    :param stage_arg: Determines whether to perform hard-thresholding or wiener filtering.
                    either BM4DStages.HARD_THRESHOLDING, BM4DStages.ALL_STAGES or an estimate
                    of the noise-free image.
                    - BM4DStages.ALL_STAGES: Perform both.
                    - BM4DStages.HARD_THRESHOLDING: Perform hard-thresholding only.
                    - ndarray, size of z: Perform Wiener Filtering with stage_arg as pilot.
    :param blockmatches: Use same blockmatches for multiple inputs: A tuple for (hard thresholding, wiener)
                    with either Bool:
                        False -> Do not use or collect repeat blockmatches (default)
                        True -> Save blockmatches, return (y_hat, (blockmatches_ht, blockmatches_wie))
                    or BlockMatchStorage object (profiles.py) that was returned by a previous application of BM4D.
    :return:
        - denoised image, same size as z
    """
    # Ensure z is 3-D a numpy array
    z = np.array(z)
    if z.ndim == 1:
        raise ValueError("z must be either a 2D or a 3D image!")
    if z.ndim == 2:
        z = np.atleast_3d(z)

    # Profile selection, if profile is a string, otherwise BM4DProfile.
    pro = _select_profile(profile, z)

    # If profile defines maximum required pad, use that, otherwise use image size
    pad_size = (int(np.ceil(z.shape[0] / 2)), int(np.ceil(z.shape[1] / 2)), int(np.ceil(z.shape[2] / 2))) \
        if pro.max_pad_size is None else pro.max_pad_size

    if z.shape[2] == 1:
        pad_size = (pad_size[0], pad_size[1], 0)

    # Conventional mode
    if pro.nf == 0 or pro.nf[0] == 0 or pro.nf[1] == 0 or pro.nf[2] == 0:
        pro.nf = (np.minimum(z.shape[0], 16), np.minimum(z.shape[1], 16), np.minimum(z.shape[2], 16))
        pro.k = 0
        pro.gamma = 0

    y_hat = None

    # If we passed a numpy array as stage_arg, presume it is a hard-thresholding estimate.
    if isinstance(stage_arg, np.ndarray):
        y_hat = np.atleast_3d(stage_arg)
        stage_arg = BM4DStages.WIENER_FILTERING
        if y_hat.shape != z.shape:
            raise ValueError("Estimate passed in stage_arg must be equal size to z!")

    elif stage_arg == BM4DStages.WIENER_FILTERING:
        raise ValueError("If you wish to only perform Wiener filtering, you need to pass an estimate as stage_arg!")

    if z.shape[0] < pro.bs_ht[0] or z.shape[1] < pro.bs_ht[1] or z.shape[2] < pro.bs_ht[2] or \
            z.shape[0] < pro.bs_wiener[0] or z.shape[1] < pro.bs_wiener[1] or z.shape[2] < pro.bs_wiener[2]:
        raise ValueError("Image cannot be smaller than block size!")

    # If this is true, we are doing hard thresholding (whether we do Wiener later or not)
    stage_ht = (stage_arg.value & BM4DStages.HARD_THRESHOLDING.value) != 0
    # If this is true, we are doing Wiener filtering
    stage_wie = (stage_arg.value & BM4DStages.WIENER_FILTERING.value) != 0

    sigma_psd = np.array(sigma_psd)
    single_d = False

    # Format single dimension (std) sigma_psds
    if np.squeeze(sigma_psd).ndim <= 1:
        sigma_psd = np.ones(z.shape) * z.size * sigma_psd ** 2
        single_d = True

    sigma_psd = np.atleast_3d(sigma_psd)

    # Process PSD to be resizable to N_f
    sigma_psd2, psd_blur, psd_k = _process_psd(sigma_psd, z, single_d, pad_size, pro)

    bm_out_ht = None
    bm_out_wie = None

    inp_complex = z.dtype == np.complex64 or z.dtype == np.complex128

    if inp_complex:
        raise ValueError("Complex input not supported!")

    if pro.transform_local_ht_name == 'fft' or pro.transform_local_wiener_name == 'fft' \
        or pro.transform_nonlocal_name == 'fft':
        z = np.array(z, dtype=np.complex64)

    ht_fn = _bm4d_ht_complex if inp_complex else _bm4d_ht
    wie_fn = _bm4d_wie_complex if inp_complex else _bm4d_wie

    try:
        bm_in_ht = blockmatches[0]
        bm_in_wie = blockmatches[1]
    except (IndexError, TypeError):
        raise ValueError("Block matches should contain a value for both stages (use False to ignore)")

    if pro.print_info:
        if pro.sharpen_alpha != 1 or pro.sharpen_alpha_3d != 1:
            print_line = "Performing sharpening with alpha = {:.2f}".format(pro.sharpen_alpha)
            if pro.sharpen_alpha != pro.sharpen_alpha_3d:
                print_line += ", 3D-DC alpha = {:.2f}".format(pro.sharpen_alpha_3d)
            print(print_line)

    # Step 1. Produce the basic estimate by HT filtering
    if stage_ht:

        if pro.print_info:
            print("Hard-thresholding with lambda = {:.2f}".format(pro.lambda_thr))
        qshifts = get_shift_params(pro.bs_ht, pro.step_ht)
        # Get used transforms and aggregation windows.
        t_forward, t_inverse, hadper_trans_single_den, \
            inverse_hadper_trans_single_den, wwin3d = _get_transforms(pro, True)

        # Call the actual hard-thresholding step with the acquired parameters
        y_hat, bm_out_ht = ht_fn(z, psd_blur, pro, t_forward, t_inverse, qshifts, hadper_trans_single_den,
                                inverse_hadper_trans_single_den, wwin3d, blockmatches=bm_in_ht)

        # Residual denoising, HT
        if pro.denoise_residual:

            if pro.print_info:
                print("Hard-thresholding residual with lambda = {:.2f}".format(pro.lambda_thr_re))

            remains, remains_psd = get_filtered_residual(z, y_hat, sigma_psd2, pad_size, pro.residual_thr)
            remains_psd = _process_psd_for_nf(remains_psd, psd_k, pro)

            if np.min(np.max(np.max(remains_psd, axis=0), axis=0)) > 1e-5:
                # Re-filter
                y_hat, bm_out_ht = ht_fn(y_hat + remains, remains_psd, pro, t_forward, t_inverse, qshifts, hadper_trans_single_den,
                                        inverse_hadper_trans_single_den, wwin3d, True, blockmatches=bm_in_ht)

        if pro.print_info:
            print('Hard-thresholding stage completed')

    # Error (probably OOM) occured, do not process further
    if np.mean(y_hat) == np.min(y_hat):
        return y_hat

    # Step 2. Produce the final estimate by Wiener filtering (using the
    # hard-thresholding initial estimate)

    if stage_wie:

        if pro.print_info:
            print("Wiener filtering with mu^2 = {:.2f}".format(pro.mu2))

        qshifts = get_shift_params(pro.bs_wiener, pro.step_wiener)
        # Get used transforms and aggregation windows.
        t_forward, t_inverse, hadper_trans_single_den, \
            inverse_hadper_trans_single_den, wwin3d = _get_transforms(pro, False)

        # Wiener filtering
        y_hat, bm_out_wie = wie_fn(z, psd_blur, pro, t_forward, t_inverse, qshifts, hadper_trans_single_den,
                                    inverse_hadper_trans_single_den, wwin3d, y_hat, blockmatches=bm_in_wie)

        # Residual denoising, Wiener
        if pro.denoise_residual:

            if pro.print_info:
                print("Wiener filtering residual with mu^2 = {:.2f}".format(pro.mu2_re))

            remains, remains_psd = get_filtered_residual(z, y_hat, sigma_psd2, pad_size, pro.residual_thr)
            remains_psd = _process_psd_for_nf(remains_psd, psd_k, pro)

            if np.min(np.max(np.max(remains_psd, axis=0), axis=0)) > 1e-5:
                y_hat, bm_out_wie = wie_fn(y_hat + remains, remains_psd, pro, t_forward, t_inverse, qshifts, hadper_trans_single_den,
                                        inverse_hadper_trans_single_den, wwin3d, y_hat, True, blockmatches=bm_in_wie)

        if pro.print_info:
            print('Wiener-filtering stage completed')

    if not stage_ht and not stage_wie:
        raise ValueError("No operation was selected!")

    if not inp_complex:
        y_hat = np.real(y_hat)

    if isinstance(bm_out_ht, BlockMatchStorage) or isinstance(bm_out_wie, BlockMatchStorage):
        return y_hat, (bm_out_ht, bm_out_wie)

    return y_hat

def get_filtered_residual(z: np.ndarray, y_hat: np.ndarray, sigma_psd: Union[np.ndarray, float],
                          pad_size: Union[list, tuple], residual_thr: float) -> (np.ndarray, np.ndarray):
    """
    Get residual, filtered by global FFT HT
    :param z: Original noisy image (MxNxC)
    :param y_hat: Estimate of noise-free image, same size as z
    :param sigma_psd: std, 1-D list of stds or MxNx1 or MxNxC "list" of PSDs.
            Note! if PSD, the size must be size of z + 2 * pad_size, not size of z!
    :param pad_size: amount to pad around z and y_hat to avoid problems due to non-circular noise.
                     Should be at least kernel size in total (1/2 on one side), but may be bigger if kernel size
                     is unknown.
    :param residual_thr: The threshold to use in the global Fourier filter.
    :return: (filtered residual, same size as z, PSD of the filtered residual, same size as z)

    """

    # Calculate the residual
    if pad_size[0]:
        pads_width = ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2]))
        resid = fftn(np.pad(z - y_hat, pads_width, 'constant'), axes=(0, 1, 2))
    else:
        resid = fftn(z - y_hat, axes=(0, 1, 2))

    # Kernel size for dilation
    ksz = [np.ceil(resid.shape[0] / 150), np.ceil(resid.shape[1] / 150), np.ceil(resid.shape[2] / 150)]
    ksz = [ksz[0] + 1 - (ksz[0] % 2), ksz[1] + 1 - (ksz[1] % 2), ksz[2] + 1 - (ksz[2] % 2)]

    psd_size_div = z.size
    psd = sigma_psd

    # Apply dilation filter
    kernel = np.atleast_3d(gaussian_kernel_3d(ksz, resid.shape[0] / 500, resid.shape[1] / 500, resid.shape[2] / 500))
    cc = correlate(np.array(np.abs(resid) > (residual_thr * np.sqrt(psd)), dtype=float), kernel, mode='wrap')

    # Threshold mask
    msk = (cc > 0.01)

    # Residual + PSD
    remains = np.real(ifftn(resid * msk, axes=(0, 1, 2)))
    remains_psd = psd * msk

    # Cut off padding
    pad_cutoff = (-1 if pad_size[0] == 0 else pad_size[0],
                  -1 if pad_size[1] == 0 else pad_size[1],
                  -1 if pad_size[2] == 0 else pad_size[2])
    remains = remains[pad_size[0]:-pad_cutoff[0], pad_size[1]:-pad_cutoff[1], pad_size[2]:-pad_cutoff[2]]

    temp_kernel = np.real(fftshift(ifftn(np.sqrt(remains_psd / psd_size_div), axes=(0, 1, 2)), axes=(0, 1, 2)))
    temp_kernel = temp_kernel[pad_size[0]:-pad_cutoff[0], pad_size[1]:-pad_cutoff[1], pad_size[2]:-pad_cutoff[2]]

    remains_psd = np.power(abs(fftn(temp_kernel, z.shape, axes=(0, 1, 2))), 2) * z.size

    return remains, remains_psd


def gaussian_kernel(size: Union[tuple, list], std: float, std2: float = -1) -> np.ndarray:
    """
    Get a 2D Gaussian kernel with the specified standard deviations.
    If std2 is not specified, both stds will be the same.
    :param size: kernel size, tuple
    :param std: std of 1st dimension
    :param std2: std of 2nd dimension, or -1 if equal to std
    :return: normalized Gaussian kernel (sum == 1)
    """
    if std2 == -1:
        std2 = std
    g1d = signal.windows.gaussian(int(size[0]), std=std).reshape(int(size[0]), 1)
    g1d2 = signal.windows.gaussian(int(size[1]), std=std2).reshape(int(size[1]), 1)

    g2d = np.outer(g1d / np.sum(g1d), g1d2 / np.sum(g1d2))
    return g2d


def gaussian_kernel_3d(size: Union[tuple, list], std: float, std2: float = -1, std3: float = -1) -> np.ndarray:
    """
    Get a 2D Gaussian kernel with the specified standard deviations.
    If std2 or std3 are not defined, they will be equal to std.
    :param size: kernel size, tuple
    :param std: std of 1st dimension
    :param std2: std of 2nd dimension, or -1 if equal to std
    :param std3: std of 3rd dimension, or -1 if equal to std
    :return: normalized Gaussian kernel (sum == 1)
    """

    if std2 == -1:
        std2 = std

    if std3 == -1:
        std3 = std

    g1d = signal.windows.gaussian(int(size[0]), std=std).reshape(int(size[0]), 1)
    g1d2 = signal.windows.gaussian(int(size[1]), std=std2).reshape(int(size[1]), 1)
    g1d3 = signal.windows.gaussian(int(size[2]), std=std3).reshape((int(size[2])))

    g2d = np.repeat(np.atleast_3d(np.outer(g1d / np.sum(g1d), g1d2 / np.sum(g1d2))), size[2], axis=2) * g1d3
    return g2d


def _process_psd_for_nf(sigma_psd: np.ndarray, psd_k: Union[np.ndarray, None], profile: BM4DProfile) \
        -> np.ndarray:
    """
    Process PSD so that Nf-size PSD is usable.
    :param sigma_psd: the PSD
    :param psd_k: a previously generated kernel to convolve the PSD with, or None if not used
    :param profile: the profile used
    :return: processed PSD
    """
    if profile.nf == 0:
        return sigma_psd

    # Reduce PSD size to start with
    max_ratio = 16
    sigma_psd_copy = np.copy(sigma_psd)
    single_kernels = [np.ones((3, 1, 1)) / 3, np.ones((1, 3, 1)) / 3, np.ones((1, 1, 3)) / 3]

    for i in range(0, 3):
        orig_ratio = sigma_psd.shape[i] / profile.nf[i]
        ratio = orig_ratio
        while ratio > max_ratio:
            mid_corr = correlate(sigma_psd_copy, single_kernels[i], mode='wrap')
            if i == 0:
                sigma_psd_copy = mid_corr[1::3, :, :]
            elif i == 1:
                sigma_psd_copy = mid_corr[:, 1::3, :]
            else:
                sigma_psd_copy = mid_corr[:, :, 1::3]

            ratio = sigma_psd_copy.shape[i] / profile.nf[i]

        # Scale PSD because the binary expects it to be scaled by size
        sigma_psd_copy *= (ratio / orig_ratio)

    if psd_k is not None:
        sigma_psd_copy = correlate(sigma_psd_copy, psd_k, mode='wrap')

    return sigma_psd_copy


def _select_profile(profile: Union[str, BM4DProfile], z: np.ndarray) -> BM4DProfile:
    """
    Select profile for BM4D
    :param profile: BM4DProfile or a string
    :return: BM4DProfile object
    """
    if isinstance(profile, BM4DProfile):
        return copy.copy(profile)

    # The default profile for shapes which can contain a 5-pixel wide block
    elif z.shape[2] >= 5:
        if profile == 'np':
            return BM4DProfile()
        elif profile == '8x8':
            return BM4DProfile2D()
        elif profile == '8x8refilter':
            return BM4DProfile2DRefilter()
        elif profile == 'refilter':
            return BM4DProfileRefilter()
    # Smaller things use 8x8x1 blocks (profile identical to BM3D)
    else:
        if profile == 'np':
            return BM4DProfile2D()
        elif profile == 'refilter':
            return BM4DProfile2DRefilter()

    raise TypeError('"profile" should be either a string of '
                    '"np"/"refilter" or a BM4DProfile object!')


def _get_transf_matrix(n: int, transform_type: str,
                       dec_levels: int = 0, flip_hardcoded: bool = False) -> (np.ndarray, np.ndarray):
    """
    Create forward and inverse transform matrices, which allow for perfect
    reconstruction. The forward transform matrix is normalized so that the
    l2-norm of each basis element is 1.
    Includes hardcoded transform matrices which are kept for matlab compatibility

    :param n: Transform size (nxn)
    :param transform_type: Transform type 'dct', 'dst', 'hadamard', or anything that is
                           supported by 'wavedec'
                           'DCrand' -- an orthonormal transform with a DC and all
                           the other basis elements of random nature
    :param dec_levels:  If a wavelet transform is generated, this is the
                           desired decomposition level. Must be in the
                           range [0, log2(N)-1], where "0" implies
                           full decomposition.
    :param flip_hardcoded: Return transpose of the hardcoded matrices.
    :return: (forward transform, inverse transform)
    """

    if n == 1:
        t_forward = 1
    elif transform_type == 'hadamard':
        t_forward = hadamard(n)
    elif n == 4 and transform_type == 'bior1.5':
        t_forward = [[0.50000000, 0.50000000, 0.50000000, 0.50000000],
                     [0.50000000, 0.50000000, -0.50000000, -0.50000000],
                     [0.70710677, -0.70710677, 0, 0],
                     [0, 0, 0.70710677, -0.70710677]]
    elif n == 8 and transform_type == 'bior1.5':
        t_forward = [[0.343550200747110, 0.343550200747110, 0.343550200747110, 0.343550200747110, 0.343550200747110,
                      0.343550200747110, 0.343550200747110, 0.343550200747110],
                     [-0.225454819240296, -0.461645582253923, -0.461645582253923, -0.225454819240296, 0.225454819240296,
                      0.461645582253923, 0.461645582253923, 0.225454819240296],
                     [0.569359398342840, 0.402347308162280, -0.402347308162280, -0.569359398342840, -0.083506045090280,
                      0.083506045090280, -0.083506045090280, 0.083506045090280],
                     [-0.083506045090280, 0.083506045090280, -0.083506045090280, 0.083506045090280, 0.569359398342840,
                      0.402347308162280, -0.402347308162280, -0.569359398342840],
                     [0.707106781186550, -0.707106781186550, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0.707106781186550, -0.707106781186550, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0.707106781186550, -0.707106781186550, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0.707106781186550, -0.707106781186550]]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif n == 8 and transform_type == 'dct':
        t_forward = [[0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274,
                      0.353553390593274, 0.353553390593274, 0.353553390593274],
                     [0.490392640201615, 0.415734806151273, 0.277785116509801, 0.097545161008064, -0.097545161008064,
                      -0.277785116509801, -0.415734806151273, -0.490392640201615],
                     [0.461939766255643, 0.191341716182545, -0.191341716182545, -0.461939766255643, -0.461939766255643,
                      -0.191341716182545, 0.191341716182545, 0.461939766255643],
                     [0.415734806151273, -0.097545161008064, -0.490392640201615, -0.277785116509801, 0.277785116509801,
                      0.490392640201615, 0.097545161008064, -0.415734806151273],
                     [0.353553390593274, -0.353553390593274, -0.353553390593274, 0.353553390593274, 0.353553390593274,
                      -0.353553390593274, -0.353553390593274, 0.353553390593274],
                     [0.277785116509801, -0.490392640201615, 0.097545161008064, 0.415734806151273, -0.415734806151273,
                      -0.097545161008064, 0.490392640201615, -0.277785116509801],
                     [0.191341716182545, -0.461939766255643, 0.461939766255643, -0.191341716182545, -0.191341716182545,
                      0.461939766255643, -0.461939766255643, 0.191341716182545],
                     [0.097545161008064, -0.277785116509801, 0.415734806151273, -0.490392640201615, 0.490392640201615,
                      -0.415734806151273, 0.277785116509801, -0.097545161008064]]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif n == 11 and transform_type == 'dct':
        t_forward = [[0.301511344577764, 0.301511344577764, 0.301511344577764, 0.301511344577764, 0.301511344577764,
                      0.301511344577764, 0.301511344577764, 0.301511344577764, 0.301511344577764, 0.301511344577764,
                      0.301511344577764],
                     [0.422061280946316, 0.387868386059133, 0.322252701275551, 0.230530019145232, 0.120131165878581,
                      -8.91292406723889e-18, -0.120131165878581, -0.230530019145232, -0.322252701275551,
                      -0.387868386059133, -0.422061280946316],
                     [0.409129178625571, 0.279233555180591, 0.0606832509357945, -0.177133556713755, -0.358711711672592,
                      -0.426401432711221, -0.358711711672592, -0.177133556713755, 0.0606832509357945, 0.279233555180591,
                      0.409129178625571],
                     [0.387868386059133, 0.120131165878581, -0.230530019145232, -0.422061280946316, -0.322252701275551,
                      1.71076608154014e-17, 0.322252701275551, 0.422061280946316, 0.230530019145232, -0.120131165878581,
                      -0.387868386059133],
                     [0.358711711672592, -0.0606832509357945, -0.409129178625571, -0.279233555180591, 0.177133556713755,
                      0.426401432711221, 0.177133556713755, -0.279233555180591, -0.409129178625571, -0.0606832509357945,
                      0.358711711672592],
                     [0.322252701275551, -0.230530019145232, -0.387868386059133, 0.120131165878581, 0.422061280946316,
                      -8.13580150049806e-17, -0.422061280946316, -0.120131165878581, 0.387868386059133,
                      0.230530019145232, -0.322252701275551],
                     [0.279233555180591, -0.358711711672592, -0.177133556713755, 0.409129178625571, 0.0606832509357945,
                      -0.426401432711221, 0.0606832509357944, 0.409129178625571, -0.177133556713755, -0.358711711672592,
                      0.279233555180591],
                     [0.230530019145232, -0.422061280946316, 0.120131165878581, 0.322252701275551, -0.387868386059133,
                      -2.87274927630557e-18, 0.387868386059133, -0.322252701275551, -0.120131165878581,
                      0.422061280946316, -0.230530019145232],
                     [0.177133556713755, -0.409129178625571, 0.358711711672592, -0.0606832509357945, -0.279233555180591,
                      0.426401432711221, -0.279233555180591, -0.0606832509357944, 0.358711711672592, -0.409129178625571,
                      0.177133556713755],
                     [0.120131165878581, -0.322252701275551, 0.422061280946316, -0.387868386059133, 0.230530019145232,
                      2.03395037512452e-17, -0.230530019145232, 0.387868386059133, -0.422061280946316,
                      0.322252701275551,
                      -0.120131165878581],
                     [0.0606832509357945, -0.177133556713755, 0.279233555180591, -0.358711711672592, 0.409129178625571,
                      -0.426401432711221, 0.409129178625571, -0.358711711672592, 0.279233555180591, -0.177133556713755,
                      0.0606832509357945]]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif n == 8 and transform_type == 'dst':
        t_forward = [[0.161229841765317, 0.303012985114696, 0.408248290463863, 0.464242826880013, 0.464242826880013,
                      0.408248290463863, 0.303012985114696, 0.161229841765317],
                     [0.303012985114696, 0.464242826880013, 0.408248290463863, 0.161229841765317, -0.161229841765317,
                      -0.408248290463863, -0.464242826880013, -0.303012985114696],
                     [0.408248290463863, 0.408248290463863, 0, -0.408248290463863, -0.408248290463863, 0,
                      0.408248290463863, 0.408248290463863],
                     [0.464242826880013, 0.161229841765317, -0.408248290463863, -0.303012985114696, 0.303012985114696,
                      0.408248290463863, -0.161229841765317, -0.464242826880013],
                     [0.464242826880013, -0.161229841765317, -0.408248290463863, 0.303012985114696, 0.303012985114696,
                      -0.408248290463863, -0.161229841765317, 0.464242826880013],
                     [0.408248290463863, -0.408248290463863, 0, 0.408248290463863, -0.408248290463863, 0,
                      0.408248290463863, -0.408248290463863],
                     [0.303012985114696, -0.464242826880013, 0.408248290463863, -0.161229841765317, -0.161229841765317,
                      0.408248290463863, -0.464242826880013, 0.303012985114696],
                     [0.161229841765317, -0.303012985114696, 0.408248290463863, -0.464242826880013, 0.464242826880013,
                      -0.408248290463863, 0.303012985114696, -0.161229841765317]]
        if flip_hardcoded:
            t_forward = np.array(t_forward).T

    elif transform_type == 'dct':
        t_forward = dct(np.eye(n), norm='ortho').T
    elif transform_type == 'eye':
        t_forward = np.eye(n)
    elif transform_type == 'dst':
        t_forward = dst(np.eye(n), norm='ortho').T
    elif transform_type == 'DCrand':
        x = np.random.normal(n)
        x[:, 0] = np.ones(len(x[:, 0]))
        q, _, _ = np.linalg.qr(x)
        if q[0] < 0:
            q = -q

        t_forward = q.T
    elif transform_type == 'fft':
        t_forward = ifft(np.eye(n)) * np.sqrt(n)
        t_inverse = np.conj(t_forward)
        return t_forward, t_inverse
    elif pywt is not None:
        # a wavelet decomposition supported by PyWavelets
        # Set periodic boundary conditions, to preserve bi-orthogonality
        t_forward = np.zeros((n, n))

        for ii in range(n):
            temp = np.zeros(n)
            temp[0] = 1.0
            temp = np.roll(temp, (ii, dec_levels))
            tt = pywt.wavedec(temp, transform_type, mode='periodization', level=int(np.log2(n)))
            cc = np.hstack(tt)
            t_forward[:, ii] = cc

    else:
        raise ValueError("Transform of " + transform_type + "couldn't be found and PyWavelets couldn't be imported!")

    t_forward = np.array(t_forward)
    # Normalize the basis elements
    if not ((n == 8) and transform_type == 'bior1.5'):
        try:
            t_forward = (t_forward.T @ np.diag(np.sqrt(1. / sum(t_forward ** 2, 0)))).T
        except TypeError:  # t_forward was not an array...
            pass

    # Compute the inverse transform matrix
    try:
        t_inverse = np.linalg.inv(t_forward)
    except LinAlgError:
        t_inverse = np.array([[1]])

    return t_forward, t_inverse


def _get_kernel_from_psd(sigma_psd: Union[np.ndarray, float], single_dim_psd: bool = False) -> np.ndarray:
    """
    Calculate a correlation kernel from the input PSD / std through IFFT2
    :param sigma_psd: PSD or std / 3-d concatenation of such
    :param single_dim_psd: True if sigma_psd is a std
    :return: a correlation kernel
    """
    if single_dim_psd:
        return np.linalg.norm(np.sqrt(sigma_psd / sigma_psd.size))

    sig = np.sqrt(sigma_psd / float(sigma_psd.size))
    return fftshift(np.real(ifftn(sig, axes=(0, 1, 2))), axes=(0, 1, 2))


def _process_psd(sigma_psd: Union[np.ndarray, float], z: np.ndarray,
                 single_dim_psd: bool, pad_size: tuple, profile: BM4DProfile) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Process input PSD for BM4D to acquire relevant inputs.
    :param sigma_psd: PSD (MxNxC) or a list of stds
    :param z: noisy image
    :param single_dim_psd: True if original input was sigma, not PSD
    :param pad_size: size to pad for refiltering
    :param profile: BM4DProfile used for this run
    :return: Tuple(sigma_psd2, psd_blur, psd_k)
            sigma_psd2 is equal to sigma_psd if refiltering is not used,
            otherwise it's the PSD in padded size
            psd_blur is equal to sigma_psd if Nf == 0 or single_dim_psd, otherwise it's a blurred PSD
            psd_k is the kernel used to blur the PSD (or [[[1]]])
    """
    temp_kernel = _get_kernel_from_psd(sigma_psd, single_dim_psd)
    
    auto_params = profile.lambda_thr is None or profile.mu2 is None or \
        (profile.denoise_residual and (profile.lambda_thr_re is None or profile.mu2_re is None))

    if auto_params and not single_dim_psd:
        # Since we are using 2-D parameter matrix, get a 2-D projection based on principal axes
        # (of max variance) of the PSD
        if temp_kernel.ndim > 2 and temp_kernel.shape[2] > 1:
            psd65 = _shrink_and_normalize_psd_for_2d(temp_kernel, (65, 65, 65))
            psd65 = _project_psd_for_parameter_est(fftshift(psd65))
            psd65 = psd65 / np.mean(psd65) * 65 * 65
        else:
            psd65 = _shrink_and_normalize_psd_for_2d(temp_kernel, (65, 65, 1))
        lambda_thr, mu2, lambda_thr_re, mu2_re = _estimate_parameters_for_psd(psd65)
    else:
        lambda_thr = [3.0]
        mu2 = [0.4]
        lambda_thr_re = [2.5]
        mu2_re = [3.6]
        
    # Create bigger PSD if needed
    if profile.denoise_residual and (pad_size[0] or pad_size[1] or pad_size[2]) and not single_dim_psd:

        pads_width = ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2]))
        temp_kernel = np.pad(temp_kernel, pads_width, 'constant')
        sigma_psd2 = abs(fftn(temp_kernel, axes=(0, 1, 2))) ** 2 * z.shape[0] * z.shape[1] * z.shape[2]
    else:
        sigma_psd2 = sigma_psd

    # Ensure PSD resized to nf is usable
    if profile.nf[0] > 0 and profile.nf[1] > 0 and profile.nf[2] > 0 and not single_dim_psd:

        sigma_psd_copy = _process_psd_for_nf(sigma_psd, None, profile)

        psd_k_sz = [1 + 2 * (np.floor(0.5 * sigma_psd_copy.shape[0] / profile.nf[0])),
                    1 + 2 * (np.floor(0.5 * sigma_psd_copy.shape[1] / profile.nf[1])),
                    1 + 2 * (np.floor(0.5 * sigma_psd_copy.shape[2] / profile.nf[2]))
                    ]
        psd_k = gaussian_kernel_3d([int(psd_k_sz[0]), int(psd_k_sz[1]), int(psd_k_sz[2])],
                                   1 + 2 * (np.floor(0.5 * sigma_psd_copy.shape[0] / profile.nf[0])) / 20,
                                   1 + 2 * (np.floor(0.5 * sigma_psd_copy.shape[1] / profile.nf[1])) / 20,
                                   1 + 2 * (np.floor(0.5 * sigma_psd_copy.shape[2] / profile.nf[2])) / 20)

        psd_k = psd_k / np.sum(psd_k)
        psd_blur = correlate(sigma_psd_copy, psd_k, mode='wrap')

    else:
        psd_k = np.array([[[1]]])
        psd_blur = np.copy(sigma_psd)

    profile.lambda_thr = lambda_thr[0] if profile.lambda_thr is None else profile.lambda_thr
    profile.mu2 = mu2[0] if profile.mu2 is None else profile.mu2
    profile.lambda_thr_re = lambda_thr_re[0] if profile.lambda_thr_re is None else profile.lambda_thr_re
    profile.mu2_re = mu2_re[0] if profile.mu2_re is None else profile.mu2_re

    profile.lambda_thr *= profile.filter_strength
    profile.mu2 *= profile.filter_strength ** 2
    profile.lambda_thr_re *= profile.filter_strength
    profile.mu2_re = profile.filter_strength ** 2

    return sigma_psd2, psd_blur, psd_k


def _get_transforms(profile_obj: BM4DProfile, stage_ht: bool) -> tuple:
    """
    Get transform matrices used by BM4D.
    :param profile_obj: profile used by the execution.
    :param stage_ht: True if we are doing hard-thresholding with the results
    :return: t_forward, t_inverse, hadper_trans_single_den, inverse_hadper_trans_single_den, wwin3d
            (forward transform, inverse transform, 3rd dim forward transforms, 3rd dim inverse transforms,
            kaiser window for aggregation)
    """
    # get (normalized) forward and inverse transform matrices
    if stage_ht:

        tf1, ti1 = _get_transf_matrix(profile_obj.bs_ht[0], profile_obj.transform_local_ht_name,
                                      profile_obj.dec_level, False)
        tf2, ti2 = _get_transf_matrix(profile_obj.bs_ht[1], profile_obj.transform_local_ht_name,
                                      profile_obj.dec_level, False)
        tf3, ti3 = _get_transf_matrix(profile_obj.bs_ht[2], profile_obj.transform_local_ht_name,
                                      profile_obj.dec_level, False)

        t_forward = [tf1, tf2, tf3]
        t_inverse = [ti1, ti2, ti3]

    else:
        tf1, ti1 = _get_transf_matrix(profile_obj.bs_wiener[0], profile_obj.transform_local_wiener_name,
                                      0, False)
        tf2, ti2 = _get_transf_matrix(profile_obj.bs_wiener[1], profile_obj.transform_local_wiener_name,
                                      0, False)
        tf3, ti3 = _get_transf_matrix(profile_obj.bs_wiener[2], profile_obj.transform_local_wiener_name,
                                      0, False)

        t_forward = [tf1, tf2, tf3]
        t_inverse = [ti1, ti2, ti3]

    if profile_obj.transform_nonlocal_name == 'haar' or profile_obj.transform_nonlocal_name[-3:] == '1.1':
        # If Haar is used in the 3-rd dimension, then a fast internal transform is used,
        # thus no need to generate transform matrices.
        hadper_trans_single_den = {}
        inverse_hadper_trans_single_den = {}
    else:
        # Create transform matrices. The transforms are later applied by
        # matrix-vector multiplication for the 1D case.
        hadper_trans_single_den = {}
        inverse_hadper_trans_single_den = {}

        rangemax = np.ceil(np.log2(np.max([profile_obj.max_stack_size_ht, profile_obj.max_stack_size_wiener]))) + 1
        for hpow in range(0, int(rangemax)):
            h = 2 ** hpow
            t_forward_3d, t_inverse_3d = _get_transf_matrix(h, profile_obj.transform_nonlocal_name, 0, True)
            hadper_trans_single_den[h] = t_forward_3d.T
            inverse_hadper_trans_single_den[h] = t_inverse_3d.T

    # 2D Kaiser windows used in the aggregation of block-wise estimates
    if profile_obj.beta_wiener == 2 and profile_obj.beta == 2 and profile_obj.bs_wiener == 8 and profile_obj.bs_ht == 8:
        # hardcode the window function so that the signal processing toolbox is not needed by default
        wwin3d = [[0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924],
                  [0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989],
                  [0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846],
                  [0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325],
                  [0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325],
                  [0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846],
                  [0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989],
                  [0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924]]
    else:
        if stage_ht:
            # Kaiser window used in the aggregation of the HT part
            wwin3d = np.transpose([np.kaiser(profile_obj.bs_ht[0], profile_obj.beta)]) @ [
                np.kaiser(profile_obj.bs_ht[1], profile_obj.beta)]
            kaiser3 = np.kaiser(profile_obj.bs_ht[2], profile_obj.beta)
            wwin3d = np.repeat(np.atleast_3d(wwin3d), profile_obj.bs_ht[2], axis=2) * kaiser3
        else:
            # Kaiser window used in the aggregation of the Wiener filt. part
            wwin3d = np.transpose([np.kaiser(profile_obj.bs_wiener[0], profile_obj.beta_wiener)]) @ [
                np.kaiser(profile_obj.bs_wiener[1], profile_obj.beta_wiener)]
            kaiser3 = np.kaiser(profile_obj.bs_wiener[2], profile_obj.beta_wiener)
            wwin3d = np.repeat(np.atleast_3d(wwin3d), profile_obj.bs_wiener[2], axis=2) * kaiser3

    wwin3d = np.array(wwin3d)
    return t_forward, t_inverse, hadper_trans_single_den, inverse_hadper_trans_single_den, wwin3d


def _estimate_parameters_for_psd(psd65_full: np.ndarray) -> (list, list, list, list):
    """
    Estimate BM3D parameters based on the PSD.
    :param psd65_full: input PSDs (65x65xn)
    :return: (lambda, mu, refiltering lambda, refiltering mu)
    """

    # Get the optimal parameters and matching features for a bunch of PSDs
    path = os.path.dirname(__file__)
    data = loadmat(os.path.join(path, 'param_matching_data.mat'))
    features = data['features']
    maxes = data['maxes']

    sz = 65
    data_sz = 500
    indices_to_take = [1, 3, 5, 7, 9, 12, 17, 22, 27, 32]

    llambda = []
    wielambda = []
    llambda2 = []
    wielambda2 = []

    # Get separate parameters for each PSD provided
    for psd_num in range(psd65_full.shape[2] if len(psd65_full.shape) > 2 else 1):

        if len(psd65_full.shape) > 2:
            psd65 = fftshift(psd65_full[:, :, psd_num], axes=(0, 1))
        else:
            psd65 = fftshift(psd65_full[:, :], axes=(0, 1))

        # Get features for this PSD
        pcaxa = _get_features(psd65, sz, indices_to_take)

        # Calculate distances to other PSDs
        mm = np.mean(features, 1)
        f2 = features - np.repeat(np.atleast_2d(mm).T, data_sz, axis=1)
        c = f2 @ f2.T
        c /= data_sz
        pcax2 = pcaxa.T - mm
        u, s, v = svd(c)
        f2 = u @ f2
        pcax2 = u @ pcax2
        f2 = f2 * np.repeat(np.atleast_2d(np.sqrt(s)).T, 500, axis=1)
        pcax2 = pcax2 * np.sqrt(s)

        diff_pcax = np.sqrt(np.sum(abs(f2 - np.repeat(np.atleast_2d(pcax2).T, data_sz, axis=1)) ** 2, 0))
        dff_i = np.argsort(diff_pcax)

        # Take 20 most similar PSDs into consideration
        count = 20
        diff_indices = dff_i[0:count]

        # Invert, smaller -> bigger weight
        diff_inv = 1. / (diff_pcax + EPS)
        diff_inv = diff_inv[diff_indices] / np.sum(diff_inv[diff_indices])

        # Weight
        param_idxs = np.sum(diff_inv * maxes[diff_indices, :].T, 1)

        lambdas = np.linspace(2.5, 4.5, 21)
        wielambdas = np.linspace(0.2, 4.2, 21)

        # Get parameters from indices
        # Interpolate lambdas and mu^2s from the list
        for ix in [0, 2]:
            param_idx = max(1, param_idxs[ix]) - 1
            param_idx2 = max(1, param_idxs[ix + 1]) - 1

            l1 = lambdas[int(np.floor(param_idx))]
            l2 = lambdas[int(min(np.ceil(param_idx), lambdas.size - 1))]

            w1 = wielambdas[int(np.floor(param_idx2))]
            w2 = wielambdas[int(min(np.ceil(param_idx2), wielambdas.size - 1))]

            param_smooth = param_idx - np.floor(param_idx)
            param_smooth2 = param_idx2 - np.floor(param_idx2)

            if ix == 0:
                llambda.append(l2 * param_smooth + l1 * (1 - param_smooth))
                wielambda.append(w2 * param_smooth2 + w1 * (1 - param_smooth2))
            elif ix == 2:
                llambda2.append(l2 * param_smooth + l1 * (1 - param_smooth))
                wielambda2.append(w2 * param_smooth2 + w1 * (1 - param_smooth2))

    return llambda, wielambda, llambda2, wielambda2


def _get_features(psd: np.ndarray, sz: int, indices_to_take: list) -> np.ndarray:
    """
    Calculate features for a PSD from integrals
    :param psd: The PSD to calculate features for.
    :param sz: Size of the PSD.
    :param indices_to_take: Indices from which to split the integrals.
    :return: array of features, length indices_to_take*2
    """
    int_rot, int_rot2 = _pcax(psd)
    f1 = np.zeros(len(indices_to_take) * 2)

    for ii in range(0, len(indices_to_take)):
        rang = indices_to_take[ii]
        if ii > 0:
            rang = [i for i in range(indices_to_take[ii - 1], rang)]
        else:
            rang -= 1
        rn = len(rang) if type(rang) == list else 1
        f1[ii] = np.sum(int_rot[np.array([np.ceil(sz / 2) + rang - 1], dtype=int)]) / rn
        f1[len(indices_to_take) + ii] = np.sum(int_rot2[np.array([np.ceil(sz / 2) + rang - 1], dtype=int)]) / rn
        pass

    return f1


def _pcax(psd: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate integrals through the principal axes of psd.
    :param psd: psd.
    :return: (intg1, intg2) : two integrals along the two axes.
    """
    n = psd.shape[0]
    [g2, g1] = np.meshgrid([i for i in range(1, n+1)], [i for i in range(1, n+1)])

    def trapz2d(tg2, tg1, p):
        return np.trapz(_trapz2(tg2, p, 1), tg1[:, 0], axis=0)

    p_n = psd / trapz2d(g2, g1, psd)

    m2 = trapz2d(g2, g1, p_n * g2)
    m1 = trapz2d(g2, g1, p_n * g1)
    c = np.zeros(4)

    q1 = [2, 1, 1, 0]
    q2 = [0, 1, 1, 2]

    for jj in [0, 1, 3]:
        c[jj] = np.squeeze(trapz2d(g2, g1, p_n * (g2 - m2) ** q1[jj] * (g1 - m1) ** q2[jj]))

    c[2] = c[1]
    c = c.reshape((2, 2))
    u, s, v = svd(c)

    n3 = 3 * n

    g2_n3, g1_n3 = np.meshgrid(np.array([i for i in range(1, n3 + 1)]) - (n3 + 1) / 2,
                               np.array([i for i in range(1, n3 + 1)]) - (n3 + 1) / 2)

    # Rotate PSDs and calculate integrals along the rotated PSDs
    theta = np.angle(u[0, 0] + 1j * u[0, 1])
    g2_rot = g2_n3[n:2*n, n:2*n] * np.cos(theta) - g1_n3[n:2*n, n:2*n] * np.sin(theta)
    g1_rot = g1_n3[n:2*n, n:2*n] * np.cos(theta) + g2_n3[n:2*n, n:2*n] * np.sin(theta)

    psd_rep = np.tile(psd, (3, 3))
    psd_rot = interpn((g2_n3[0, :], g2_n3[0, :]), psd_rep, (g1_rot, g2_rot))
    int_rot = _trapz2(g1, psd_rot, 0)

    theta2 = np.angle(u[1, 0] + 1j * u[1, 1])
    g2_rot = g2_n3[n:2*n, n:2*n] * np.cos(theta2) - g1_n3[n:2*n, n:2*n] * np.sin(theta2)
    g1_rot = g1_n3[n:2*n, n:2*n] * np.cos(theta2) + g2_n3[n:2*n, n:2*n] * np.sin(theta2)
    psd_rot2 = interpn((g2_n3[0, :], g2_n3[0, :]), psd_rep, (g1_rot, g2_rot))
    int_rot2 = _trapz2(g1, psd_rot2, 0)

    return int_rot, int_rot2

def _project_psd_for_parameter_est(psd: np.ndarray) -> np.ndarray:
    """
    Return the 2-D projection of maximum variance squeezed along a principal axis.
    :param psd: psd, NxNxN
    :return: 2-D PSD for parameter estimation
    """

    n = psd.shape[0]
    [g2, g1, g3] = np.meshgrid([i for i in range(1, n+1)], [i for i in range(1, n+1)], [i for i in range(1, n+1)])

    def trapz2d(tg3, tg2, p):
        return _trapz3(tg2[:, :, 0:1], _trapz3(tg3, p, 2), 1)

    def trapz3d(tg3, tg2, tg1, p):
        return np.trapz(trapz2d(tg3, tg2, p), tg1[:, 0:1, 0:1], axis=0)

    p_n = psd / trapz3d(g3, g2, g1, psd)

    m2 = trapz3d(g3, g2, g1, p_n * g2)
    m1 = trapz3d(g3, g2, g1, p_n * g1)
    m3 = trapz3d(g3, g2, g1, p_n * g3)

    c = np.zeros((9,))

    q1 = [2, 1, 1, 1, 0, 0, 1, 0, 0]
    q2 = [0, 1, 0, 1, 2, 1, 0, 1, 0]
    q3 = [0, 0, 1, 0, 0, 1, 1, 1, 2]

    for jj in [0, 1, 2, 4, 5, 8]:
        c[jj] = np.squeeze(trapz3d(g3, g2, g1, p_n * (g1 - m1) ** q1[jj] * (g2 - m2) ** q2[jj] * (g3 - m3) ** q3[jj]))

    c[3] = c[1]
    c[6] = c[2]
    c[7] = c[5]
    c = c.reshape((3, 3))
    u, s, v = svd(c)

    # Select highest variance
    p_rot_f = 0
    p_rot_var_f = 0

    for ax_ix in range(0, 3):
        v = u[:, ax_ix]
        p_rot = ifftshift(rotate_to(psd, v))
        p_rot_proj = np.squeeze(p_rot[0, :, :])
        p_var = np.sum(p_rot_proj)
        if p_var > p_rot_var_f:
            p_rot_f = p_rot_proj
            p_rot_var_f = p_var

    return p_rot_f


def rotate_to(psd: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate the input array (NxNxN) (psd) to be oriented by a (3-D) vector v
    :param psd: PSD, NxNxN
    :param v: direction vector (3)
    :return: rotated PSD
    """

    r = np.array([1, 0, 0])
    
    # Already correct rotation
    if np.sum(np.abs(v-r)) < 1e-3 or np.sum(np.abs(v+r)) < 1e-3:
        return psd



    rot_ax = np.cross(v, r)
    rot_ax = rot_ax / np.linalg.norm(rot_ax.ravel())
    rot_angle = np.arccos(np.minimum(np.maximum(np.dot(v, r), -1), 1))
    
    rot_ax_cr = np.array([[0, -rot_ax[2], rot_ax[1]],
                        [rot_ax[2], 0, -rot_ax[0]],
                        [-rot_ax[1], rot_ax[0], 0]])

    rotation_matrix = np.cos(rot_angle) * np.eye(3) + np.sin(rot_angle) * rot_ax_cr +\
                        (1 - np.cos(rot_angle)) * (np.atleast_2d(rot_ax).T @ np.atleast_2d(rot_ax))

    n = psd.shape[0]
    n3 = 3 * n

    g2_n3, g1_n3, g3_n3 = np.meshgrid(np.array([i for i in range(1, n3 + 1)]) - (n3 + 1) / 2,
                                      np.array([i for i in range(1, n3 + 1)]) - (n3 + 1) / 2,
                                      np.array([i for i in range(1, n3 + 1)]) - (n3 + 1) / 2)

    gns = np.array((g1_n3, g2_n3, g3_n3))
    g_rot = np.zeros((n, n, n, 3))

    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                rot = rotation_matrix @ np.squeeze(gns[:, n+i, n+j, n+k])
                g_rot[i, j, k, :] = rot

    # Rotate PSD, tile to ensure continuity on edges since it is periodic anyway
    psd_rep = np.tile(psd, (3, 3, 3))
    psd_rot = interpn((g2_n3[0, :, 0], g2_n3[0, :, 0], g2_n3[0, :, 0]),
                      psd_rep, (g_rot[:, :, :, 0], g_rot[:, :, :, 1], g_rot[:, :, :, 2]))

    return psd_rot


def _trapz2(x: np.ndarray, y: np.ndarray, dimm: int) -> np.ndarray:
    """
    Calculate the integals of an 2-D array along specified dimension
    :param x: values of x
    :param y: values of y
    :param dimm: 1 or 0
    :return: integrals along the axis
    """
    if dimm == 1:
        intg = np.sum((y[:, 1:] + y[:, 0:-1]) / 2. * (x[:, 1:] - x[:, 0:-1]), axis=1)
    else:
        intg = np.sum((y[1:, :] + y[0:-1, :]) / 2. * (x[1:, :] - x[0:-1, :]), axis=0)
    return intg


def _shrink_and_normalize_psd_for_2d(temp_kernel: np.ndarray, new_size_3d: tuple) -> np.ndarray:
    """
    Calculate shrunk PSD from image-size, normalized, kernel. Ignore 3rd dimension when multiplying by |X|
    :param temp_kernel: Input kernel(s), MxNxC
    :param new_size_3d: new size
    :return: PSD of the normalized kernel
    """
    minus_size = np.array(np.ceil((np.array(temp_kernel.shape) - np.array(new_size_3d)) / 2), dtype=int)
    minus_size = np.maximum(minus_size, 0)
    temp_kernel_shrunk = np.copy(temp_kernel[minus_size[0]:minus_size[0] + new_size_3d[0],
                                 minus_size[1]:minus_size[1] + new_size_3d[1],
                                 minus_size[2]:minus_size[2] + new_size_3d[2]])

    temp_kernel_shrunk /= np.sqrt(np.sum(temp_kernel_shrunk ** 2))

    return np.abs(fftn(temp_kernel_shrunk, shape=new_size_3d)) ** 2 * new_size_3d[0] * new_size_3d[1]

def _trapz3(x: np.ndarray, y: np.ndarray, dimm: int) -> np.ndarray:
    """
    Calculate the integals of an 2-D array along specified dimension
    :param x: values of x
    :param y: values of y
    :param dimm: 1 or 0
    :return: integrals along the axis
    """
    if dimm == 2:
        intg = np.sum((y[:, :, 1:] + y[:, :, 0:-1]) / 2. * (x[:, :, 1:] - x[:, :, 0:-1]), axis=2, keepdims=True)
    elif dimm == 1:
        intg = np.sum((y[:, 1:, :] + y[:, 0:-1, :]) / 2. * (x[:, 1:, :] - x[:, 0:-1, :]), axis=1, keepdims=True)
    else:
        intg = np.sum((y[1:, :, :] + y[0:-1, :, :]) / 2. * (x[1:, :, :] - x[0:-1, :, :]), axis=0, keepdims=True)
    return intg


def get_shift_params(block_size: Tuple[int], step_size: Tuple[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select pre-optimized block shifts based on the 1st and 3rd dimension block and step sizes,
    avoiding boundary overlaps.
    :param block_size: block-size tuple
    :param step_size: step-size tuple
    :return: tuple of list of shift params of each dimension, 2nd always empty.
    """

    block_size_1 = min(block_size[0], block_size[1])
    block_size_3 = block_size[2]

    step_size_1 = min(step_size[0], step_size[1])
    step_size_3 = step_size[2]

    shift_list = [[0], [0, 1], [0, 1, 2, 1], [0, 2, 1], [0, 2, 0, 1], [0, 2], [0, 1, 2], [0, 3, 2, 1], [0, 3],
                  [0, 2, 3, 1], [0, 1, 1, 0], [0, 3, 1], [0, 2, 1, 3], [0, 4, 3, 1], [0, 3, 4, 1], [0, 4, 2],
                  [0, 4], [0, 2, 4, 2], [0, 2, 4], [0, 4, 1, 3], [0, 1, 2, 3], [0, 3, 1, 2], [0, 3, 0, 2],
                  [0, 3, 1, 4], [0, 5, 1, 4], [0, 5], [0, 1, 4, 5], [0, 5, 1], [0, 1, 3], [0, 3, 5, 2],
                  [0, 5, 2, 3], [0, 1, 3, 4], [0, 5, 4, 1], [0, 2, 4, 1], [0, 3, 1, 5], [0, 6, 2, 4],
                  [0, 1, 4, 2], [0, 4, 6, 2], [0, 6, 3], [0, 2, 3, 5], [0, 1, 2, 4], [0, 1, 3, 5],
                  [0, 2, 4, 6], [0, 5, 2], [0, 4, 0, 1], [0, 5, 0, 2], [0, 6, 4, 2], [0, 7, 2, 5],
                  [0, 6], [0, 1, 4], [0, 5, 3, 2], [0, 2, 5], [0, 5, 7, 2], [0, 7, 2], [0, 7, 1, 6],
                  [0, 4, 5, 1], [0, 1, 5], [0, 3, 2, 5], [0, 4, 2, 6], [0, 3, 6, 1], [0, 6, 1],
                  [0, 1, 2, 6], [0, 6, 2], [0, 3, 6], [0, 5, 2, 7], [0, 1, 6, 7], [0, 6, 7, 1],
                  [0, 2, 7], [0, 7, 6, 1], [0, 1, 7], [0, 3, 6, 3]]

    shifts_1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 1, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 5,
                5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 5,
                5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 10, 0, 0, 0, 6, 6, 6, 6, 6, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 5,
                5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 10, 1, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 20, 20, 20, 0, 0, 0, 0, 0, 22,
                5, 5, 0, 0, 0, 0, 0, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 20, 20, 20, 20, 0, 0, 0, 0, 22,
                22, 5, 5, 0, 0, 0, 0, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 20, 20, 20, 20, 20, 0, 0, 0, 22,
                22, 5, 5, 5, 0, 0, 0, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 20, 20, 20, 20, 20, 20, 0, 0, 22,
                22, 22, 5, 5, 5, 0, 0, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 22,
                5, 5, 0, 0, 0, 0, 0, 15, 8, 8, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 22,
                22, 5, 5, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 10, 0, 0, 0, 6, 6, 6, 6, 6, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 22,
                22, 5, 5, 5, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 10, 1, 10, 10, 10, 10, 0, 0, 6, 6, 6, 6, 6, 6, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 22,
                22, 22, 5, 5, 5, 0, 0, 15, 15, 8, 8, 8, 8, 0, 0, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 10, 10, 10, 0, 6, 6, 6, 6, 6, 6, 6, 0, 5, 5, 5, 5, 5, 5, 5, 0, 22,
                22, 28, 5, 5, 5, 5, 0, 15, 15, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 44,
                44, 44, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 45, 5, 5, 0, 0, 0, 0, 0, 46, 46, 46, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 10, 10, 10, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 9, 9, 9, 9, 0, 0, 0, 0, 44,
                44, 44, 44, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 45, 5, 5, 5, 0, 0, 0, 0, 46, 46, 46, 46, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 10, 0, 0, 0, 6, 6, 6, 6, 6, 0, 0, 0, 9, 9, 9, 9, 9, 0, 0, 0, 44,
                44, 1, 44, 44, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 46, 46, 46, 46, 46, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 0, 0, 3, 3, 6, 3, 3, 6, 0, 0, 9, 9, 9, 9, 9, 9, 0, 0, 44,
                44, 44, 44, 44, 44, 0, 0, 8, 8, 8, 8, 8, 8, 0, 0, 45, 45, 5, 5, 5, 5, 0, 0, 46, 46, 46, 46, 46, 46, 0,
                0, 0,
                0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 0, 3, 3, 3, 3, 3, 3, 6, 0, 9, 9, 9, 9, 9, 9, 9, 0, 44,
                44, 44, 6, 44, 44, 44, 0, 8, 8, 8, 8, 8, 8, 8, 0, 45, 45, 5, 5, 5, 5, 5, 0, 46, 46, 46, 46, 46, 46, 46,
                0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 10, 10, 10, 10, 10, 10, 10, 3, 6, 3, 6, 3, 3, 3, 6, 9, 9, 9, 9, 9, 9, 9, 9, 44,
                44, 44, 44, 44, 44, 44, 44, 8, 8, 8, 8, 8, 8, 8, 8, 45, 45, 5, 5, 5, 5, 5, 5, 46, 46, 46, 46, 46, 46,
                46, 46, 71]

    shifts_2 = [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 4, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 4, 5, 5, 0, 0, 0, 0, 0, 7, 8, 8, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 7, 8, 7, 0, 0, 0, 0, 6, 6, 5, 5, 0, 0, 0, 0, 9, 11, 8, 8, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 3, 5, 5, 0, 0, 0, 0, 0, 12, 5, 5, 0, 0, 0, 0, 0, 13,
                8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 7, 8, 8, 0, 0, 0, 0, 6, 6, 5, 5, 0, 0, 0, 0, 7, 5, 5, 5, 0, 0, 0, 0, 14,
                15, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 7, 7, 8, 8, 8, 0, 0, 0, 15, 15, 16, 16, 16, 0, 0, 0, 17, 18, 5, 5, 5, 0, 0, 0, 19,
                15, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 4, 5, 5, 0, 0, 0, 0, 0, 21, 8, 8, 0, 0, 0, 0, 0, 23,
                8, 8, 0, 0, 0, 0, 0, 24, 16, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 20, 20, 20, 0, 0, 0, 0, 3, 3, 5, 5, 0, 0, 0, 0, 12, 20, 8, 8, 0, 0, 0, 0, 13,
                18, 8, 8, 0, 0, 0, 0, 26, 27, 16, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 20, 9, 1, 20, 20, 0, 0, 0, 18, 18, 5, 5, 5, 0, 0, 0, 20, 28, 8, 8, 8, 0, 0, 0, 18,
                18, 8, 8, 8, 0, 0, 0, 24, 15, 16, 16, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 20, 20, 20, 20, 20, 0, 0, 15, 18, 18, 5, 5, 5, 0, 0, 29, 30, 30, 25, 25, 25, 0,
                0, 31,
                23, 18, 8, 8, 8, 0, 0, 32, 24, 27, 16, 16, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 12, 5, 5, 0, 0, 0, 0, 0, 33,
                5, 5, 0, 0, 0, 0, 0, 34, 16, 16, 0, 0, 0, 0, 0, 35, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 7, 8, 8, 0, 0, 0, 0, 6, 6, 1, 1, 0, 0, 0, 0, 7, 5, 5, 5, 0, 0, 0, 0, 36,
                18, 5, 5, 0, 0, 0, 0, 18, 18, 16, 16, 0, 0, 0, 0, 37, 38, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 7, 7, 8, 8, 8, 0, 0, 0, 6, 3, 1, 1, 1, 0, 0, 0, 12, 12, 5, 5, 5, 0, 0, 0, 18,
                18, 5, 5, 5, 0, 0, 0, 15, 15, 16, 16, 16, 0, 0, 0, 35, 38, 25, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0,
                0, 0, 0, 0, 0, 0, 0, 32, 7, 32, 8, 8, 8, 0, 0, 3, 6, 6, 1, 1, 1, 0, 0, 39, 29, 17, 5, 5, 5, 0, 0, 40,
                33, 18, 5, 5, 5, 0, 0, 41, 34, 18, 16, 16, 16, 0, 0, 37, 35, 38, 25, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0,
                0, 0, 0, 0, 0, 0, 0, 7, 9, 32, 8, 8, 8, 8, 0, 6, 9, 3, 1, 1, 1, 1, 0, 29, 39, 42, 42, 42, 42, 42, 0, 42,
                42, 38, 5, 5, 5, 5, 0, 37, 37, 15, 16, 16, 16, 16, 0, 35, 35, 43, 25, 25, 25, 25, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 3, 6, 6, 0, 0, 0, 0, 0, 7, 5, 5, 0, 0, 0, 0, 0, 19,
                16, 16, 0, 0, 0, 0, 0, 29, 8, 8, 0, 0, 0, 0, 0, 37, 25, 25, 0, 0, 0, 0, 0, 47, 48, 48, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 7, 8, 8, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 20, 11, 5, 5, 0, 0, 0, 0, 49,
                49, 16, 16, 0, 0, 0, 0, 50, 51, 8, 8, 0, 0, 0, 0, 35, 51, 25, 25, 0, 0, 0, 0, 52, 53, 48, 48, 0, 0, 0,
                0, 0,
                0, 0, 0, 0, 0, 0, 0, 7, 9, 8, 8, 8, 0, 0, 0, 15, 15, 15, 15, 15, 0, 0, 0, 12, 11, 5, 5, 5, 0, 0, 0, 49,
                6, 1, 16, 16, 0, 0, 0, 29, 39, 8, 8, 8, 0, 0, 0, 51, 43, 25, 25, 25, 0, 0, 0, 54, 53, 48, 48, 48, 0, 0,
                0, 0,
                0, 0, 0, 0, 0, 0, 0, 55, 26, 32, 8, 8, 8, 0, 0, 27, 27, 15, 56, 56, 15, 0, 0, 39, 29, 51, 5, 5, 5, 0, 0,
                26,
                55, 55, 16, 16, 16, 0, 0, 39, 57, 8, 8, 8, 8, 0, 0, 58, 42, 51, 25, 25, 25, 0, 0, 52, 47, 54, 48, 48,
                48, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 26, 26, 32, 8, 8, 8, 8, 0, 56, 27, 56, 56, 56, 56, 15, 0, 59, 59, 60, 48, 48, 48,
                48, 0, 61,
                61, 62, 48, 48, 48, 48, 0, 29, 29, 63, 8, 8, 8, 8, 0, 37, 37, 51, 25, 25, 25, 25, 0, 47, 64, 53, 48, 48,
                48, 48, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 20, 55, 32, 32, 8, 8, 8, 8, 56, 15, 27, 15, 56, 56, 56, 15, 65, 66, 67, 54, 48, 48,
                48, 48, 66,
                68, 69, 66, 48, 48, 48, 48, 29, 50, 38, 70, 8, 8, 8, 8, 37, 46, 51, 51, 25, 25, 25, 25, 64, 52, 67, 54,
                48, 48, 48, 48, 71]

    if block_size_3 == 1 and step_size_3 == 1 and 2 < block_size_1 < 9 and step_size_1 < 8:
        shift_list = [[0], [0, 1], [0, 2, 1], [0, 2], [0, 1, 2], [0, 3, 2, 1], [0, 3], [0, 1, 1, 0], [0, 1, 2, 3],
                      [0, 5], [0, 4], [0, 2, 3, 1], [0, 4, 0, 1], [0, 6]]
        shifts_1 = [0, 1, 3, 0, 0, 0, 0, 0, 0, 1, 3, 6, 0, 0, 0, 0, 0, 1, 3, 3, 6, 0, 0, 0, 0, 1, 3, 6, 6, 9, 0, 0, 0,
                    1, 1, 3, 3, 10, 9, 0, 0, 1, 4, 3, 10, 6, 9, 13, 14]
        block_size_1 -= 1
        step_size_1 -= 1
        ix = (block_size_1 - 2) * 8 + step_size_1
        sh_1 = shift_list[shifts_1[ix]]
        return np.array([0], dtype=int), np.array(sh_1, dtype=int), np.array([0], dtype=int)

    if block_size_1 > 8 or block_size_3 > 8 or step_size_1 > block_size_1 or step_size_3 > block_size_3 or \
        block_size_3 > block_size_1 or min(block_size_1, block_size_3) < 3:
        return np.array([0], dtype=int), np.array([0], dtype=int), np.array([0], dtype=int)

    block_size_1 -= 1
    block_size_3 -= 1
    step_size_1 -= 1
    step_size_3 -= 1

    ix = (block_size_1 - 2) * 6 * 8 * 8 + (block_size_3 - 2) * 8 * 8 + step_size_1 * 8 + step_size_3
    sh_1 = shift_list[shifts_1[ix]]
    sh_2 = shift_list[shifts_2[ix]]

    return np.array(sh_2, dtype=int), np.array(sh_1, dtype=int), np.array([0], dtype=int)
