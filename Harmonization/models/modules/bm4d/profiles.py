"""
profiles.py
Contains definitions of BM4DStages and BM4DProfiles classes,
as well as several predefined BM4DProfile subclasses.
"""

import enum
from typing import Tuple


class BM4DStages(enum.Enum):
    HARD_THRESHOLDING = 1
    WIENER_FILTERING = 2  # Pass a hard-thresholding estimate to the function instead of WIENER_FILTERING only
    ALL_STAGES = HARD_THRESHOLDING + WIENER_FILTERING


class BM4DProfile:
    """
    BM4DProfile object, containing the default settings for BM4D.
    Default values for our profile = 'np'
    """

    filter_strength = 1

    print_info = False

    # Transforms used
    transform_local_ht_name = 'bior1.5'
    transform_local_wiener_name = 'dct'
    transform_nonlocal_name = 'haar'

    # -- Exact variances for correlated noise: --

    # Variance calculation parameters
    nf = (16, 16, 16)  # domain size for FFT computations
    k = 4  # how many layers of var3D to calculate

    # Refiltering
    denoise_residual = False  # Perform residual thresholding and re-denoising
    residual_thr = 3  # Threshold for the residual HT (times sqrt(PSD))
    max_pad_size = None  # Maximum required pad size (= half of the kernel size), or 0 -> use image size

    # Block matching
    gamma = 3.0  # Block matching correction factor

    # Hard-thresholding (HT) parameters:
    bs_ht = (4, 4, 4)  # N1 x N1 is the block size used for the hard-thresholding (HT) filtering
    step_ht = (3, 3, 3)  # sliding step to process every next reference block
    max_stack_size_ht = 16  # maximum number of similar blocks (maximum size of the 3rd dimension of a 3D array)
    search_window_ht = (7, 7, 7)  # half side length of the search neighborhood for full-search block-matching (BM)
    tau_match = 2.9527  # threshold for the block-distance (d-distance)

    lambda_thr = None
    mu2 = None
    lambda_thr_re = None
    mu2_re = None
    beta = 2.0  # parameter of the 2D Kaiser window used in the reconstruction
    adjust_complex_params = False  # Adjust shrinkage parameters when input type is complex

    # Wiener filtering parameters:
    bs_wiener = (5, 5, 5)
    step_wiener = (3, 3, 3)
    max_stack_size_wiener = 32
    search_window_wiener = (7, 7, 7)
    tau_match_wiener = 0.7689
    beta_wiener = 2.0
    dec_level = 0  # dec. levels of the dyadic wavelet 2D transform for blocks
    #  (0 means full decomposition, higher values decrease the dec. number)

    min_3d_size_ht = 2
    min_3d_size_wiener = 2

    # Toggles partitioning block extraction from image, one for each dimension
    # Switching to True should be done progressively from the last dimension
    # If False, blocks across that dimension will be extracted & transformed at once.
    # If True, they will instead be extracted and transformed as the reference block
    # moves across that dimension (to conserve memory)
    split_block_extraction = (False, False, False)

    # If not equal to 1, sharpening will be done with power 1/sharpen_alpha
    sharpen_alpha = 1
    # Sharpening parameter for the "3rd dimension DC" of the group spectrum
    sharpen_alpha_3d = 1

    # Number of threads in concurrent application.
    # 0 = automatic
    # 1 = single threaded application
    # Note that if num_threads != 1, the output may vary very slightly due to the
    # changing order of operations in aggregation and the precision of float.
    num_threads = 0

    def set_sharpen(self, sharpen_alpha):
        """
        Set all sharpening values to sharpen_alpha.
        """
        self.sharpen_alpha = sharpen_alpha
        self.sharpen_alpha_3d = sharpen_alpha

    def get_block_size(self, mode: BM4DStages) -> Tuple[int, int, int]:
        """
        Get block size parameter.
        :param mode: BM4DStages enum value
        :return: block size
        """
        if mode == BM4DStages.HARD_THRESHOLDING or mode == BM4DStages.ALL_STAGES:
            return self.bs_ht
        else:
            return self.bs_wiener

    def get_step_size(self, mode: BM4DStages) -> Tuple[int, int, int]:
        """
        Get step size parameter.
        :param mode: BM4DStages enum value
        :return: step size
        """
        if mode == BM4DStages.HARD_THRESHOLDING or mode == BM4DStages.ALL_STAGES:
            return self.step_ht
        else:
            return self.step_wiener

    def get_max_3d_size(self, mode: BM4DStages) -> int:
        """
        Get maximum stack size in the 3rd dimension.
        :param mode: BM4DStages enum value
        :return: maximum stack size in the 3rd dimension
        """
        if mode == BM4DStages.HARD_THRESHOLDING or mode == BM4DStages.ALL_STAGES:
            return self.max_stack_size_ht
        else:
            return self.max_stack_size_wiener

    def get_search_window(self, mode: BM4DStages) -> Tuple[int, int, int]:
        """
        Get search window size parameter.
        :param mode: BM4DStages enum value
        :return: search window size
        """
        if mode == BM4DStages.HARD_THRESHOLDING or mode == BM4DStages.ALL_STAGES:
            return self.search_window_ht
        else:
            return self.search_window_wiener

    def get_block_threshold(self, mode: BM4DStages) -> float:
        """
        Get block matching threshold parameter.
        :param mode: BM4DStages enum value
        :return: block matching threshold
        """
        if mode == BM4DStages.HARD_THRESHOLDING or mode == BM4DStages.ALL_STAGES:
            return self.tau_match
        else:
            return self.tau_match_wiener

    def __setattr__(self, attr, value):
        """
        Override __setattr__ to prevent adding new values (by typo).
        Raises AttributeError if a new attribute is added.
        :param attr: Attribute name to modify
        :param value: Value to modify
        """
        if not hasattr(self, attr):
            raise AttributeError("Unknown attribute name: " + attr)
        super(BM4DProfile, self).__setattr__(attr, value)


class BM4DProfile2D(BM4DProfile):
    """
    Profile object for when the 3rd dimension is not big enough for the default block size.
    """
    print_info = False

    # Transforms used
    transform_local_ht_name = 'bior1.5'
    transform_local_wiener_name = 'dct'
    transform_nonlocal_name = 'haar'

    # -- Exact variances for correlated noise: --

    # Variance calculation parameters
    nf = (16, 16, 16)
    k = 4  # how many layers of var3D to calculate

    # Refiltering
    denoise_residual = False  # Perform residual thresholding and re-denoising
    residual_thr = 3  # Threshold for the residual HT (times sqrt(PSD))
    max_pad_size = None  # Maximum required pad size (= half of the kernel size), or 0 -> use image size

    # Block matching
    gamma = 3.0  # Block matching correction factor

    # Hard-thresholding (HT) parameters:
    bs_ht = (8, 8, 1)  # N1 x N1 is the block size used for the hard-thresholding (HT) filtering
    step_ht = (3, 3, 1)  # sliding step to process every next reference block
    max_stack_size_ht = 16  # maximum number of similar blocks (maximum size of the 3rd dimension of a 3D array)
    search_window_ht = (7, 7, 7)  # half side length of the search neighborhood for full-search block-matching (BM)
    tau_match = 2.9527  # threshold for the block-distance (d-distance)
    
    lambda_thr = None
    mu2 = None
    lambda_thr_re = None
    mu2_re = None
    beta = 2.0  # parameter of the 2D Kaiser window used in the reconstruction

    # Wiener filtering parameters:
    bs_wiener = (8, 8, 1)
    step_wiener = (3, 3, 1)
    max_stack_size_wiener = 32
    search_window_wiener = (7, 7, 7)
    tau_match_wiener = 0.3937
    beta_wiener = 2.0
    dec_level = 0  # dec. levels of the dyadic wavelet 2D transform for blocks
    #  (0 means full decomposition, higher values decrease the dec. number)


class BM4DProfileRefilter(BM4DProfile):
    """
    Refiltering enabled
    """
    denoise_residual = True


class BM4DProfile2DRefilter(BM4DProfile2D):
    """
    Refiltering enabled, 2-D blocks
    """
    denoise_residual = True


class BM4DProfile2DComplex(BM4DProfile2D):
    """
    Refiltering enabled, 2-D blocks
    """
    transform_local_ht_name = 'fft'
    transform_local_wiener_name = 'fft'
    adjust_complex_params = True


class BM4DProfileBM3D(BM4DProfile2D):
    nf = (32, 32, 1)
    search_window_ht = (19, 19, 1)
    search_window_wiener = (19, 19, 1)

class BM4DProfileBM3DComplex(BM4DProfile2D):
    nf = (32, 32, 1)
    search_window_ht = (19, 19, 1)
    search_window_wiener = (19, 19, 1)
    transform_local_ht_name = 'fft'
    transform_local_wiener_name = 'fft'
    adjust_complex_params = True

class BM4DProfileComplex(BM4DProfile):
    transform_local_ht_name = 'fft'
    transform_local_wiener_name = 'fft'
    adjust_complex_params = True

class BM4DProfileLC(BM4DProfile):
    """
    'lc'
    """
    max_stack_size_wiener = 16
    search_window_ht = (4, 4, 4)
    search_window_wiener = search_window_ht

