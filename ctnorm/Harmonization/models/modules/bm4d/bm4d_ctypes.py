import ctypes
import numpy as np
from . import profiles
import os
from sys import platform
from platform import machine  # used to distinguish M-series and intel
from typing import Tuple, Union, Optional

"""
Structure types for per-dimension parameters for the binary
"""


class Complex(ctypes.Structure):
    """
    3 integers
    """
    _fields_ = [
        ("re", ctypes.c_float),
        ("im", ctypes.c_float),
    ]


class Vec3i(ctypes.Structure):
    """
    3 integers
    """
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("z", ctypes.c_int)
    ]


class Vec3b(ctypes.Structure):
    """
    3 bools
    """
    _fields_ = [
        ("x", ctypes.c_bool),
        ("y", ctypes.c_bool),
        ("z", ctypes.c_bool)
    ]


class Vec3fp(ctypes.Structure):
    """
    3 pointers to floats
    """
    _fields_ = [
        ("x", ctypes.POINTER(ctypes.c_float)),
        ("y", ctypes.POINTER(ctypes.c_float)),
        ("z", ctypes.POINTER(ctypes.c_float))
    ]


class Vec3cp(ctypes.Structure):
    """
    3 pointers to floats
    """
    _fields_ = [
        ("x", ctypes.POINTER(Complex)),
        ("y", ctypes.POINTER(Complex)),
        ("z", ctypes.POINTER(Complex))
    ]


class Vec3ip(ctypes.Structure):
    """
    3 pointers to ints
    """
    _fields_ = [
        ("x", ctypes.POINTER(ctypes.c_int)),
        ("y", ctypes.POINTER(ctypes.c_int)),
        ("z", ctypes.POINTER(ctypes.c_int))
    ]


"""
Functions to instance the above objects from three variables
"""


def create_vec3i(x: int, y: int, z: int) -> Vec3i:
    v = Vec3i()
    v.x = ctypes.c_int(x)
    v.y = ctypes.c_int(y)
    v.z = ctypes.c_int(z)
    return v


def create_vec3i_arr(arr3) -> Vec3i:
    v = Vec3i()
    v.x = ctypes.c_int(arr3[0])
    v.y = ctypes.c_int(arr3[1])
    v.z = ctypes.c_int(arr3[2])
    return v


def create_vec3fp(x: ctypes.POINTER(ctypes.c_float),
                  y: ctypes.POINTER(ctypes.c_float),
                  z: ctypes.POINTER(ctypes.c_float)) -> Vec3fp:
    v = Vec3fp()
    v.x = x
    v.y = y
    v.z = z
    return v


def create_vec3cp(x: ctypes.POINTER(Complex),
                  y: ctypes.POINTER(Complex),
                  z: ctypes.POINTER(Complex)) -> Vec3cp:
    v = Vec3cp()
    v.x = x
    v.y = y
    v.z = z
    return v


def create_vec3ip(x: ctypes.POINTER(ctypes.c_int),
                  y: ctypes.POINTER(ctypes.c_int),
                  z: ctypes.POINTER(ctypes.c_int)) -> Vec3ip:
    v = Vec3ip()
    v.x = x
    v.y = y
    v.z = z
    return v


def create_vec3b(x: bool, y: bool, z: bool) -> Vec3b:
    v = Vec3b()
    v.x = x
    v.y = y
    v.z = z
    return v


class Parameters(ctypes.Structure):
    """
    All parameters to be passed to the binary.
    """
    _fields_ = [
        ("blockSize", Vec3i),
        ("psdSizeSmall", Vec3i),
        ("cutPSD", ctypes.c_bool),
        ("maxStackSize", ctypes.c_int),
        ("kin", ctypes.c_int),
        ("lambdaSq", ctypes.c_float),
        ("muSq", ctypes.c_float),
        ("gamma", ctypes.c_float),
        ("searchWindowSize", Vec3i),
        ("imageSize", Vec3i),
        ("psdSizeOrig", Vec3i),
        ("step", Vec3i),
        ("minStackSize", ctypes.c_int),
        ("blockMatchCap", ctypes.c_float),
        ("splitBlockExtraction", Vec3b),
        ("blockShiftSize", Vec3i),
        ("blockShifts", Vec3ip),
        ("muSqDC", ctypes.c_float),
        ("sharpen", ctypes.c_bool),
        ("sharpenAggregation", ctypes.c_bool),
        ("sharpenAlphaInv", ctypes.c_float),
        ("sharpenAlphaInv3D", ctypes.c_float),
        ("numThreads", ctypes.c_int),
    ]


class Transforms(ctypes.Structure):
    """
    All transforms to be passed to the binary
    """
    _fields_ = [
        ("localTransforms", Vec3fp),
        ("localInverseTransforms", Vec3fp),
        ("transformNLs", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
        ("invTransformNLs", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
        ("windowingFunction", ctypes.POINTER(ctypes.c_float)),
    ]


class TransformsComplex(ctypes.Structure):
    """
    All transforms to be passed to the binary
    """
    _fields_ = [
        ("localTransforms", Vec3cp),
        ("localInverseTransforms", Vec3cp),
        ("transformNLs", ctypes.POINTER(ctypes.POINTER(Complex))),
        ("invTransformNLs", ctypes.POINTER(ctypes.POINTER(Complex))),
        ("windowingFunction", ctypes.POINTER(ctypes.c_float)),
    ]


class BlockMatchStorage(ctypes.Structure):
    """
    Pre-computed block-matches or storage for new block-matches
    """
    _fields_ = [
        ("maxStackSize", ctypes.c_int),
        ("stackSizes", ctypes.POINTER(ctypes.c_int)),
        ("blockPositions", ctypes.POINTER(Vec3i)),
        ("shiftedPositions", ctypes.POINTER(Vec3i)),
        ("currentStack", ctypes.c_int),
        ("status", ctypes.c_int)
    ]

    def get_python_structs(self):
        """
        Get contents in python-readable format.
        returns: stack sizes, block positions, in-group shifted positions
        """
        match_positions = []
        match_positions_shift = []
        total_size = self.currentStack * self.maxStackSize
        group_sizes = []

        for i in range(0, total_size):
            group_sizes.append(self.stackSizes[i])

        for i in range(0, total_size):
            match_positions.append(
                [self.blockPositions[i].x, self.blockPositions[i].y, self.blockPositions[i].z])
            match_positions_shift.append(
                [self.shiftedPositions[i].x, self.shiftedPositions[i].y, self.shiftedPositions[i].z])

        return group_sizes, match_positions, match_positions_shift


# Argument types for HT function call
ARGTYPES_THR = [ctypes.POINTER(ctypes.c_float),  # sourceImage
                ctypes.POINTER(ctypes.c_float),  # PSD
                Parameters,
                Transforms,
                ctypes.POINTER(ctypes.c_float),  # estimate
                ctypes.POINTER(BlockMatchStorage)
                ]

# Argument types for Wiener function call
ARGTYPES_WIE = [ctypes.POINTER(ctypes.c_float),  # sourceImage
                ctypes.POINTER(ctypes.c_float),  # PSD
                Parameters,
                Transforms,
                ctypes.POINTER(ctypes.c_float),  # reference
                ctypes.POINTER(ctypes.c_float),  # estimate
                ctypes.POINTER(BlockMatchStorage)
                ]

# Argument types for complex HT function call
ARGTYPES_THR_COMPLEX = \
    [np.ctypeslib.ndpointer(np.complex64, ndim=1, flags='C'),  # sourceImage
     ctypes.POINTER(ctypes.c_float),  # PSD
     Parameters,
     TransformsComplex,
     np.ctypeslib.ndpointer(np.complex64, ndim=1, flags='C'),  # estimate
     ctypes.POINTER(BlockMatchStorage)
     ]

# Argument types for complex Wiener function call
ARGTYPES_WIE_COMPLEX = \
    [np.ctypeslib.ndpointer(np.complex64, ndim=1, flags='C'),  # sourceImage
     ctypes.POINTER(ctypes.c_float),  # PSD
     Parameters,
     TransformsComplex,
     np.ctypeslib.ndpointer(np.complex64, ndim=1, flags='C'),  # reference
     np.ctypeslib.ndpointer(np.complex64, ndim=1, flags='C'),  # estimate
     ctypes.POINTER(BlockMatchStorage)
     ]


def get_stack_storage_size(z_sz, step_sz):
    return (z_sz[0] // step_sz[0] + 1) * (z_sz[1] // step_sz[1] + 1) * (z_sz[2] // step_sz[2] + 1)


def conv_to_array(pyarr, ctype=ctypes.c_float):
    """
    Convert Python array to C array
    :param pyarr: python array
    :param ctype: type of resulting element
    :return: C array (pointer)
    """
    return (ctype * len(pyarr))(*pyarr)


def flatten_transf(transf_dict: dict, dtype=np.float32, cdtype=ctypes.c_float):  # -> ctypes.POINTER(ctypes.POINTER()):
    """
    Flatten the stack transforms computed by __get_transforms to format used by the binary.
    :param transf_dict: a stack transform dict returned by __get_transforms
    :return: 1-d list of transforms
    """
    total_list = [None] * pow(2, len(transf_dict))
    for key in sorted(transf_dict):
        if dtype == np.complex64:
            flattened = convert_to_complex_array(np.ascontiguousarray(transf_dict[key].T.flatten()))
        else:
            flattened = conv_to_array(np.ascontiguousarray(transf_dict[key].T.flatten(), dtype=dtype))
        total_list[key] = flattened

    total_list = conv_to_array(total_list, ctype=ctypes.POINTER(cdtype))
    return total_list


def get_dll_name() -> str:
    """
    Get the correct DLL name based on OS
    :return: library name to load
    """
    path = os.path.dirname(__file__)
    if platform == "darwin":
        if machine() == "arm64":
            dll_name = "libbm4d_mac_arm"
        else:
            dll_name = "libbm4d_mac"
    elif platform == "win32":
        dll_name = "libbm4d_win"
    elif platform == "linux":
        dll_name = "libbm4d"
    else:
        # Presume linux anyway ...
        dll_name = "libbm4d"

    return os.path.join(path, dll_name) + ".so"


# Find and create ctypes DLL object
dll = ctypes.CDLL(get_dll_name())

func_ht = dll.bm4dInterfaceHT
func_ht.argtypes = ARGTYPES_THR

func_wie = dll.bm4dInterfaceWie
func_wie.argtypes = ARGTYPES_WIE


# func_ht_complex = dll.bm4dInterfaceComplexHT
# func_ht_complex.argtypes = ARGTYPES_THR_COMPLEX

# func_wie_complex = dll.bm4dInterfaceComplexWie
# func_wie_complex.argtypes = ARGTYPES_WIE_COMPLEX

def get_blockmatch_storage(blockmatches: Union[BlockMatchStorage, bool], stack_size: int, block_count: int):
    if isinstance(blockmatches, BlockMatchStorage):
        return blockmatches
    elif blockmatches:
        storage = BlockMatchStorage()
        storage.status = ctypes.c_int(0)

        ArrTypePos = Vec3i * (block_count * stack_size)
        ArrTypeSize = ctypes.c_int * block_count

        block_positions = ArrTypePos()
        shift_positions = ArrTypePos()

        storage.blockPositions = block_positions
        storage.shiftedPositions = shift_positions
        storage.stackSizes = ArrTypeSize()
        storage.maxStackSize = stack_size
        return storage
    else:
        return None


def get_params_ht(pro: profiles.BM4DProfile, z_shape: tuple, psd_shape: tuple, z_max: float, refilter: bool,
                  qshifts: Tuple[np.ndarray, np.ndarray, np.ndarray]) \
        -> Parameters:
    """
    Convert parameters from BM4DProfile to C-input struct for hard-thresholding
    :param pro: profile object
    :param z_shape: shape of z
    :param psd_shape: shape of psd
    :param z_max: maximum value of z to scale match thresholds
    :param refilter: if true, use refiltering parameters
    :param qshifts: tuple of block-shift loops
    :return: parameter object
    """
    simple_shape = (pro.nf[0] * pro.nf[1] * pro.nf[2] == pro.nf[0] ** 3) or (
                pro.nf[0] * pro.nf[1] == pro.nf[0] ** 2 and pro.nf[2] == 1)
    params = Parameters()
    params.blockSize = create_vec3i(pro.bs_ht[0], pro.bs_ht[1], pro.bs_ht[2])
    params.psdSizeSmall = create_vec3i(pro.nf[0], pro.nf[1], pro.nf[2])
    params.cutPSD = ctypes.c_bool(simple_shape)
    params.maxStackSize = ctypes.c_int(pro.max_stack_size_ht)
    params.kin = ctypes.c_int(pro.k)
    params.lambdaSq = ctypes.c_float(pro.lambda_thr * pro.lambda_thr) if not refilter else \
        ctypes.c_float(pro.lambda_thr_re * pro.lambda_thr_re)
    params.muSq = ctypes.c_float(pro.mu2) if not refilter else \
        ctypes.c_float(pro.mu2_re)
    params.muSqDC = params.muSq
    params.gamma = ctypes.c_float(pro.gamma)
    params.searchWindowSize = create_vec3i(pro.search_window_ht[0], pro.search_window_ht[1], pro.search_window_ht[2])
    params.imageSize = create_vec3i(z_shape[1], z_shape[0], z_shape[2])
    params.psdSizeOrig = create_vec3i(psd_shape[1], psd_shape[0], psd_shape[2])
    params.step = create_vec3i(pro.step_ht[0], pro.step_ht[1], pro.step_ht[2])
    params.minStackSize = ctypes.c_int(pro.min_3d_size_ht)
    params.blockMatchCap = ctypes.c_float(pro.tau_match * np.abs(z_max ** 2))
    params.splitBlockExtraction = create_vec3b(pro.split_block_extraction[0], pro.split_block_extraction[1],
                                               pro.split_block_extraction[2])

    q_x = conv_to_array(np.ascontiguousarray(qshifts[0].flatten(), dtype=np.int32), ctypes.c_int)
    q_y = conv_to_array(np.ascontiguousarray(qshifts[1].flatten(), dtype=np.int32), ctypes.c_int)
    q_z = conv_to_array(np.ascontiguousarray(qshifts[2].flatten(), dtype=np.int32), ctypes.c_int)
    params.blockShiftSize = create_vec3i(qshifts[0].size, qshifts[1].size, qshifts[2].size)
    params.blockShifts = create_vec3ip(q_x, q_y, q_z)
    params.sharpen = ctypes.c_bool(pro.sharpen_alpha != 1 or pro.sharpen_alpha_3d != 1)
    params.sharpenAggregation = params.sharpen

    params.sharpenAlphaInv = ctypes.c_float(1 / pro.sharpen_alpha)
    if z_shape[2] > 1:
        params.sharpenAlphaInv3D = ctypes.c_float(1 / pro.sharpen_alpha_3d)
    else:  # Don't use this parameter, if 2-D
        params.sharpenAlphaInv3D = params.sharpenAlphaInv

    params.numThreads = pro.num_threads
    return params


def get_params_wie(pro: profiles.BM4DProfile, z_shape: tuple, psd_shape: tuple, z_max: float, refilter: bool,
                   qshifts: Tuple[np.ndarray, np.ndarray, np.ndarray]) \
        -> Parameters:
    """
    Convert parameters from BM4DProfile to C-input struct for Wiener filtering
    :param pro: profile object
    :param z_shape: shape of z
    :param psd_shape: shape of psd
    :param z_max: maximum value of z to scale match thresholds
    :param refilter: if true, use refiltering parameters
    :param qshifts: tuple of block-shift loops
    :return: parameter object
    """
    simple_shape = (pro.nf[0] * pro.nf[1] * pro.nf[2] == pro.nf[0] ** 3) or (
                pro.nf[0] * pro.nf[1] == pro.nf[0] ** 2 and pro.nf[2] == 1)
    params = Parameters()
    params.blockSize = create_vec3i(pro.bs_wiener[0], pro.bs_wiener[1], pro.bs_wiener[2])
    params.psdSizeSmall = create_vec3i(pro.nf[0], pro.nf[1], pro.nf[2])
    params.cutPSD = ctypes.c_bool(simple_shape)
    params.maxStackSize = ctypes.c_int(pro.max_stack_size_wiener)
    params.kin = ctypes.c_int(pro.k)
    params.lambdaSq = ctypes.c_float(pro.lambda_thr * pro.lambda_thr) if not refilter else \
        ctypes.c_float(pro.lambda_thr_re * pro.lambda_thr_re)
    params.muSq = ctypes.c_float(pro.mu2) if not refilter else \
        ctypes.c_float(pro.mu2_re)
    params.muSqDC = params.muSq
    params.gamma = ctypes.c_float(pro.gamma)
    params.searchWindowSize = create_vec3i(pro.search_window_wiener[0], pro.search_window_wiener[1],
                                           pro.search_window_wiener[2])
    params.imageSize = create_vec3i(z_shape[1], z_shape[0], z_shape[2])
    params.psdSizeOrig = create_vec3i(psd_shape[1], psd_shape[0], psd_shape[2])
    params.step = create_vec3i(pro.step_wiener[0], pro.step_wiener[1], pro.step_wiener[2])
    params.minStackSize = ctypes.c_int(pro.min_3d_size_wiener)
    params.blockMatchCap = ctypes.c_float(pro.tau_match_wiener * np.abs(z_max ** 2))
    params.splitBlockExtraction = create_vec3b(pro.split_block_extraction[0], pro.split_block_extraction[1],
                                               pro.split_block_extraction[2])

    q_x = conv_to_array(np.ascontiguousarray(qshifts[0].flatten(), dtype=np.int32), ctypes.c_int)
    q_y = conv_to_array(np.ascontiguousarray(qshifts[1].flatten(), dtype=np.int32), ctypes.c_int)
    q_z = conv_to_array(np.ascontiguousarray(qshifts[2].flatten(), dtype=np.int32), ctypes.c_int)
    params.blockShiftSize = create_vec3i(qshifts[0].size, qshifts[1].size, qshifts[2].size)
    params.blockShifts = create_vec3ip(q_x, q_y, q_z)
    params.sharpen = ctypes.c_bool(False)
    params.sharpenAggregation = params.sharpen
    params.numThreads = pro.num_threads

    return params


def convert_to_complex_array(a: np.ndarray):
    out = []
    for i in range(a.size):
        c = Complex()
        c.re = ctypes.c_float(np.real(a[i]))
        c.im = ctypes.c_float(np.imag(a[i]))
        out.append(c)
    return conv_to_array(out, Complex)


def get_transforms(t_forward: Tuple[np.ndarray], t_inverse: Tuple[np.ndarray],
                   hadper_trans_single_den: Union[list, None], inverse_hadper_trans_single_den: Union[list, None],
                   wwin: np.ndarray) -> Transforms:
    """
    Convert transforms to be passed to C
    :param t_forward: forward block transforms
    :param t_inverse: inverse block transforms
    :param hadper_trans_single_den: stack transforms; None -> haar
    :param inverse_hadper_trans_single_den: inverse stack transforms, None -> haar
    :param wwin: windowing function
    :param dtype: transforms type
    :return: transforms C struct
    """
    if t_forward is not None:

        t_inv_nl_flat = flatten_transf(hadper_trans_single_den, np.float32, ctypes.c_float)
        t_fwd_nl_flat = flatten_transf(inverse_hadper_trans_single_den, np.float32, ctypes.c_float)
        c_t_fwd_nl_flat = t_fwd_nl_flat
        c_t_inv_nl_flat = t_inv_nl_flat

        c_wwin = conv_to_array(np.ascontiguousarray(wwin.T.flatten(), dtype=np.float32), ctype=ctypes.c_float)

        transforms = Transforms()

        c_t_fwd_x = conv_to_array(np.ascontiguousarray(t_forward[0].flatten(), dtype=np.float32))
        c_t_fwd_y = conv_to_array(np.ascontiguousarray(t_forward[1].flatten(), dtype=np.float32))
        c_t_fwd_z = conv_to_array(np.ascontiguousarray(t_forward[2].flatten(), dtype=np.float32))

        c_t_inv_x = conv_to_array(np.ascontiguousarray(t_inverse[0].flatten(), dtype=np.float32))
        c_t_inv_y = conv_to_array(np.ascontiguousarray(t_inverse[1].flatten(), dtype=np.float32))
        c_t_inv_z = conv_to_array(np.ascontiguousarray(t_inverse[2].flatten(), dtype=np.float32))

        transforms.localTransforms = create_vec3fp(c_t_fwd_x, c_t_fwd_y, c_t_fwd_z)
        transforms.localInverseTransforms = create_vec3fp(c_t_inv_x, c_t_inv_y, c_t_inv_z)

        transforms.transformNLs = None if len(hadper_trans_single_den) == 0 else c_t_fwd_nl_flat
        transforms.invTransformNLs = None if len(inverse_hadper_trans_single_den) == 0 else c_t_inv_nl_flat
        transforms.windowingFunction = c_wwin
    else:
        transforms = Transforms()
        transforms.localTransforms = create_vec3fp(None, None, None)

    return transforms


def get_transforms_complex(t_forward: Tuple[np.ndarray], t_inverse: Tuple[np.ndarray],
                           hadper_trans_single_den: Union[list, None],
                           inverse_hadper_trans_single_den: Union[list, None],
                           wwin: np.ndarray) -> Transforms:
    """
    Convert transforms to be passed to C
    :param t_forward: forward block transforms
    :param t_inverse: inverse block transforms
    :param hadper_trans_single_den: stack transforms; None -> haar
    :param inverse_hadper_trans_single_den: inverse stack transforms, None -> haar
    :param wwin: windowing function
    :param dtype: transforms type
    :return: transforms C struct
    """
    if t_forward is not None:

        t_inv_nl_flat = flatten_transf(hadper_trans_single_den, np.complex64, Complex)
        t_fwd_nl_flat = flatten_transf(inverse_hadper_trans_single_den, np.complex64, Complex)
        c_t_fwd_nl_flat = t_fwd_nl_flat
        c_t_inv_nl_flat = t_inv_nl_flat

        c_wwin = conv_to_array(np.ascontiguousarray(wwin.T.flatten(), dtype=np.float32), ctype=ctypes.c_float)

        transforms = TransformsComplex()

        c_t_fwd_x = convert_to_complex_array(np.ascontiguousarray(t_forward[0].flatten(), dtype=np.complex64))
        c_t_fwd_y = convert_to_complex_array(np.ascontiguousarray(t_forward[1].flatten(), dtype=np.complex64))
        c_t_fwd_z = convert_to_complex_array(np.ascontiguousarray(t_forward[2].flatten(), dtype=np.complex64))

        c_t_inv_x = convert_to_complex_array(np.ascontiguousarray(t_inverse[0].flatten(), dtype=np.complex64))
        c_t_inv_y = convert_to_complex_array(np.ascontiguousarray(t_inverse[1].flatten(), dtype=np.complex64))
        c_t_inv_z = convert_to_complex_array(np.ascontiguousarray(t_inverse[2].flatten(), dtype=np.complex64))

        transforms.localTransforms = create_vec3cp(c_t_fwd_x, c_t_fwd_y, c_t_fwd_z)
        transforms.localInverseTransforms = create_vec3cp(c_t_inv_x, c_t_inv_y, c_t_inv_z)

        transforms.transformNLs = None if len(hadper_trans_single_den) == 0 else c_t_fwd_nl_flat
        transforms.invTransformNLs = None if len(inverse_hadper_trans_single_den) == 0 else c_t_inv_nl_flat
        transforms.windowingFunction = c_wwin
    else:
        transforms = TransformsComplex()
        transforms.localTransforms = create_vec3cp(None, None, None)

    return transforms


def bm4d_ht(z: np.ndarray, psd: np.ndarray, pro: profiles.BM4DProfile,
            t_forward: Tuple[np.ndarray], t_inverse: Tuple[np.ndarray],
            qshifts: Tuple[np.ndarray, np.ndarray, np.ndarray],
            hadper_trans_single_den: Union[list, None],
            inverse_hadper_trans_single_den: Union[list, None],
            wwin3d: np.ndarray, refilter: bool = False,
            blockmatches=False) \
        -> Tuple[np.ndarray, Optional[BlockMatchStorage]]:
    """
    Perform hard-thresholding through the BM4D binary.
    :param z: noisy image
    :param psd: noise PSD
    :param pro: profile object for parameters
    :param t_forward: forward transforms
    :param t_inverse: inverse transforms
    :param qshifts: block shift data
    :param hadper_trans_single_den: stack forward transforms by size
    :param inverse_hadper_trans_single_den: stack inverse transforms by size
    :param wwin3d: windowing function for aggregation
    :param refilter: use refiltering parameters
    :param blockmatches: block-matching data, or True to collect, False to ignore
    :return: hard-thresholded estimate
    """
    z_shape = z.shape
    psd_shape = psd.shape

    params = get_params_ht(pro, z_shape, psd_shape, np.max(z) - np.min(z), refilter, qshifts)
    transforms = get_transforms(t_forward, t_inverse, hadper_trans_single_den,
                                inverse_hadper_trans_single_den, wwin3d)

    stack_storage_size = get_stack_storage_size(z.shape, pro.step_ht)

    matchtables = get_blockmatch_storage(blockmatches, pro.max_stack_size_ht, stack_storage_size)

    z = np.transpose(z, [2, 0, 1])
    psd = np.transpose(psd, [2, 0, 1])
    res = np.zeros(z_shape)

    c_z = conv_to_array(np.ascontiguousarray(z.flatten(), dtype=np.float32), ctype=ctypes.c_float)
    c_psd = conv_to_array(np.ascontiguousarray(psd.flatten(), dtype=np.float32), ctype=ctypes.c_float)
    c_est = conv_to_array(np.ascontiguousarray(res.flatten(), dtype=np.float32), ctype=ctypes.c_float)

    func_ht(c_z, c_psd, params, transforms, c_est,
            ctypes.byref(matchtables) if matchtables is not None else matchtables)

    for i in range(z_shape[0]):
        for j in range(z_shape[1]):
            for k in range(z_shape[2]):
                res[i, j, k] = c_est[k * z_shape[0] * z_shape[1] +
                                     i * z_shape[1] + j]

    bm_out = None
    if isinstance(blockmatches, bool) and blockmatches:
        bm_out = matchtables
        bm_out.status = ctypes.c_int(1)

    return res, bm_out


def bm4d_ht_complex(z: np.ndarray, psd: np.ndarray, pro: profiles.BM4DProfile,
                    t_forward: Tuple[np.ndarray], t_inverse: Tuple[np.ndarray],
                    qshifts: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    hadper_trans_single_den: Union[list, None],
                    inverse_hadper_trans_single_den: Union[list, None],
                    wwin3d: np.ndarray, refilter: bool = False,
                    blockmatches=False) \
        -> Tuple[np.ndarray, Optional[BlockMatchStorage]]:
    """
    Perform hard-thresholding through the BM4D binary.
    :param z: noisy image
    :param psd: noise PSD
    :param pro: profile object for parameters
    :param t_forward: forward transforms
    :param t_inverse: inverse transforms
    :param qshifts: block shift data
    :param hadper_trans_single_den: stack forward transforms by size
    :param inverse_hadper_trans_single_den: stack inverse transforms by size
    :param wwin3d: windowing function for aggregation
    :param refilter: use refiltering parameters
    :param blockmatches: block-matching data, or True to collect, False to ignore
    :return: hard-thresholded estimate
    """
    z_shape = z.shape
    psd_shape = psd.shape

    params = get_params_ht(pro, z_shape, psd_shape, np.max(z) - np.min(z), refilter, qshifts)
    transforms = get_transforms_complex(t_forward, t_inverse, hadper_trans_single_den,
                                        inverse_hadper_trans_single_den, wwin3d)
    params.cutPSD = ctypes.c_bool(False)  # Only guaranteed symmetric with real transforms

    stack_storage_size = get_stack_storage_size(z.shape, pro.step_ht)

    matchtables = get_blockmatch_storage(blockmatches, pro.max_stack_size_ht, stack_storage_size)

    z = np.transpose(z, [2, 0, 1])
    psd = np.transpose(psd, [2, 0, 1])
    res = np.zeros(z_shape, dtype=np.complex64)

    c_z = (np.ascontiguousarray(z.flatten(), dtype=np.complex64))
    c_psd = conv_to_array(np.ascontiguousarray(psd.flatten(), dtype=np.float), ctype=ctypes.c_float)
    c_est = np.ascontiguousarray(res.flatten(), dtype=np.complex64)
    if pro.adjust_complex_params:
        params.lambdaSq = lambda_convert(np.sqrt(params.lambdaSq)) ** 2

    func_ht_complex(c_z, c_psd, params, transforms, c_est,
                    ctypes.byref(matchtables) if matchtables is not None else matchtables)

    for i in range(z_shape[0]):
        for j in range(z_shape[1]):
            for k in range(z_shape[2]):
                res[i, j, k] = c_est[k * z_shape[0] * z_shape[1] +
                                     i * z_shape[1] + j]

    bm_out = None
    if isinstance(blockmatches, bool) and blockmatches:
        bm_out = matchtables
        bm_out.status = ctypes.c_int(1)


    return res, bm_out


def bm4d_wie(z: np.ndarray, psd: np.ndarray, pro: profiles.BM4DProfile,
             t_forward: Tuple[np.ndarray], t_inverse: Tuple[np.ndarray],
             qshifts: Tuple[np.ndarray, np.ndarray, np.ndarray],
             hadper_trans_single_den: Union[list, None],
             inverse_hadper_trans_single_den: Union[list, None], wwin3d: np.ndarray,
             ref: np.ndarray, refilter: bool = False,
             blockmatches=False) \
        -> Tuple[np.ndarray, Optional[BlockMatchStorage]]:
    """
    Perform Wiener filtering through the BM4D binary.
    :param z: noisy image
    :param psd: noise PSD
    :param pro: profile object for parameters
    :param t_forward: forward transforms
    :param t_inverse: inverse transforms
    :param qshifts: block shift data
    :param hadper_trans_single_den: stack forward transforms by size
    :param inverse_hadper_trans_single_den: stack inverse transforms by size
    :param wwin3d: windowing function for aggregation
    :param ref: reference signal same size as z (usually HT estimate)
    :param refilter: use refiltering parameters
    :param blockmatches: block-matching data, or True to collect, False to ignore

    :return: Wiener estimate
    """

    z_shape = z.shape
    psd_shape = psd.shape

    params = get_params_wie(pro, z_shape, psd_shape, np.max(ref) - np.min(ref), refilter, qshifts)
    transforms = get_transforms(t_forward, t_inverse, hadper_trans_single_den, inverse_hadper_trans_single_den, wwin3d)

    z = np.transpose(z, [2, 0, 1])
    psd = np.transpose(psd, [2, 0, 1])
    ref = np.transpose(ref, [2, 0, 1])

    res = np.zeros(z_shape)

    c_z = conv_to_array(np.ascontiguousarray(z.flatten(), dtype=np.float32), ctype=ctypes.c_float)
    c_psd = conv_to_array(np.ascontiguousarray(psd.flatten(), dtype=np.float32), ctype=ctypes.c_float)
    c_est = conv_to_array(np.ascontiguousarray(res.flatten(), dtype=np.float32), ctype=ctypes.c_float)
    c_ref = conv_to_array(np.ascontiguousarray(ref.flatten(), dtype=np.float32), ctype=ctypes.c_float)

    stack_storage_size = get_stack_storage_size(z.shape, pro.step_wiener)

    matchtables = get_blockmatch_storage(blockmatches, pro.max_stack_size_wiener, stack_storage_size)

    func_wie(c_z, c_psd, params, transforms, c_ref, c_est,
             ctypes.byref(matchtables) if matchtables is not None else matchtables)

    for i in range(z_shape[0]):
        for j in range(z_shape[1]):
            for k in range(z_shape[2]):
                res[i, j, k] = c_est[k * z_shape[0] * z_shape[1] +
                                     i * z_shape[1] + j]

    bm_out = None
    if isinstance(blockmatches, bool) and blockmatches:
        bm_out = matchtables
        bm_out.status = ctypes.c_int(1)


    return res, bm_out


def bm4d_wie_complex(z: np.ndarray, psd: np.ndarray, pro: profiles.BM4DProfile,
                     t_forward: Tuple[np.ndarray], t_inverse: Tuple[np.ndarray],
                     qshifts: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     hadper_trans_single_den: Union[list, None],
                     inverse_hadper_trans_single_den: Union[list, None], wwin3d: np.ndarray,
                     ref: np.ndarray, refilter: bool = False,
                     blockmatches=False) \
        -> Tuple[np.ndarray, Optional[BlockMatchStorage]]:
    """
    Perform Wiener filtering through the BM4D binary.
    :param z: noisy image
    :param psd: noise PSD
    :param pro: profile object for parameters
    :param t_forward: forward transforms
    :param t_inverse: inverse transforms
    :param qshifts: block shift data
    :param hadper_trans_single_den: stack forward transforms by size
    :param inverse_hadper_trans_single_den: stack inverse transforms by size
    :param wwin3d: windowing function for aggregation
    :param ref: reference signal same size as z (usually HT estimate)
    :param refilter: use refiltering parameters
    :param blockmatches: block-matching data, or True to collect, False to ignore

    :return: Wiener estimate
    """

    z_shape = z.shape
    psd_shape = psd.shape

    params = get_params_wie(pro, z_shape, psd_shape, np.max(ref) - np.min(ref), refilter, qshifts)
    transforms = get_transforms_complex(t_forward, t_inverse, hadper_trans_single_den, inverse_hadper_trans_single_den,
                                        wwin3d)
    params.cutPSD = ctypes.c_bool(False)  # Only guaranteed symmetric with real transforms

    z = np.transpose(z, [2, 0, 1])
    psd = np.transpose(psd, [2, 0, 1])
    ref = np.transpose(ref, [2, 0, 1])

    res = np.zeros(z_shape, dtype=np.complex64)

    c_z = (np.ascontiguousarray(z.flatten(), dtype=np.complex64))
    c_psd = conv_to_array(np.ascontiguousarray(psd.flatten(), dtype=np.float), ctype=ctypes.c_float)
    c_est = np.ascontiguousarray(res.flatten(), dtype=np.complex64)
    c_ref = np.ascontiguousarray(ref.flatten(), dtype=np.complex64)

    stack_storage_size = get_stack_storage_size(z.shape, pro.step_wiener)

    matchtables = get_blockmatch_storage(blockmatches, pro.max_stack_size_wiener, stack_storage_size)

    func_wie_complex(c_z, c_psd, params, transforms, c_ref, c_est,
                     ctypes.byref(matchtables) if matchtables is not None else matchtables)

    for i in range(z_shape[0]):
        for j in range(z_shape[1]):
            for k in range(z_shape[2]):
                res[i, j, k] = c_est[k * z_shape[0] * z_shape[1] +
                                     i * z_shape[1] + j]

    bm_out = None
    if isinstance(blockmatches, bool) and blockmatches:
        bm_out = matchtables
        bm_out.status = ctypes.c_int(1)


    return res, bm_out
