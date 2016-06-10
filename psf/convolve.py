#  @file transform.py
#
#  CONVOLUTION ROUTINES
#
#  Functions for convolving data. Based on work by Yinghao Ge and Fred Ngole.
#
#  @author Samuel Farrens
#  @version 2.0
#  @date 2015
#

import numpy as np
from astropy.convolution import convolve_fft
from functions.np_adjust import *
from functions.image import FetchWindows


##
#  Function that convolves the input data with a given kernel using FFT.
#  This is the default convolution used for all routines.
#
#  @param[in] data: Input data.
#  @param[in] kernel: Kernel.
#
#  @return Convolved data.
#
def convolve(data, kernel):

    return convolve_fft(data, kernel, boundary='wrap', crop=True)


##
#  Function that convolves an image with a PSF.
#
#  @param[in] data: Input data.
#  @param[in] psf: PSF.
#  @param[in] psf_rot: Option to rotate PSF.
#  @param[in] psf_type: PSF type. ('fixed' or 'obj_var')
#  @param[in] data_format: Data format. ('cube' or 'map')
#
#  @return Convolved image.
#
#  @exception ValueError for invalid PSF type.
#  @exception ValueError for invalid data type.
#
def psf_convolve(data, psf, psf_rot=False, psf_type='fixed',
                 data_format='cube'):

    # Check input values.
    if psf_type not in ('fixed', 'obj_var'):
        raise ValueError('Invalid PSF type! Options are fixed or obj_var')

    if data_format not in ('map', 'cube'):
        raise ValueError('Invalid data type! Options are map or cube')

    if data_format == 'map' and psf_type != 'fixed':
        raise ValueError('Incompatible data and PSF types! PSF must be fixed'
                         'for map data.')

    # Rotate the PSF(s) by 180 degrees.
    if psf_rot and psf_type == 'fixed':
        psf = rotate(psf)

    elif psf_rot:
        psf = rotate_stack(psf)

    # Convolve the PSF with the data.
    if data_format == 'map':

        return convolve(data, psf)

    elif psf_type == 'fixed':
        return np.array([convolve(data_i, psf) for data_i in data])

    elif psf_type == 'obj_var':
        return np.array([convolve(data_i, psf_i) for data_i, psf_i in
                        zip(data, psf)])


##
#  Function that convolves an image with a pixel variant PSF.
#
#  @param[in] image: Input image data.
#  @param[in] psf: Pixel variant PSF.
#
#  @return Convolved image.
#
def psf_var_convolve(image, psf):

    def get_convolve(sub_image, psf):
        return np.sum(sub_image * np.rot90(psf, 2))

    w = FetchWindows(image, psf.shape[-1] / 2, all=True)

    return w.scan(get_convolve, psf, arg_type='list').reshape(image.shape)


##
#  Function that convolves the input data with the principal components of a
#  PSF.
#
#  @param[in] data: Input data.
#  @param[in] psf_pcs: PSF principal components.
#  @param[in] psf_coef: PSF coefficients.
#  @param[in] pcs_rot: Option to rotate principal components.
#
#  @return Convolved data.
#
def pca_convolve(data, psf_pcs, psf_coef, pcs_rot=False):

    if pcs_rot:
        return sum([convolve(data * b, rotate(a)) for a, b in
                   zip(psf_pcs, psf_coef)])

    else:
        return sum([(convolve(data, a) * b) for a, b in
                   zip(psf_pcs, psf_coef)])


##
#  Function that convolves the input data stack with the principal components
#  of a PSF.
#
#  @param[in] data_stack: Input data stack.
#  @param[in] psf_pcs: PSF principal components.
#  @param[in] psf_coef_stack: Stack of PSF coefficients.
#  @param[in] pcs_rot: Option to rotate principal components.
#
#  @return Convolved data stack.
#
def pca_convolve_stack(data_stack, psf_pcs, psf_coef_stack, pcs_rot=False):

    return np.array([pca_convolve(data, psf_pcs, psf_coef, pcs_rot) for
                     data, psf_coef in zip(data_stack, psf_coef_stack)])
