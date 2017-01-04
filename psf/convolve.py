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
from scipy.signal import fftconvolve
from astropy.convolution import convolve_fft
from functions.np_adjust import rotate, rotate_stack


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

    # return fftconvolve(data, kernel, mode='same')
    return convolve_fft(data, kernel, boundary='wrap', crop=True)


##
#  Function that convolves an image with a PSF.
#
#  @param[in] data: Input data.
#  @param[in] psf: PSF.
#  @param[in] psf_rot: Option to rotate PSF.
#  @param[in] psf_type: PSF type. ('fixed' or 'obj_var')
#
#  @return Convolved image.
#
#  @exception ValueError for invalid PSF type.
#
def psf_convolve(data, psf, psf_rot=False, psf_type='fixed'):

    # Check input values.
    if psf_type not in ('fixed', 'obj_var'):
        raise ValueError('Invalid PSF type! Options are fixed or obj_var')

    # Rotate the PSF(s) by 180 degrees.
    if psf_rot and psf_type == 'fixed':
        psf = rotate(psf)

    elif psf_rot:
        psf = rotate_stack(psf)

    # Convolve the PSF with the data.
    if psf_type == 'fixed':
        return np.array([convolve(data_i, psf) for data_i in data])

    elif psf_type == 'obj_var':
        return np.array([convolve(data_i, psf_i) for data_i, psf_i in
                        zip(data, psf)])
