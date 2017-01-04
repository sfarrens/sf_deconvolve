#  @file psf_gen.py
#
#  PSF GENERATION ROUTINES
#
#  Functions for generating
#  PSFs. Based on work by
#  Yinghao Ge.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from scipy.ndimage import gaussian_filter
from itertools import product
from functions.np_adjust import data2np


##
#  Function that produces a PSF.
#
#  @param[in] shape: 2D Shape of PSF.
#  @param[in] var: 2D Variance of PSF.
#
#  @return FITS image array.
#
#  @exception ValueError for invalid shape length.
#  @exception ValueError for invalid var length.
#
def single_psf(shape, var):

    # Convert inputs to Numpy arrays
    shape = data2np(shape)
    var = data2np(var)

    if len(shape) != 2:
        raise ValueError('Invalid shape length [%d]. 2D shape must have'
                         'length = 2.' % len(shape))
    if len(var) != 2:
        raise ValueError('Invalid var length [%d]. 2D variance must have'
                         'length = 2.' % len(var))

    # Single pixel point
    psf = np.zeros(shape)
    psf[zip(shape / 2)] = 1

    # Gaussian filter with given variance
    psf = gaussian_filter(psf, var)

    return psf / psf.sum()


##
#  Function that produces a pixel
#  varying PSF.
#
#  @param[in] shape: 2D Shape of image.
#  @param[in] var: 2D Variance of PSF field.
#  @param[in] psf_threshold: PSF n-sigma treshold.
#
#  @return FITS image array.
#
def pixel_var_psf(shape, var, psf_threshold=6):

    shape = data2np(shape, 'float')
    var = data2np(var, 'float')

    if len(shape) != 2:
        raise ValueError('Invalid shape length [%d]. 2D shape must have'
                         'length = 2.' % len(shape))

    if len(var) != 2:
        raise ValueError('Invalid var length [%d]. 2D variance must have'
                         'length = 2.' % len(var))

    # Define variance in x and y axes.
    var_2d_array = ((var / (np.floor(shape / 100.) * 100. + 100 *
                     (shape < 100))) * np.array([np.arange(1, x) for x in
                                                 shape + 1]).T).T

    # Define the shape of each pixel PSF.
    psf_shape = psf_threshold * var + 1

    # Generate the PSF cube.
    pv_psf = [single_psf(psf_shape, x).T for x in
              np.array(list(product(*var_2d_array)), dtype='float')]

    return np.array(pv_psf).T
