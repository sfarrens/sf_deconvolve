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


##
#  Function that produces a PSF.
#
#  @param[in] shape: 2D Shape of PSF.
#  @param[in] var: 2D Variance of PSF.
#
#  @return A single PSF.
#
#  @exception ValueError for invalid shape length.
#  @exception ValueError for invalid var length.
#
def single_psf(shape, var):

    # Convert inputs to Numpy arrays.
    shape = np.array(shape)
    var = np.array(var)

    if shape.size is not 2:
        raise ValueError('Invalid number of shape dimensions [%d]. The shape '
                         'must be 2D.' % shape.size)

    if var.size is not 2:
        raise ValueError('Invalid number of dimensions for variance [%d]. '
                         'Variance (var) must be 2D.' % var.size)

    # Place single pixel point a the centre of a matrix of zeros.
    psf = np.zeros(shape)
    psf[zip(shape / 2)] = 1

    # Add Gaussian filter with given variance.
    psf = gaussian_filter(psf, var)

    # Return normalised PSF.
    return psf / psf.sum()


##
#  Function that produces a pixel varying PSF. i.e. a stack of distinct PSFs
#  for each pixel in the image.
#
#  @param[in] shape: 2D or 3D Shape of image(s).
#  @param[in] var: 2D Variance of PSF field.
#  @param[in] psf_threshold: PSF radius. Defines shape of the individual PSFs.
#
#  @return Stack of PSFs.
#
#  @exception ValueError for invalid shape dimensions.
#  @exception ValueError for invalid var dimensions.
#
def pixel_var_psf(shape, var, psf_radius=6):

    shape = np.array(shape, dtype='float')
    var = np.array(var, dtype='float')

    if shape.size is not 2:
        raise ValueError('Invalid number of shape dimensions [%d]. The shape '
                         'must have either 2 or 3 dimensions.' % shape.size)

    if var.size is not 2:
        raise ValueError('Invalid number of dimensions for variance [%d]. '
                         'Variance (var) must be 2D.' % var.size)

    # Define variance in x and y axes. Notes: This takes the input Variance
    # (var) and divides by the nearest (rounding down) 100 with regards to the
    # image shape. If the shape is less than 100 it rounds up to 100. These
    # values are multiplied by ranges in the x and y directions for the
    # corresponding number of pixels.
    var_2d_array = ((var / (np.floor(shape / 100.) * 100. + 100 *
                     (shape < 100))) * np.array([np.arange(1, x) for x in
                                                 shape + 1]).T).T

    # Define the shape of each pixel PSF.
    psf_shape = psf_radius * var + 1

    # Generate the PSF cube.f
    pv_psf = [single_psf(psf_shape, x).T for x in
              np.array(list(product(*var_2d_array)), dtype='float')]

    return np.array(pv_psf)
