#  @file transform.py
#
#  CONVOLUTION ROUTINES
#
#  Functions for convolving
#  data. Based on work by
#  Yinghao Ge.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from scipy.signal import fftconvolve
from functions.np_adjust import data2np, pad2d


##
#  Function that convolves an image with
#  a PSF.
#
#  @param[in] image: Input image data.
#  @param[in] psf: PSF.
#
#  @return Convolved image.
#
def psf_convolve(image, psf):

    # Function to select slices of an image.
    def select_slices(image_shape, sub_shape):
        ranges = np.array([np.arange(i) for i in image_shape])
        limits = np.array([ranges.T + sub_shape / 2 + 1,
                           ranges.T + 1.5 * sub_shape + 1]).T
        return np.array([np.array([slice(*i), slice(*j)]) for i in limits[0]
                         for j in limits[1]])

    # Function to convolve a PSF with a sub-image.
    def get_convolve(sub_image, psf):
        return np.multiply(sub_image, np.rot90(psf, 2)).sum()

    # Pad image borders by PSF size.
    image_pad = pad2d(image, psf.shape[:2])

    # Get sub-image slices of the padded image.
    slices = select_slices(image.shape, data2np(psf.shape[:2]))

    # Convolve sub-images with PSFs.
    image_conv = np.array([[get_convolve(image_pad[list(x)], y.T)]
                           for x, y in zip(slices, psf.T)])

    return np.reshape(image_conv, image.shape)


##
#  Function that convolves an image with
#  a PSF.
#
#  @param[in] image: Input image data.
#  @param[in] psf_pcs: PSF principal components.
#  @param[in] psf_coef: PSF coefficients.
#
#  @return Convolved image.
#
def pca_convolve(image, psf_pcs, psf_coef):

    return sum([np.multiply(fftconvolve(image, a.T, mode='same'), b.T)
                for a, b in zip(psf_pcs.T, psf_coef.T)])
