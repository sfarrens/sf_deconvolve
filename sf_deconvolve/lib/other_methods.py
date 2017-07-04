# -*- coding: utf-8 -*-

"""OTHER METHODS

This module contains methods for calculating alternatives to deconvolution.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 17/01/2017

"""

import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift


def pseudo_inverse(image, kernel, weight=None):
    """Pseudo inverse

    This method calculates the pseudo inverse of the input image for the given
    kernel using FFT

    Parameters
    ----------
    image : np.ndarray
        Input image, 2D array
    kernel : np.ndarray
        Input kernel, 2D array
    weight : np.ndarray, optional
        Optional weights, 2D array

    Returns
    -------
    np.ndarray result of the pseudo inverse

    """

    y_hat = fftshift(fftn(image))
    h_hat = fftshift(fftn(kernel))
    h_hat_star = np.conj(h_hat)

    res = ((h_hat_star * y_hat) / (h_hat_star * h_hat))

    if not isinstance(weight, type(None)):
        res *= weight

    return np.real(fftshift(ifftn(ifftshift(res))))
