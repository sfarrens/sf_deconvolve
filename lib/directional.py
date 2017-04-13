# -*- coding: utf-8 -*-

"""DIRECTIONAL TRANSFORM ROUTINES

This module contains methods for directional transforms data based on work by
Hao Shan and Fred Ngole

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.1

:Date: 07/01/2017

"""

import numpy as np
from creepy.image.convolve import convolve
from creepy.math.matrix import rotate
from creepy.signal.filter import *


def get_dir_filters(shape, angle_num, sigma):
    """Get directional filters

    This method returns the directional transform filters

    Parameters
    ----------
    shape : tuple
        Shape of input data
    angle_num : int
        Nuber of rotation angles
    sigma : float
        Filter width

    Returns
    -------
    np.ndarray 3D array of filter coefficients

    """

    angles = np.arange(angle_num) * np.pi / angle_num

    shift = (shape[0] - 1) / 2

    index_matrix = np.tile(np.arange(shape[0]), (shape[0], 1)).T - shift

    def func(x, y):

        return (Gaussian_filter(rotate(index_matrix, x), y, fourier=True) *
                mex_hat(rotate(index_matrix, x + np.pi / 2), y))

    return np.array([func(angle, sigma) for angle in angles])


def convolve_dir_filters(data, filters):
    """Convolve with directional filters

    This method convolves the input data with the provided filters

    Parameters
    ----------
    data : np.ndarray
        Input data array
    filters : np.ndarray
        3D array of filters

    Returns
    -------
    np.ndarray of convolved data

    """

    return np.array([convolve(data, f) for f in filters])
