#  @file directional.py
#
#  DIRECTIONAL TRANSFORM ROUTINES
#
#  Functions for transforming data based on work by Hao Shan and Fred Ngole.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2016
#

import numpy as np
from convolve import convolve
from functions.matrix import rotate
from functions.signal import *


##
#  Function that produces directional filters for the given number of angles
#  and scales.
#
#  @param[in] data: 2D Input array.
#  @param[in] filters: Wavelet filters.
#  @param[in] filter_rot: Option to rotate wavelet filters.
#
#  @return Convolved data.
#
def get_dir_filters(shape, angle_num, sigma):

    angles = np.arange(angle_num) * np.pi / angle_num

    shift = (shape[0] - 1) / 2

    index_matrix = np.tile(np.arange(shape[0]), (shape[0], 1)).T - shift

    def func(x, y):

        return (Gaussian_filter(rotate(index_matrix, x), y, fourier=True) *
                mex_hat(rotate(index_matrix, x + np.pi / 2), y))

    return np.array([func(angle, sigma) for angle in angles])


def convolve_dir_filters(data, filters):

    return np.array([convolve(data, f) for f in filters])
