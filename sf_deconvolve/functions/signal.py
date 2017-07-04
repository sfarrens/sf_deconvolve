#  @file signal.py
#
#  SIGNAL PROCESSING FUNCTIONS
#
#  Basic functions for signal processing.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from functions.comp import check_float


##
#  Function that implements a Gaussian filter.
#
#  @param[in] x: Input data point.
#  @param[in] sigma: Filter scale.
#
#  @return Guassian filtered data.
#
def Gaussian_filter(x, sigma, fourier=False):

    x = check_float(x)
    sigma = check_float(sigma)

    val = np.exp(-0.5 * (x / sigma) ** 2)

    if fourier:
        return val

    else:
        return val / (np.sqrt(2 * np.pi) * sigma)


##
#  Function that implements a Mexican hat (or Ricker) wavelet.
#
#  @param[in] x: Input data point.
#  @param[in] sigma: Filter scale.
#
#  @return Wavelet filtered data.
#
def mex_hat(x, sigma):

    x = check_float(x)
    sigma = check_float(sigma)

    xs = (x / sigma) ** 2
    val = 2 * (3 * sigma) ** -0.5 * np.pi ** -0.25

    return val * (1 - xs) * np.exp(-0.5 * xs)


##
#  Function that implements a directional Mexican hat (or Ricker) wavelet.
#
#  @param[in] x: Input data point.
#  @param[in] sigma: Filter scale.
#
#  @return Wavelet transformed data.
#
def mex_hat_dir(x, y, sigma):

    return -0.5 * (x / sigma) ** 2 * mex_hat(y, sigma)


##
#  Function that tests two operators to see if they are the transpose of each
#  other.
#
#  @param[in] operator: Operator function.
#  @param[in] operator_t: Transpose operator function.
#  @param[in] data_shape: 2D Data shape.
#
def transpose_test(operator, operator_t, x_shape, x_args, y_shape=None,
                   y_args=None):

    if isinstance(y_shape, type(None)):
        y_shape = x_shape

    if isinstance(y_args, type(None)):
        y_args = x_args

    # Generate random arrays.
    x = np.random.ranf(x_shape)
    y = np.random.ranf(y_shape)

    # Calculate <MX, Y>
    mx_y = np.sum(np.multiply(operator(x, *x_args), y))

    # Calculate <X, M.TY>
    x_mty = np.sum(np.multiply(x, operator_t(y, *y_args)))

    # Test the difference between the two.
    print ' - |<MX, Y> - <X, M.TY>| =', np.abs(mx_y - x_mty)
