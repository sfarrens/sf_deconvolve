#  @file noise.py
#
#  NOISE ROUTINES
#
#  Functions for adding and
#  removing noise from the data.
#  Based on work by Yinghao Ge.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from scipy.signal import fftconvolve


##
#  Function that adds noise to the input data.
#
#  @param[in] data: Input data.
#  @param[in] sigma: Standard deviation. (Default = 1.0)
#  @param[in] noise_type: Type of noise. [gauss, poisson] (Default = 'gauss')
#
#  @return Noisy data.
#
def add_noise(data, sigma=1.0, noise_type='gauss'):

    if noise_type is 'gauss':
        # Return the data with Gaussian noise.
        return data + sigma * np.random.randn(*data.shape)

    elif noise_type is 'poisson':
        # Return the data with Poissonian noise.
        return data + np.random.poisson(np.abs(data))


##
#  Function that finds the minimum number of principal components required.
#
#  @param[in] u: Left singular vector.
#  @param[in] factor: Factor for testing auto correlation.
#
#  @return Number of principal components.
#
def find_n_pc(u, factor=0.5):

    # Get the shape of the galaxy images.
    gal_shape = np.repeat(np.int(np.sqrt(u.shape[0])), 2)

    # Find the auto correlation of the left singular vector.
    u_auto = [fftconvolve(a.reshape(gal_shape), np.rot90(a.reshape(gal_shape),
                          2), 'same') for a in u.T]

    # Return the required number of principal components.
    return np.sum([(a[zip(gal_shape / 2)] ** 2 <= factor * np.sum(a ** 2))
                   for a in u_auto])


##
#  Function that thresholds the input data.
#
#  @param[in] data: Input data.
#  @param[in] n_pc: Number of principal components.
#  @param[in] treshold_type: Type of threshold. [hard, soft]
#
#  @return Thresholded data.
#
def threshold(data, n_pc=None, threshold_type='hard'):

    # Get SVD of input data.
    u, s, v = np.linalg.svd(data)

    # Find the required number of principal components if not specified.
    if isinstance(n_pc, type(None)):
        n_pc = find_n_pc(u, factor=0.01)

    # If the number of PCs is too large use all of the singular values.
    if n_pc >= s.size:
        n_pc = s.size - 1
        print 'Warning! Using all singular values.'

    # Preserve only the required PCs.
    s[n_pc + 1:] = 0

    # Subtract lowest PC value for soft thresholding.
    if threshold_type is 'soft':
        s[:n_pc + 1] -= s[n_pc]

    # Resize the singular values to the shape of the input image.
    s = np.diag(s)
    s.resize(data.shape)

    # Return the thresholded image.
    return np.dot(u, np.dot(s, v))
