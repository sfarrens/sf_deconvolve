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
from convolve import convolve
from functions import image, np_adjust, stats
from psf import image_file_io, transform

from scipy.sparse import linalg


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
#  Function that removes noise from the input data.
#
#  @param[in] data: Input data.
#  @param[in] level: Threhold level for removing noise.
#  @param[in] threshold_type: Type of threhold. [hard, soft] (Default: 'hard')
#
#  @return Denoised data.
#
def denoise(data, level, threshold_type='hard'):

    if threshold_type == 'soft':
        # Return soft threshold.
        return np.sign(data) * (np.abs(data) - level) * (np.abs(data) >= level)

    else:
        # Return hard threshold.
        return data * (np.abs(data) >= level)


##
#  Function calculates the L2 norm of the noise in a data map.
#
#  @param[in] data: Input noisy data map.
#  @param[in] layout: 2D layout of data map.
#  @param[in] pixel_rad: Pixel radius.
#
#  @return L2 norm of noise.
#
def get_l2norm(data, layout, pixel_rad=9):

    radius = data.shape[0] / layout[0] / 2
    centres = image.image_centres(data.shape, layout)
    cube = image_file_io.gen_data_cube(data, centres, pixel_rad)
    cube = transform.cube2map(np.array([np_adjust.pad2d(x, radius - pixel_rad)
                              for x in cube]), layout)
    diff = data - cube
    l2norm = np.linalg.norm(diff) / np.sum(diff > 0) * np.prod(data.shape)

    print ' - L2 Norm of noise from data:', l2norm

    return l2norm


##
#  Function that retrieves the autocorrelation map of the input data.
#
#  @param[in] data: Input data.
#  @param[in] pixel_rad: Pixel radius.
#
#  @return Autocorrelation map.
#
def get_auto_correl_map(data, pixel_rad):

    # Calculate the autocorrelation around each pixel in the input data
    windows = image.FetchWindows(data, pixel_rad, all=True)
    map = windows.scan(func=stats.autocorr2d)

    # Return a normalised map.
    return (map / map.max()).reshape(data.shape)


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
    u_auto = [convolve(a.reshape(gal_shape), np.rot90(a.reshape(gal_shape), 2))
              for a in u.T]

    # Return the required number of principal components.
    return np.sum(((a[zip(gal_shape / 2)] ** 2 <= factor * np.sum(a ** 2))
                   for a in u_auto))


##
#  Function that thresholds the singular values of the input data.
#
#  @param[in] data: Input data.
#  @param[in] threshold: Threshold value.
#  @param[in] n_pc: Number of principal components.
#  @param[in] treshold_type: Type of threshold. [hard, soft]
#
#  @return Thresholded data.
#
def svd_thresh(data, threshold=None, n_pc=None, thresh_type='hard'):

    # Get SVD of input data.
    u, s, v = np.linalg.svd(data)

    # Find the threshold if not provided.
    if isinstance(threshold, type(None)):

        # Find the required number of principal components if not specified.
        if isinstance(n_pc, type(None)):
            n_pc = find_n_pc(u, factor=0.1)

        # If the number of PCs is too large use all of the singular values.
        if n_pc >= s.size or n_pc == 'all':
            n_pc = s.size - 1
            print 'Warning! Using all singular values.'

        threshold = s[n_pc]

    # Remove noise from singular values.
    s_new = denoise(s, threshold, thresh_type)

    # Resize the singular values to the shape of the input image.
    s_new = np.diag(s_new)
    s_new.resize(data.shape)

    # Return the thresholded image.
    return np.dot(u, np.dot(s_new, v))


##
#  Function that thresholds the SVD coefficients of the input data.
#
#  @param[in] data: Input data.
#  @param[in] threshold: Threshold value.
#  @param[in] treshold_type: Type of threshold. [hard, soft]
#
#  @return Thresholded data.
#
def svd_thresh_coef(data, operator, threshold, thresh_type='hard'):

        # Convert data cube to matrix.
        data_matrix = transform.cube2matrix(data)

        # Get SVD of data matrix.
        u, s, v = np.linalg.svd(data_matrix, full_matrices=False)

        # Compute coefficients.
        a = np.dot(np.diag(s), v)

        # Compute threshold matrix.
        u_cube = transform.matrix2cube(u, data.shape[1:])
        ti = np.array([np.linalg.norm(x) for x in operator(u_cube)])
        ti = np.repeat(ti, a.shape[1]).reshape(a.shape)
        threshold *= ti

        # Remove noise from coefficients.
        a_new = denoise(a, threshold, thresh_type)

        # Return the thresholded image.
        return np.dot(u, a_new)
