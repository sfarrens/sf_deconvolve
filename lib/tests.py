# -*- coding: utf-8 -*-

"""DECONVOLUTION RESULT TESTS

This module contains methods for measuring the pixel and ellipticity errors of
a given stack of deconvolved images

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 16/01/2017

"""

from file_io import read_file
from quality import *
from functions.stats import gaussian_kernel, psnr_stack


def test_deconvolution(deconv_data, clean_data_file,
                       random_seed=None, kernel=None, metric='mean'):
    """Test deconvolution

    This method tests the quality of the deconvolved images

    Parameters
    ----------
    deconv_data : np.ndarray
        Deconvolved data, 3D array
    clean_data_file : str
        Clean data file name
    random_seed : int, optional
        Random seed
    kernel : int, optional
        Standard deviation of Gaussian kernel
    metric : str {mean, median}, optional
        Metric for averaging results (default is 'mean')

    Returns
    -------
    np.ndarray pixel errors, np.ndarray ellipticity errors

    Raises
    ------
    ValueError
        If the number of clean images does not match the number of deconvolved
        images

    """

    if not isinstance(random_seed, type(None)):
        np.random.seed(random_seed)
        clean_data = read_file(clean_data_file)
        clean_data = np.random.permutation(clean_data)[:deconv_data.shape[0]]
    else:
        clean_data = read_file(clean_data_file)[:deconv_data.shape[0]]

    if clean_data.shape != deconv_data.shape:
        raise ValueError('The number of clean images must match the number '
                         'deconvolved images.')

    if not isinstance(kernel, type(None)):

        def add_weights(data, weight):

            return np.array([x * weight for x in data])

        gk = gaussian_kernel(clean_data[0].shape, kernel)

        deconv_data = add_weights(deconv_data, gk)
        clean_data = add_weights(clean_data, gk)

    if metric == 'median':
        metric = np.median
    else:
        metric = np.mean

    px_err = nmse(clean_data, deconv_data, metric)
    ellip_err = e_error(clean_data, deconv_data, metric)
    psnr = psnr_stack(clean_data, deconv_data, metric)

    return (px_err, ellip_err, psnr)
