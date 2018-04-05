# -*- coding: utf-8 -*-

"""DECONVOLUTION RESULT TESTS

This module contains methods for measuring the pixel and ellipticity errors of
a given stack of deconvolved images

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 16/01/2017

"""

from . file_io import read_file
from sf_tools.image.quality import *
from modopt.math.stats import gaussian_kernel, psnr_stack


def test_images(results, truth, kernel=None, metric='mean'):
    """Test Image Results

    This method tests the quality of the recovered images

    Parameters
    ----------
    results : np.ndarray
        Resulting images, 3D array
    truth : str
        True images, 3D array
    kernel : int, optional
        Standard deviation of Gaussian kernel
    metric : str {mean, median}, optional
        Metric for averaging results (default is 'mean')

    Returns
    -------
    np.ndarray pixel errors, ellipticity errors, PSNR

    Raises
    ------
    ValueError
        If the number of clean images does not match the number of deconvolved
        images

    """

    if not isinstance(kernel, type(None)):

        def add_weights(data, weight):

            return np.array([x * weight for x in data])

        gk = gaussian_kernel(truth[0].shape, kernel)

        results = add_weights(results, gk)
        truth = add_weights(truth, gk)

    if metric == 'median':
        metric = np.median
    else:
        metric = np.mean

    px_err = nmse(truth, results, metric)
    ellip_err = e_error(truth, results, metric)
    psnr = psnr_stack(truth, results, metric)

    return (px_err, ellip_err, psnr)


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

    return test_images(deconv_data, clean_data, kernel, metric)


def test_psf_estimation(psf_data, true_psf_file, kernel=None, metric='mean'):
    """Test PSF Estimation

    This method tests the quality of the estimated PSFs

    Parameters
    ----------
    psf_data : np.ndarray
        Estimated PSFs, 3D array
    true_psf_file : str
        True PSFs file name
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

    true_psf = read_file(true_psf_file)

    if true_psf.shape != psf_data.shape:
        raise ValueError('The number of true PSF images must match the number '
                         'estimated PSF images.')

    return test_images(psf_data, true_psf, kernel, metric)
