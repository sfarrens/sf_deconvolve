# -*- coding: utf-8 -*-

"""CONVOLUTION ROUTINES

This module contains methods for convolving data.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 2.1

:Date: 04/01/2017

"""

import numpy as np
from scipy.signal import fftconvolve
try:
    from astropy.convolution import convolve_fft
except:
    pass
from sf_deconvolve.functions.np_adjust import rotate, rotate_stack


def convolve(data, kernel, method='astropy'):
    """Convolve data with kernel

    This method convolves the input data with a given kernel using FFT and
    is the default convolution used for all routines

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally a 2D image
    kernel : np.ndarray
        Input kernel array, normally a 2D kernel
    method : str {'astropy', 'scipy'}, optional
        Convolution method (default is 'astropy')

        'astropy':
            Uses the astropy.convolution.convolve_fft method provided in
            Astropy (http://www.astropy.org/)

        'scipy':
            Uses the scipy.signal.fftconvolve method provided in SciPy
            (https://www.scipy.org/)

    Returns
    -------
    np.ndarray convolved data

    Raises
    ------
    ValueError
        If `data` and `kernel` do not have the same number of dimensions
    ValueError
        If `method` is not 'astropy' or 'scipy'

    """

    if data.ndim != kernel.ndim:
        raise ValueError('Data and kernel must have the same dimensions.')

    if method not in ('astropy', 'scipy'):
        raise ValueError('Invalid method. Options are "astropy" or "scipy".')

    if method == 'astropy':
        return convolve_fft(data, kernel, boundary='wrap', crop=True)

    elif method == 'scipy':
        return fftconvolve(data, kernel, mode='same')


def psf_convolve(data, psf, psf_rot=False, psf_type='fixed'):
    """Convolve data with PSF

    This method convolves an image with a PSF

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally an array of 2D images
    psf : np.ndarray
        Input PSF array, normally either a single 2D PSF or an array of 2D
        PSFs
    psf_rot: bool
        Option to rotate PSF by 180 degrees
    psf_type : str {'fixed', 'obj_var'}, optional
        PSF type (default is 'fixed')

        'fixed':
            The PSF is fixed, i.e. it is the same for each image

        'obj_var':
            The PSF is object variant, i.e. it is different for each image

    Returns
    -------
    np.ndarray convolved data

    Raises
    ------
    ValueError
        If `psf_type` is not 'fixed' or 'obj_var'

    """

    if psf_type not in ('fixed', 'obj_var'):
        raise ValueError('Invalid PSF type. Options are "fixed" or "obj_var"')

    if psf_rot and psf_type == 'fixed':
        psf = rotate(psf)

    elif psf_rot:
        psf = rotate_stack(psf)

    if psf_type == 'fixed':
        return np.array([convolve(data_i, psf) for data_i in data])

    elif psf_type == 'obj_var':
        return np.array([convolve(data_i, psf_i) for data_i, psf_i in
                        zip(data, psf)])
