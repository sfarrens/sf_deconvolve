# -*- coding: utf-8 -*-

"""DECONVOLUTION FILE INPUT/OUTPUT

This module defines methods for file input and output for
deconvolution_script.py.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 13/03/2017

"""

import numpy as np
from astropy.io import fits


def check_data_format(data, n_dim):
    """Check data format

    This method checks that the input data has the correct number of dimensions

    Parameters
    ----------
    data : np.ndarray
        Input data array
    n_dim : int or list of ints
        Expected number of dimensions

    Returns
    -------
    np.ndarray (reshaped) data array

    Raises
    ------
    ValueError
        For invalid array dimensions

    """

    if data.ndim not in list(n_dim):
        raise ValueError('Input data array has an invalid number of '
                         'dimensions.')

    if data.ndim == 2:
        data = data.reshape(1, *data.shape)

    return data


def read_from_fits(file_name):
    """Read FITS file

    This method reads image array data from a FITS file.

    Parameters
    ----------
    file_name : str
        Name of file with path

    Retunrs
    -------
    np.ndarray array of image data

    """

    return fits.getdata(file_name)


def write_to_fits(file_name, data):
    """Write FITS file

    This method writes the output image array data to a FITS file.

    Parameters
    ----------
    file_name : str
        Name of file with path
    data : np.ndarray
        Image data array

    """

    fits.PrimaryHDU(data).writeto(file_name)


def read_file(file_name):
    """Read file

    This method reads image array data from a file.

    Parameters
    ----------
    file_name : str
        Name of file with path

    Retunrs
    -------
    np.ndarray array of image data

    Raises
    ------
    ValueError
        For invalid file extension

    """

    if file_name.endswith('.npy'):
        data = np.load(file_name)

    elif file_name.endswith(('.fits', '.fit', '.FITS', '.FIT')):
        data = read_from_fits(file_name)

    else:
        raise ValueError('Invalid file extension. Files must be FITS or numpy '
                         'binary.')

    data = check_data_format(data, [2, 3])

    return data


def read_input_files(data_file_name, psf_file_name, current_file_name=None):
    """Read input files

    This method reads image array data from the specified input files.

    Parameters
    ----------
    data_file_name : str
        Name of file with path for the noisy image data
    psf_file_name : str
        Name of file with path for the PSF image data
    current_file_name : str, optional
        Name of file with path for the current results

    Returns
    -------
    tuple of np.ndarray arrays of image data

    Raises
    ------
    ValueError
        If number of noisy images less than the number of PSFs
    ValueError
        If the shape of the current results does not match the input data

    """

    input_data = read_file(data_file_name)
    psf_data = read_file(psf_file_name)

    if input_data.shape[0] < psf_data.shape[0]:
        raise ValueError('The number of input images must be greater than or '
                         'or equal to the number of PSF images.')

    if not isinstance(current_file_name, type(None)):
        current_data = read_file(current_file_name)

        if current_data.shape != input_data.shape:
            raise ValueError('The number of current rescontruction images '
                             'must match the number of input images.')

    else:
        current_data = None

    return input_data, psf_data, current_data


def write_output_files(output_file_name, primal_res, dual_res=None,
                       output_format='npy'):

    """Write output files

    This method writes the image data results to the specified output file(s)

    Parameters
    ----------
    output_file_name : str
        Name of file with path for the output data
    primal_res : np.ndarray
        Array of primal output results
    dual_res : np.ndarray, optional
        Array of dual output results
    output_format : str, optional
        Output file format (numpy binary or FITS)

    """

    if output_format == 'fits':
        write_to_fits(output_file_name + '_primal.fits', primal_res)

        if not isinstance(dual_res, type(None)):
            write_to_fits(output_file_name + '_dual.fits', dual_res)

    else:
        np.save(output_file_name + '_primal', primal_res)

        if not isinstance(dual_res, type(None)):
            np.save(output_file_name + '_dual', dual_res)
