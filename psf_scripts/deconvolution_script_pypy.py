#! /usr/bin/env Python
# -*- coding: utf-8 -*-

"""PSF DECONVOLUTION SCRIPT

This module is executable and contains methods for deconvolving a set of
observed galaxy images.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 3.0

:Date: 12/12/2016

"""

import numpy as np
from deconvolution_args import get_opts
from psf_pypy import deconvolve as dc
from creepy.image.quality import *
from functions.log import set_up_log, close_log
from functions.errors import catch_error, warn


def check_data_format(data):
    """Check data format

    This method checks that the input data is correctly formatted.

    Parameters
    ----------
    data : np.ndarray
        Input data array

    """

    # The data must have 3 dimensions.
    if data.ndim != 3:
        raise ValueError('Input data must have 3 dimensions.')


def test_deconvolution(deconv_data):
    """Test deconvolution

    This method tests the quality of the deconvolved images.

    Parameters
    ----------
    deconv_data : np.ndarray
        Deconvolved data

    Returns
    -------
    np.ndarray pixel errors, np.ndarray ellipticity errors

    """

    if not isinstance(opts.random_seed, type(None)):
        np.random.seed(opts.random_seed)
        clean_data = np.load(opts.clean_data)
        clean_data = np.random.permutation(clean_data)[:deconv_data.shape[0]]
    else:
        clean_data = np.load(opts.clean_data)[:deconv_data.shape[0]]

    px_err = nmse(clean_data, deconv_data)
    ellip_err = e_error(clean_data, deconv_data)

    return (px_err, ellip_err)


def run_script(log):
    """Run script

    This method runs the script.

    Parameters
    ----------
    log : logging.Logger
        Log instance

    """

    h_line = ' ----------------------------------------'

    # Begin log
    output_text = 'Running Deconvolution Script...'
    print output_text
    log.info(output_text)

    print h_line

    # Read noisy data file
    data_noisy = np.load(opts.input)
    check_data_format(data_noisy)
    print ' - Input:', opts.input
    log.info(' - Input: ' + opts.input)

    # Read PSF file
    psf = np.load(opts.psf)
    print ' - PSF:', opts.psf
    log.info(' - PSF: ' + opts.psf)

    # Read current deconvolution results file
    if not isinstance(opts.current_res, type(None)):
        primal = np.load(opts.current_res)
        print ' - Current Results:', opts.current_res
        log.info(' - Current Results: ' + opts.current_res)
    else:
        primal = None

    # Log set-up options
    print ' - Mode:', opts.mode
    print ' - PSF Type:', opts.psf_type
    print ' - Optimisation:', opts.opt_type
    print ' - Positivity:', opts.no_pos
    print ' - Gradient Descent: ', opts.no_grad
    print ' - Number of Reweightings:', opts.n_reweights
    print ' - Number of Iterations:', opts.n_iter
    log.info(' - Mode: ' + opts.mode)
    log.info(' - PSF Type: ' + opts.psf_type)
    log.info(' - Optimisation: ' + opts.opt_type)
    log.info(' - Positivity: ' + str(opts.no_pos))
    log.info(' - Gradient Descent: ' + str(opts.no_grad))
    log.info(' - Number of Reweightings: ' + str(opts.n_reweights))
    log.info(' - Number of Iterations: ' + str(opts.n_iter))

    # Log sparsity options
    if opts.mode in ('all', 'sparse'):
        print ' - Wavelet Levels:', opts.wavelet_levels
        print ' - Wavelet Type:', opts.wavelet_type
        print ' - Wavelet Threshold Factor:', opts.wave_tf
        log.info(' - Wavelet Levels: ' + str(opts.wavelet_levels))
        log.info(' - Wavelet Type: ' + str(opts.wavelet_type))
        log.info(' - Wavelet Threshold Factor: ' +
                 str(opts.wave_tf))

    # Log low-rank options
    if opts.mode in ('all', 'lowr'):
        print ' - Low Rank Threshold Factor:', opts.lowr_tf
        print ' - Low Rank Threshold Type:', opts.lowr_thresh_type
        print ' - Low Rank Type:', opts.lowr_type
        log.info(' - Low Rank Threshold Factor: ' +
                 str(opts.lowr_tf))
        log.info(' - Low Rank Threshold Type: ' + str(opts.lowr_thresh_type))
        log.info(' - Low Rank Type: ' + str(opts.lowr_type))

    print h_line

    # Set mr_transform wavelet type option
    wavelet_opt = ['-t ' + opts.wavelet_type]

    # Perform deconvolution.
    primal_res, dual_res = dc.go(data_noisy, psf,
                                 noise_est=opts.noise_est,
                                 primal=primal,
                                 psf_type=opts.psf_type,
                                 wavelet_levels=opts.wavelet_levels,
                                 wavelet_opt=wavelet_opt,
                                 wave_thresh_factor=np.array(opts.wave_tf),
                                 lowr_thresh_factor=opts.lowr_tf,
                                 lowr_thresh_type=opts.lowr_thresh_type,
                                 lowr_type=opts.lowr_type,
                                 n_reweights=opts.n_reweights,
                                 n_iter=opts.n_iter,
                                 relax=opts.relax,
                                 mode=opts.mode,
                                 pos=opts.no_pos,
                                 grad=opts.no_grad,
                                 opt_type=opts.opt_type,
                                 log=log,
                                 output=opts.output,
                                 liveplot=opts.liveplot)

    # Test the deconvolution
    if not isinstance(opts.clean_data, type(None)):

        image_errors = test_deconvolution(primal_res)

        print ' - Clean Data:', opts.clean_data
        print ' - Random Seed:', opts.random_seed
        print ' - Pixel Error:', image_errors[0]
        print ' - Shape Error:', image_errors[1]
        log.info(' - Clean Data: ' + opts.clean_data)
        log.info(' - Random Seed: ' + str(opts.random_seed))
        log.info(' - Pixel Error: ' + str(image_errors[0]))
        log.info(' - Shape Error: ' + str(image_errors[1]))

    print h_line

    # Save outputs to numpy binary files
    np.save(opts.output + '_primal', primal_res)
    print ' Output 1 saved to: ' + opts.output + '_primal' + '.npy'
    log.info('Output 1 saved to: ' + opts.output + '_primal' + '.npy')

    if opts.opt_type == 'condat':
        np.save(opts.output + '_dual', dual_res)
        print ' Output 2 saved to: ' + opts.output + '_dual' + '.npy'
        log.info('Output 2 saved to: ' + opts.output + '_dual' + '.npy')

    log.info('Script successfully completed!')
    log.info('')

    close_log(log)

    print h_line


def main():

    try:
        global opts
        opts = get_opts()
        log = set_up_log(opts.output)
        run_script(log)

    except Exception, err:
        catch_error(err, log)
        return 1


if __name__ == "__main__":
    main()
