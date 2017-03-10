#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""PSF DECONVOLUTION SCRIPT

This module is executable and contains methods for deconvolving a set of
observed galaxy images.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 3.3

:Date: 13/12/2016

"""

import numpy as np
from sf_deconvolve_args import get_opts
from lib import deconvolve as dc
from lib.tests import test_deconvolution
from functions.errors import catch_error, warn
from functions.log import set_up_log, close_log


def set_out_string():
    """Set output string

    This method checks if an output string has been specified and if not
    creates and ouput string from the input string and the mode

    """

    if isinstance(opts.output, type(None)):
        opts.output = opts.input + '_' + opts.mode


def check_data_format(data, n_dim):
    """Check data format

    This method checks that the input data has the correct number of dimensions

    Parameters
    ----------
    data : np.ndarray
        Input data array
    n_dim : int
        Expected number of dimensions

    """

    if data.ndim != n_dim:
        raise ValueError('Input data must have ' + str(n_dim) + ' dimensions.')


def check_psf(data, n_obj, log):
    """Check PSF

    This method checks that the input PSFs are properly normalised

    Parameters
    ----------
    data : np.ndarray
        Input data array
    n_obj : int
        Expected number of PSFs
    log : logging.Logger
        Log instance

    """

    if opts.psf_type == 'fixed':
        check_data_format(data, 2)
    else:
        check_data_format(data, 3)
        if data.shape[0] != n_obj:
            raise ValueError('Number of PSFs must match number of images')

    if not np.all(np.abs(np.sum(data, axis=(1, 2)) - 1) < 1e-5):
        warn('Not all PSFs integrate to 1.0.')
        log.info(' - Not all PSFs integrate to 1.0.')


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
    check_data_format(data_noisy, 3)
    print ' - Input:', opts.input
    log.info(' - Input: ' + opts.input)

    # Read PSF file
    psf = np.load(opts.psf_file)
    check_psf(psf, data_noisy.shape[0], log)
    print ' - PSF:', opts.psf_file
    log.info(' - PSF: ' + opts.psf_file)

    # Read current deconvolution results file
    if not isinstance(opts.current_res, type(None)):
        primal = np.load(opts.current_res)
        check_data_format(primal, 3)
        print ' - Current Results:', opts.current_res
        log.info(' - Current Results: ' + opts.current_res)
    else:
        primal = None

    # Log set-up options
    print ' - Mode:', opts.mode
    print ' - PSF Type:', opts.psf_type
    print ' - Optimisation:', opts.opt_type
    print ' - Positivity:', not opts.no_pos
    print ' - Gradient Descent: ', not opts.no_grad
    print ' - Number of Reweightings:', opts.n_reweights
    print ' - Number of Iterations:', opts.n_iter
    print ' - Cost Function Window:', opts.cost_window
    print ' - Convergence Tolerance:', opts.convergence
    log.info(' - Mode: ' + opts.mode)
    log.info(' - PSF Type: ' + opts.psf_type)
    log.info(' - Optimisation: ' + opts.opt_type)
    log.info(' - Positivity: ' + str(not opts.no_pos))
    log.info(' - Gradient Descent: ' + str(not opts.no_grad))
    log.info(' - Number of Reweightings: ' + str(opts.n_reweights))
    log.info(' - Number of Iterations: ' + str(opts.n_iter))
    log.info(' - Cost Function Window: ' + str(opts.cost_window))
    log.info(' - Convergence Tolerance: ' + str(opts.convergence))

    # Log sparsity options
    if opts.mode in ('all', 'sparse'):
        print ' - Wavelet Type:', opts.wavelet_type
        print ' - Wavelet Threshold Factor:', opts.wave_thresh_factor
        log.info(' - Wavelet Type: ' + str(opts.wavelet_type))
        log.info(' - Wavelet Threshold Factor: ' +
                 str(opts.wave_thresh_factor))

    # Log low-rank options
    if opts.mode in ('all', 'lowr'):
        print ' - Low Rank Threshold Factor:', opts.lowr_thresh_factor
        print ' - Low Rank Threshold Type:', opts.lowr_thresh_type
        print ' - Low Rank Type:', opts.lowr_type
        log.info(' - Low Rank Threshold Factor: ' +
                 str(opts.lowr_thresh_factor))
        log.info(' - Low Rank Threshold Type: ' + str(opts.lowr_thresh_type))
        log.info(' - Low Rank Type: ' + str(opts.lowr_type))

    print h_line

    # Perform deconvolution.
    primal_res, dual_res = dc.run(data_noisy, psf, primal=primal, log=log,
                                  **vars(opts))

    # Test the deconvolution
    if not isinstance(opts.clean_data, type(None)):

        image_errors = test_deconvolution(primal_res, opts.clean_data,
                                          opts.random_seed, opts.kernel,
                                          opts.metric)

        print ' - Clean Data:', opts.clean_data
        print ' - Random Seed:', opts.random_seed
        log.info(' - Clean Data: ' + opts.clean_data)
        log.info(' - Random Seed: ' + str(opts.random_seed))

        if not isinstance(opts.kernel, type(None)):
            print ' - Gaussian Kernel:', opts.kernel
            log.info(' - Gaussian Kernel: ' + str(opts.kernel))

        print ' - Pixel Error:', image_errors[0]
        print ' - Shape Error:', image_errors[1]
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
        set_out_string()
        log = set_up_log(opts.output)
        run_script(log)

    except Exception, err:
        catch_error(err, log)
        return 1


if __name__ == "__main__":
    main()
