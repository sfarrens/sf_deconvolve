#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""SF_DECONVOLVE

This module is executable and contains methods for deconvolving a set of
observed galaxy images.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 3.4

:Date: 14/03/2017

References
----------

.. [F2017] Farrens et al., Space variant deconvolution of galaxy survey images,
    2017, A&A. [https://arxiv.org/abs/1703.02305]

Notes
-----
This code implements equations 8 and 17 from [F2017]_.

"""

import numpy as np
from os.path import splitext
from lib.sf_deconvolve_args import get_opts
from lib.file_io import *
from lib.deconvolve import run
from lib.tests import test_deconvolution
from functions.errors import catch_error, warn
from functions.log import set_up_log, close_log


def set_out_string():
    """Set output string

    This method checks if an output string has been specified and if not
    creates and ouput string from the input string and the mode

    """

    if isinstance(opts.output, type(None)):
        opts.output = splitext(opts.input)[0] + '_' + opts.mode


def check_psf(data, log):
    """Check PSF

    This method checks that the input PSFs are properly normalised and updates
    the opts namespace with the PSF type

    Parameters
    ----------
    data : np.ndarray
        Input data array
    log : logging.Logger
        Log instance

    """

    psf_sum = np.sum(data, axis=tuple(range(data.ndim - 2, data.ndim)))

    if not np.all(np.abs(psf_sum - 1) < 1e-5):
        warn('Not all PSFs integrate to 1.0.')
        log.info(' - Not all PSFs integrate to 1.0.')

    opts_dict = vars(opts)
    opts_dict['psf_type'] = 'obj_var'

    if data.ndim == 2:
        opts_dict['psf_type'] = 'fixed'


def run_script(log):
    """Run script

    This method runs the script.

    Parameters
    ----------
    log : logging.Logger
        Log instance

    """

    h_line = ' ' + '-' * 70
    print h_line

    # Begin log
    output_text = ' Running SF_DECONVOLVE'
    print output_text
    log.info(output_text)

    print h_line

    ###########################################################################
    # Read the input data files
    data_noisy, psf_data, primal = read_input_files(opts.input, opts.psf_file,
                                                    opts.current_res)
    check_psf(psf_data, log)
    ###########################################################################

    # Log input options
    print ' - Input File:', opts.input
    print ' - PSF File:', opts.psf_file
    log.info(' - Input File: ' + opts.input)
    log.info(' - PSF File: ' + opts.psf_file)
    if not isinstance(opts.current_res, type(None)):
        print ' - Current Results:', opts.current_res
        log.info(' - Current Results: ' + opts.current_res)

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

    ###########################################################################
    # Perform deconvolution.
    primal_res, dual_res = run(data_noisy, psf_data, primal=primal, log=log,
                               **vars(opts))
    ###########################################################################

    if not isinstance(opts.clean_data, type(None)):

        #######################################################################
        # Test the deconvolution
        image_errors = test_deconvolution(primal_res, opts.clean_data,
                                          opts.random_seed, opts.kernel,
                                          opts.metric)
        #######################################################################

        # Log testing options
        print ' - Clean Data:', opts.clean_data
        print ' - Random Seed:', opts.random_seed
        log.info(' - Clean Data: ' + opts.clean_data)
        log.info(' - Random Seed: ' + str(opts.random_seed))

        # Log kernel options
        if not isinstance(opts.kernel, type(None)):
            print ' - Gaussian Kernel:', opts.kernel
            log.info(' - Gaussian Kernel: ' + str(opts.kernel))

        # Log test results
        print ' - Pixel Error:', image_errors[0]
        print ' - Shape Error:', image_errors[1]
        log.info(' - Pixel Error: ' + str(image_errors[0]))
        log.info(' - Shape Error: ' + str(image_errors[1]))

    print h_line

    ###########################################################################
    # Save outputs to numpy binary files
    write_output_files(opts.output, primal_res, dual_res=dual_res,
                       output_format=opts.output_format)
    ###########################################################################

    # Log output options
    print (' Output 1 saved to: ' + opts.output + '_primal' + '.' +
           opts.output_format)
    log.info('Output 1 saved to: ' + opts.output + '_primal' + '.' +
             opts.output_format)
    if opts.opt_type == 'condat':
        print (' Output 2 saved to: ' + opts.output + '_dual' + '.' +
               opts.output_format)
        log.info('Output 2 saved to: ' + opts.output + '_dual' + '.' +
                 opts.output_format)

    # Close log
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
