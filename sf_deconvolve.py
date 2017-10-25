#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""SF_DECONVOLVE

This module is executable and contains methods for deconvolving a set of
observed galaxy images.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 4.3

:Date: 23/10/2017

Notes
-----
This code implements equations 8 and 17 from [F2017]_.

"""

from __future__ import print_function
import numpy as np
from os.path import splitext
from lib import __version__
from lib.args import get_opts
from lib.file_io import *
from lib.deconvolve import run
from lib.tests import test_deconvolution, test_psf_estimation
from sf_tools.interface.errors import catch_error, warn
from sf_tools.interface.log import set_up_log, close_log


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

    if data.ndim == 2:
        opts.psf_type = 'fixed'


def run_script(log):
    """Run script

    This method runs the script.

    Parameters
    ----------
    log : logging.Logger
        Log instance

    """

    h_line = ' ' + '-' * 70
    print(h_line)

    # Begin log
    output_text = ' Running SF_DECONVOLVE'
    print(output_text)
    log.info(output_text)
    opts.log = log

    print(h_line)

    ###########################################################################
    # Read the input data files
    data_noisy, psf_data, primal = read_input_files(opts.input, opts.psf_file,
                                                    opts.current_res)
    check_psf(psf_data, log)
    opts.primal = primal
    ###########################################################################

    # Log input options
    print(' - Input File:', opts.input)
    print(' - PSF File:', opts.psf_file)
    log.info(' - Input File: ' + opts.input)
    log.info(' - PSF File: ' + opts.psf_file)
    if not isinstance(opts.current_res, type(None)):
        print(' - Current Results:', opts.current_res)
        log.info(' - Current Results: ' + opts.current_res)

    # Log set-up options
    print(' - Mode:', opts.mode)
    print(' - PSF Type:', opts.psf_type)
    print(' - Gradient Type: ', opts.grad_type)
    print(' - Optimisation:', opts.opt_type)
    print(' - Positivity:', not opts.no_pos)
    print(' - Number of Reweightings:', opts.n_reweights)
    print(' - Number of Iterations:', opts.n_iter)
    print(' - Cost Function Window:', opts.cost_window)
    print(' - Convergence Tolerance:', opts.convergence)
    print(' - Suppress Plots:', opts.no_plots)
    log.info(' - Mode: ' + opts.mode)
    log.info(' - PSF Type: ' + opts.psf_type)
    log.info(' - Gradient Type: ' + opts.grad_type)
    log.info(' - Optimisation: ' + opts.opt_type)
    log.info(' - Positivity: ' + str(not opts.no_pos))
    log.info(' - Number of Reweightings: ' + str(opts.n_reweights))
    log.info(' - Number of Iterations: ' + str(opts.n_iter))
    log.info(' - Cost Function Window: ' + str(opts.cost_window))
    log.info(' - Convergence Tolerance: ' + str(opts.convergence))
    log.info(' - Suppress Plots:' + str(opts.no_plots))

    # Log sparsity options
    if opts.mode in ('all', 'sparse'):
        print(' - Wavelet Type:', opts.wavelet_type)
        print(' - Wavelet Threshold Factor:', opts.wave_thresh_factor)
        log.info(' - Wavelet Type: ' + str(opts.wavelet_type))
        log.info(' - Wavelet Threshold Factor: ' +
                 str(opts.wave_thresh_factor))

    # Log low-rank options
    if opts.mode in ('all', 'lowr'):
        print(' - Low Rank Threshold Factor:', opts.lowr_thresh_factor)
        print(' - Low Rank Threshold Type:', opts.lowr_thresh_type)
        print(' - Low Rank Type:', opts.lowr_type)
        log.info(' - Low Rank Threshold Factor: ' +
                 str(opts.lowr_thresh_factor))
        log.info(' - Low Rank Threshold Type: ' + str(opts.lowr_thresh_type))
        log.info(' - Low Rank Type: ' + str(opts.lowr_type))

    # Log PSF estimation options
    if opts.grad_type == 'psf_unknown':
        print(' - PSF Estimation Control Parameter (Lambda):', opts.lambda_psf)
        print(' - PSF Estimation Gradient Step Size (Beta):', opts.beta_psf)
        log.info(' - PSF Estimation Lambda: ' + str(opts.lambda_psf))
        log.info(' - PSF Estimation Beta: ' + str(opts.beta_psf))

    # Log PSF estimation options
    if opts.grad_type == 'shape':
        print(' - Shape Constraint Control Parameter (Lambda):',
              opts.lambda_shape)
        log.info(' - Shape Constraint Lambda: ' + str(opts.lambda_shape))

    print(h_line)

    ###########################################################################
    # Perform deconvolution.
    results = run(data_noisy, psf_data, **vars(opts))
    ###########################################################################

    if not isinstance(opts.clean_data, type(None)):

        print(h_line)

        #######################################################################
        # Test the deconvolution
        image_errors = test_deconvolution(results[0], opts.clean_data,
                                          opts.random_seed, opts.kernel,
                                          opts.metric)
        #######################################################################

        # Log testing options
        print(' - Clean Data:', opts.clean_data)
        print(' - Random Seed:', opts.random_seed)
        log.info(' - Clean Data: ' + opts.clean_data)
        log.info(' - Random Seed: ' + str(opts.random_seed))

        # Log kernel options
        if not isinstance(opts.kernel, type(None)):
            print(' - Gaussian Kernel:', opts.kernel)
            log.info(' - Gaussian Kernel: ' + str(opts.kernel))

        # Log test results
        print(' - Galaxy Pixel Error:', image_errors[0])
        print(' - Galaxy Shape Error:', image_errors[1])
        print(' - Galaxy PSNR:', image_errors[2])
        log.info(' - Galaxy Pixel Error: ' + str(image_errors[0]))
        log.info(' - Galaxy Shape Error: ' + str(image_errors[1]))
        log.info(' - Galaxy PSNR: ' + str(image_errors[2]))

    if (not isinstance(opts.true_psf, type(None)) and
            opts.grad_type == 'psf_unknown'):

        ###################################################################
        # Test the psf estimation
        psf_errors = test_psf_estimation(results[2], opts.true_psf,
                                         opts.kernel, opts.metric)
        ###################################################################

        print('')

        # Log testing options
        print(' - True PSFs:', opts.true_psf)
        log.info(' - True PSFs: ' + opts.true_psf)

        # Log test results
        print(' - PSF Pixel Error:', psf_errors[0])
        print(' - PSF Shape Error:', psf_errors[1])
        print(' - PSF PSNR:', psf_errors[2])
        log.info(' - PSF Pixel Error: ' + str(psf_errors[0]))
        log.info(' - PSF Shape Error: ' + str(psf_errors[1]))
        log.info(' - PSF PSNR: ' + str(psf_errors[2]))

    print(h_line)

    ###########################################################################
    # Save outputs to numpy binary files
    write_output_files(opts.output, results[0], dual_res=results[1],
                       psf_res=results[2], output_format=opts.output_format)
    ###########################################################################

    # Log output options
    print(' Output 1 saved to: ' + opts.output + '_primal' + '.' +
          opts.output_format)
    log.info('Output 1 saved to: ' + opts.output + '_primal' + '.' +
             opts.output_format)
    if opts.opt_type == 'condat':
        print(' Output 2 saved to: ' + opts.output + '_dual' + '.' +
              opts.output_format)
        log.info('Output 2 saved to: ' + opts.output + '_dual' + '.' +
                 opts.output_format)
    if opts.grad_type == 'psf_unknown':
        print(' Output 3 saved to: ' + opts.output + '_psf' + '.' +
              opts.output_format)
        log.info('Output 3 saved to: ' + opts.output + '_psf' + '.' +
                 opts.output_format)

    # Close log
    log.info('Script successfully completed!')
    log.info('')
    close_log(log)

    print(h_line)


def main(args=None):

    try:
        global opts
        opts = get_opts(args)
        set_out_string()
        log = set_up_log(opts.output)
        run_script(log)

    except Exception as err:
        catch_error(err, log)
        return 1


if __name__ == "__main__":
    main()
