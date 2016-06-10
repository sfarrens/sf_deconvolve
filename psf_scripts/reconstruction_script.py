#! /usr/bin/env Python

#  @file reconstruction_script.py
#
#  Galaxy image reconstruction script
#
#  @author Samuel Farrens
#  @version 2.1
#  @date 2016
#

import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
from psf import *
from psf import reconstruct4 as r4
from functions.extra_math import mfactor
from functions.log import set_up_log, close_log
from functions.errors import catch_error, warn
from functions.stats import sigma_mad


##
#  Funciton to get script arguments.
#
def get_opts():

    # Make the arguments global.
    global opts

    # Define the parser.
    format = ap.ArgumentDefaultsHelpFormatter
    parser = ap.ArgumentParser('RECONSTRUCTION SCRIPT:',
                               formatter_class=format)

    # Add arguments.

    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s v2.1')

    parser.add_argument('-i', '--input', dest='input',
                        required=True, help='Input noisy data file name.')

    parser.add_argument('-p', '--psf', dest='psf', required=False,
                        help='PSF file name.')

    parser.add_argument('--psf_type', dest='psf_type', required=False,
                        default='obj_var',
                        choices=('fixed', 'obj_var', 'pix_var'),
                        help='PSF fortmat type. [fixed, obj_var or pix_var]')

    parser.add_argument('--psf_pcs', dest='psf_pcs', required=False,
                        help='PSF principal components file name.')

    parser.add_argument('--psf_coef', dest='psf_coef', required=False,
                        help='PSF coefficients file name.')

    parser.add_argument('--noise_est', dest='noise_est', required=False,
                        help='Initial noise estimate.')

    parser.add_argument('-o', '--output', dest='output',
                        required=True, help='Output file name.')

    parser.add_argument('-l', '--layout', dest='layout', type=int, nargs=2,
                        required=False, help='Map layout. [Map format only]')

    parser.add_argument('-m', '--mode', dest='mode', default='all',
                        choices=('all', 'wave', 'lowr', 'grad'),
                        required=False, help='Option to specify the '
                        'optimisation mode. [all, wave, lowr or grad]')

    parser.add_argument('--opt_type', dest='opt_type', default='condat',
                        choices=('condat', 'fwbw', 'gfwbw'), required=False,
                        help='Option to specify the optimisation method to be'
                        'implemented. [condat or fwbw]')

    parser.add_argument('-w', '--wavelet_levels', dest='wavelet_levels',
                        type=int, required=False, default=3,
                        help='Number of wavelet levels.')

    parser.add_argument('--wavelet_type', dest='wavelet_type', required=False,
                        default='1', help='Wavelet type.')

    parser.add_argument('--wave_thresh_factor', dest='wave_tf', type=float,
                        nargs='+', required=False, default=[3.0, 3.0, 4.0],
                        help='Wavelet threshold factor.')

    parser.add_argument('--lowr_thresh_factor', dest='lowr_tf', type=float,
                        required=False, default=1,
                        help='Low rank threshold factor.')

    parser.add_argument('--lowr_thresh_type', dest='lowr_thresh_type',
                        required=False, default='soft',
                        help='Low rank threshold type. [soft or hard]')

    parser.add_argument('--n_reweights', dest='n_reweights',
                        type=int, required=False, default=1,
                        help='Number of reweightings.')

    parser.add_argument('--n_iter', dest='n_iter',
                        type=int, required=False, default=150,
                        help='Number of iterations.')

    parser.add_argument('--relax', dest='relax',
                        type=float, required=False, default=0.5,
                        help='Relaxation parameter (rho_n).')

    parser.add_argument('--data_format', dest='data_format', default='cube',
                        required=False, help='Data format.')

    parser.add_argument('--no_pos', dest='no_pos',
                        action='store_false', required=False,
                        help='Option to turn off postivity constraint.')

    parser.add_argument('--no_grad', dest='no_grad',
                        action='store_false', required=False,
                        help='Option to turn off gradinet calculation.')

    # Define an object to store the arguments.
    opts = parser.parse_args()


##
#  Funciton to check the input data format.
#
#  @param[in] data: Input data.
#
def check_data_format(data, opts):

    # If the data is in map format it must have 2 dimensions.
    if opts.data_format == 'map' and data.ndim != 2:
        raise ValueError('Data in map format must have 2 dimensions.')

    # If the data is in cube format it must have 3 dimensions.
    elif opts.data_format == 'cube' and data.ndim != 3:
        raise ValueError('Data in cube format must have 3 dimensions.')


##
#  Funciton to run the script.
#
def run_script(opts, log=None):

    # # Create ouput file name if not provided.
    # if isinstance(opts.output, type(None)):
    #     opts.output = opts.input + '_' + opts.mode
    #
    # # Set-up log file if not provided.
    # if isinstance(log, type(None)):
    #     log = set_up_log(opts.output)

    print 'Running Reconstruction Script...'
    print ' ----------------------------------------'
    print ' - Input:', opts.input
    print ' - PSF:', opts.psf
    print ' - Mode:', opts.mode
    print ' - Optimisation:', opts.opt_type
    print ' - Positivity:', opts.no_pos
    print ' - Gradient Descent: ', opts.no_grad
    log.info('Running Reconstruction Script...')
    log.info(' - Input: ' + opts.input)
    log.info(' - PSF: ' + opts.psf)
    log.info(' - Mode: ' + opts.mode)
    log.info(' - Optimisation: ' + opts.opt_type)
    log.info(' - Positivity: ' + str(opts.no_pos))
    log.info(' - Gradient Descent: ' + str(opts.no_grad))

    if opts.mode in ('all', 'wave'):
        print ' - Wavelet Levels:', opts.wavelet_levels
        print ' - Wavelet Type:', opts.wavelet_type
        print ' - Wavelet Threshold Factor:', opts.wave_tf
        print ' - Number of Reweightings:', opts.n_reweights
        log.info(' - Wavelet Levels: ' + str(opts.wavelet_levels))
        log.info(' - Wavelet Type: ' + str(opts.wavelet_type))
        log.info(' - Wavelet Threshold Factor: ' +
                 str(opts.wave_tf))
        log.info(' - Number of Reweightings: ' + str(opts.n_reweights))

    if opts.mode in ('all', 'lowr'):
        print ' - Low Rank Threshold Factor:', opts.lowr_tf
        print ' - Low Rank Threshold Type:', opts.lowr_thresh_type
        log.info(' - Low Rank Threshold Factor: ' +
                 str(opts.lowr_tf))
        log.info(' - Low Rank Threshold Type: ' + str(opts.lowr_thresh_type))

    print ' - Number of Iterations:', opts.n_iter
    log.info(' - Number of Iterations: ' + str(opts.n_iter))

    # Read noisy data file.
    data_noisy = np.load(opts.input)

    # Read PSF file(s).
    if not isinstance(opts.psf, type(None)):
        psf = np.load(opts.psf)
        psf_pcs = None
        psf_coef = None
        if isinstance(opts.noise_est, type(None)):
            noise_est = sigma_mad(data_noisy)
        else:
            noise_est = float(opts.noise_est)
        print ' - Noise Estimate:', noise_est
        log.info(' - Noise Estimate: ' + str(noise_est))

    else:
        psf = None
        psf_pcs = np.load(opts.psf_pcs)
        psf_coef = np.load(opts.psf_coef)
        noise_est = np.load(opts.noise_est)

    # Check data format.
    check_data_format(data_noisy, opts)

    # Set mr_transform wavelet type option.
    wavelet_opt = ['-t ' + opts.wavelet_type]

    # Set the number of objects
    if opts.data_format == 'cube':
        layout = mfactor(data_noisy.shape[0])

    elif isinstance(opts.layout, type(None)):
        raise ValueError('Must specify the layout in map format.')

    print ' ----------------------------------------'

    # Perform reconstruction.
    primal_rec, dual_rec = r4.rec(data_noisy, noise_est, layout, psf=psf,
                                  psf_type=opts.psf_type, psf_pcs=psf_pcs,
                                  psf_coef=psf_coef,
                                  wavelet_levels=opts.wavelet_levels,
                                  wavelet_opt=wavelet_opt,
                                  wave_thresh_factor=np.array(opts.wave_tf),
                                  lowr_thresh_factor=opts.lowr_tf,
                                  lowr_thresh_type=opts.lowr_thresh_type,
                                  n_reweights=opts.n_reweights,
                                  n_iter=opts.n_iter, relax=opts.relax,
                                  mode=opts.mode, pos=opts.no_pos,
                                  grad=opts.no_grad,
                                  data_format=opts.data_format,
                                  opt_type=opts.opt_type, log=log)

    # Save outputs to numpy binary files.
    np.save(opts.output + '_primal', primal_rec)
    print 'Output 1 saved to: ' + opts.output + '_primal' + '.npy'
    log.info('Output 1 saved to: ' + opts.output + '_primal' + '.npy')

    if opts.opt_type == 'condat':
        np.save(opts.output + '_dual', dual_rec)
        print 'Output 2 saved to: ' + opts.output + '_dual' + '.npy'
        log.info('Output 2 saved to: ' + opts.output + '_dual' + '.npy')

    log.info('Script successfully completed!')

    close_log(log)


##
#  Script main.
#
def main():

    try:
        get_opts()
        log = set_up_log(opts.output)
        run_script(opts, log)

    except Exception, err:
        catch_error(err, log)
        return 1


if __name__ == "__main__":
    main()
