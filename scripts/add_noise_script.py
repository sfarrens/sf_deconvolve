#! /usr/bin/env Python
# -*- coding: utf-8 -*-

"""ADD NOISE SCRIPT

This script does the following:
- Reads in "clean" images and PSF(s).
- Convolves the PSF(s) with the images.
- Adds Gaussian noise to the convolved images.
- Results are saved to a numpy binary file.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 5.0

:Date: 01/03/2017

"""

import numpy as np
import argparse as ap
import datetime
from psf.noise import add_noise
from psf.convolve import psf_convolve
from functions.stats import sigma_from_snr


def get_opts():

    '''Get script arguments'''

    global opts

    formatter = ap.ArgumentDefaultsHelpFormatter

    parser = ap.ArgumentParser('SCRIPT OPTIONS:',
                               formatter_class=formatter)

    parser.add_argument('-i', '--input', dest='input',
                        required=True, help='Input data file name.')

    parser.add_argument('-p', '--psf', dest='psf', required=True,
                        help='PSF file name.')

    parser.add_argument('--psf_type', dest='psf_type', required=False,
                        default='obj_var', help='PSF fortmat type. ' +
                        '[fixed, obj_var or pix_var]')

    parser.add_argument('-o', '--output', dest='output',
                        required=True, help='Output file name.')

    parser.add_argument('-r', '--random_seed', dest='random_seed', type=int,
                        nargs='+', required=False, help='Random seed.')

    parser.add_argument('-s', '--sigma', dest='sigma', type=float,
                        required=False, default=0.02, help='Noise level.')

    parser.add_argument('--snr', dest='snr', type=float,
                        required=False, help='Signal-to-noise ratio.')

    opts = parser.parse_args()


def add_noise_conv(data, psf):

    '''Convolve data with PSF and then add Gaussian random noise.'''

    print 'Convolving images with PSF(s)...'

    data_conv = psf_convolve(data, psf, psf_type=opts.psf_type)

    print 'Adding noise...'

    if isinstance(opts.snr, type(None)):
        print ' - Using sigma =', opts.sigma

    else:
        print ' - Using SNR =', opts.snr
        opts.sigma = np.std(data_conv, axis=(1, 2)) / opts.snr

    data_noisy = add_noise(data_conv, opts.sigma)

    return data_noisy


def output_file(data, psf, seed=None):

    '''Output noisy data file'''

    # Convolve data map with PSF and add noise.
    data_noisy = add_noise_conv(data, psf)

    # Save noisy data to binary file.
    if not isinstance(opts.snr, type(None)):
        tag = '_snr' + str(opts.snr)
    elif isinstance(opts.sigma, (list, np.ndarray)):
        tag = '_sig' + str(opts.sigma[0]).split('.')[1][:4]
    else:
        tag = '_sig' + str(opts.sigma).split('.')[1][:4]

    file_name = opts.output + tag

    if not isinstance(seed, type(None)):
        file_name += '_rand' + str(seed).zfill(2)

    np.save(file_name, data_noisy)

    print 'Output saved to: ' + file_name + '.npy'

    output_log(file_name)


def output_log(file_name):

    '''Output a log file'''

    print 'Writing log file...'

    log_file = open(file_name + '.log', 'w')
    log_file.write('Log file created: ' + str(datetime.datetime.now()) + '\n')
    log_file.write('Clean data file: ' + opts.input + '\n')
    log_file.write('PSF file: ' + opts.psf + '\n')
    log_file.write('PSF type: ' + opts.psf_type + '\n')
    log_file.write('Random seed: ' + str(opts.random_seed) + '\n')
    log_file.write('Sigma of noise added: ' + str(opts.sigma) + '\n')
    log_file.write('Desired SNR: ' + str(opts.snr) + '\n')
    log_file.write('Ouput file: ' + file_name + '.npy')
    log_file.close()


def run_script():

    '''Run the script'''

    print 'Running Add Noise Script...'

    # Read PSF file.
    psf = np.load(opts.psf)
    n_obj = psf.shape[0]

    if opts.psf_type == 'obj_var' and n_obj != psf.shape[0]:
        raise RuntimeError('The number of PSFs does not match the number ' +
                           'of galaxies selected.')

    # Read input data and output noisy data.
    if not isinstance(opts.random_seed, type(None)):
        for seed in opts.random_seed:
            np.random.seed(seed)
            data = np.random.permutation(np.load(opts.input))[:n_obj]
            output_file(data, psf, seed)
    else:
        data = np.load(opts.input)[:n_obj]
        output_file(data, psf)


def main():

    get_opts()
    run_script()


if __name__ == "__main__":
    main()
