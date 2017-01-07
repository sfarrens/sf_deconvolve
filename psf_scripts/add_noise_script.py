#! /usr/bin/env Python
# -*- coding: utf-8 -*-

# This script does the following:
# - Reads in the data and PSF.
# - Convolves the PSF with the data map.
# - Adds Gaussian noise to the convolved image.
# - Results are saved to a numpy binary file.

import numpy as np
import argparse as ap
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

    parser.add_argument('-r', '--random_seed', dest='random_seed',
                        type=int, required=False, help='Random seed.')

    parser.add_argument('-s', '--sigma', dest='sigma', type=float,
                        required=False, default=0.02, help='Noise level.')

    parser.add_argument('--snr', dest='snr', type=float,
                        required=False, help='Signal-to-noise ratio.')

    opts = parser.parse_args()


def get_sigma(data):

    '''Estimate sigma from signal-to-noise ratio.'''

    if not isinstance(opts.snr, type(None)):

        print ' - Using SNR =', opts.snr

        opts.sigma = sigma_from_snr(data, opts.snr)


def add_noise_conv(data, psf):

    '''Convolve data with PSF and then add Gaussian random noise.'''

    print 'Convolving PSF with image and adding noise...'
    print ' - Using sigma =', opts.sigma

    data_noisy = add_noise(psf_convolve(data, psf, psf_type=opts.psf_type),
                           opts.sigma)

    return data_noisy


def run_script():

    '''Run the script'''

    print 'Running Add Noise Script...'

    # Read input files.
    psf = np.load(opts.psf)
    n_obj = psf.shape[0]
    if not isinstance(opts.random_seed, type(None)):
        np.random.seed(opts.random_seed)
        data = np.random.permutation(np.load(opts.input))[:n_obj]
    else:
        data = np.load(opts.input)[:n_obj]

    if opts.psf_type == 'obj_var' and n_obj != psf.shape[0]:
        raise RuntimeError('The number of PSFs does not match the number ' +
                           'of galaxies selected.')

    # Estimate sigma from SNR.
    get_sigma(data)

    # Convolve data map with PSF and add noise.
    data_noisy = add_noise_conv(data, psf)

    # Save noisy data to binary file.
    np.save(opts.output, data_noisy)

    print 'Output saved to: ' + opts.output + '.npy'


def main():

    get_opts()
    run_script()


if __name__ == "__main__":
    main()
