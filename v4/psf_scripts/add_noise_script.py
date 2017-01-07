#!/Users/sfarrens/Documents/Library/anaconda/bin/python

# This script does the following:
# - Reads in the data and PSF.
# - Convolves the PSF with the data map.
# - Adds Gaussian noise to the convolved image.
# - Results are saved to a numpy binary file.

import numpy as np
import argparse as ap
from psf import *
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

    parser.add_argument('-n', '--n_obj', dest='n_obj', type=int,
                        required=False, help='Number of objects.')

    parser.add_argument('-r', '--random_seed', dest='random_seed',
                        type=int, required=False, help='Random seed.')

    parser.add_argument('-s', '--sigma', dest='sigma', type=float,
                        required=False, default=0.02, help='Noise level.')

    parser.add_argument('--snr', dest='snr', type=float,
                        required=False, help='Signal-to-noise ratio.')

    parser.add_argument('-l', '--layout', dest='layout', type=int, nargs=2,
                        required=False, help='Map layout. [Map format only]')

    parser.add_argument('--data_format', dest='data_format', default='cube',
                        required=False, help='Data format.')

    opts = parser.parse_args()


def check_data_format(data):

    '''Check that data format matches the input data shape.'''

    if opts.data_format == 'map' and data.ndim != 2:
        raise ValueError('Data in map format must have 2 dimensions.')

    elif opts.data_format == 'cube' and data.ndim != 3:
        raise ValueError('Data in cube format must have 3 dimensions.')


def get_sigma(data):

    '''Estimate sigma from signal-to-noise ratio.'''

    if not isinstance(opts.snr, type(None)):

        print ' - Using SNR =', opts.snr

        if opts.data_format == 'map':

            if isinstance(opts.layout, type(None)):
                raise ValueError('Layout must be specified for map format')

            y = transform.map2cube(data, opts.layout)

        else:
            y = data

        opts.sigma = sigma_from_snr(data, opts.snr)


def add_noise_conv(data, psf):

    '''Convolve data with PSF and then add Gaussian random noise.'''

    print 'Convolving PSF with image and adding noise...'
    print ' - Using sigma =', opts.sigma

    data_noisy = noise.add_noise(convolve.psf_convolve(data, psf,
                                 psf_type=opts.psf_type,
                                 data_format=opts.data_format), opts.sigma)

    return data_noisy


def run_script():

    '''Run the script'''

    print 'Running Add Noise Script...'

    # Read input files.
    if not isinstance(opts.random_seed, type(None)):
        np.random.seed(opts.random_seed)
        data = np.random.permutation(np.load(opts.input))[:opts.n_obj]
    else:
        data = np.load(opts.input)[:opts.n_obj]
    psf = np.load(opts.psf)

    if opts.psf_type == 'obj_var' and opts.n_obj != psf.shape[0]:
        raise RuntimeError('The number of PSFs does not match the number ' +
                           'of galaxies selected.')

    # Check format.
    check_data_format(data)

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
