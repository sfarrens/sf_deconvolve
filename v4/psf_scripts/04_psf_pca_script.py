#!/Users/sfarrens/Documents/Library/anaconda/bin/python

# This script does the following:
# - Reads in the data map and PSF corresponding to a given layout.
# - Generates the principal components and coefficients of the PSF.
# - Results are saved to .npy files.

import numpy as np
import argparse as ap
from psf import *

path = '/Users/sfarrens/Documents/Projects/PSF/pixel_variant/data/'


def get_opts():

    global opts

    parser = ap.ArgumentParser('SCRIPT OPTIONS:',
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-n', '--n_obj', dest='n_obj', type=int,
                        required=False, help='Number of objects.')

    parser.add_argument('-t', '--threshold', dest='threshold', type=float,
                        required=False, default=0.95, help='PCA theshold.')

    parser.add_argument('--full_map_psf', dest='full_map_psf',
                        action='store_true', required=False,
                        help='Option to specify if the PSF is in data map '
                        'format.')

    opts = parser.parse_args()


def get_psf_pca():

    print '1- Finding principal components and coefficients of the PSF.'

    if opts.full_map_psf:
        form_tag = 'map_'
    else:
        form_tag = 'cube_'

    # Read the PSF stack from binary file.
    psf = np.load(path + 'deep_galaxy_psf_' + form_tag + str(opts.n_obj) +
                  '.npy')

    # Reshape the PSF stack and read in the corresponding data cube.
    if not opts.full_map_psf:
        psf = psf.reshape([np.prod(psf.shape[:2])] + list(psf.shape[2:]))
        data = np.load(path + 'deep_galaxy_cube_' + str(opts.n_obj) + '.npy')

    # Read in the relevant data map.
    else:
        data = np.load(path + 'deep_galaxy_map_' + str(opts.n_obj) + '.npy')

    # Calculate the principal components and coefficients of the PSF stack.
    psf_pcs = pca.psf_pca(psf, opts.threshold)

    psf_coef = pca.get_coef(psf, psf_pcs, shape=data.shape)

    if not opts.full_map_psf:
        psf_coef = np.transpose(psf_coef, axes=[1, 0, 2, 3])

    # Save the results to binary files.
    np.save(path + 'psf_pcs_' + form_tag + str(opts.n_obj) + '.npy', psf_pcs)
    np.save(path + 'psf_coef_' + form_tag + str(opts.n_obj) + '.npy', psf_coef)


def main():

    get_opts()
    get_psf_pca()


if __name__ == "__main__":
    main()
