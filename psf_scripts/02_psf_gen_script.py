#!/Users/sfarrens/Documents/Library/anaconda/bin/python

# This script does the following:
# - Reads in the full data cube of deep field galaxies and selects the
#   first N galaxies where N is set by the map layout.
# - Converts this set of galaxies into a data map.
# - Generates a pixel variant PSF for every pixel in the map.
# - Both the map and the PSF are saved to .npy files.

import numpy as np
import argparse as ap
from psf import *
from functions.extra_math import mfactor

path = '/Users/sfarrens/Documents/Projects/PSF/pixel_variant/data/'
data_file = 'deep_galaxy_catalog_clean.npy'


def get_opts():

    global opts

    parser = ap.ArgumentParser('SCRIPT OPTIONS:',
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-n', '--n_obj', dest='n_obj', type=int,
                        required=False, help='Number of objects.')

    parser.add_argument('-v', '--variance', dest='variance', nargs=2,
                        default=(2, 2), type=int, required=False,
                        help='PSF pixel variance.')

    parser.add_argument('--save_cube', dest='save_cube', action='store_true',
                        required=False, help='Option to save data cube.')

    parser.add_argument('--save_map', dest='save_map', action='store_true',
                        required=False, help='Option to save data map.')

    parser.add_argument('--full_map_psf', dest='full_map_psf',
                        action='store_true', required=False,
                        help='Option to save the PSF in data map format.')

    parser.add_argument('--no_psf_gen', dest='no_psf_gen',
                        action='store_false', required=False,
                        help='Option to turn off the PSF generation.')

    opts = parser.parse_args()


def do_stuff():

    print '1- Reading full clean image data cube from file.'

    # Read data cube binary and store first N objects.
    data_cube = np.load(path + data_file)[:opts.n_obj]

    # Save the sampled data cube sample to a binary file.
    if opts.save_cube:
        np.save(path + 'deep_galaxy_cube_' + str(opts.n_obj) + '.npy',
                data_cube)

    print '2- Constructing data map from cube sample.'

    # Find the most appropriate map layout for N objects.
    layout = mfactor(data_cube.shape[0])

    # Transform the data cube into a map.
    data_map = transform.cube2map(data_cube, layout)

    # Save the data map to a binary file.
    if opts.save_map:
        np.save(path + 'deep_galaxy_map_' + str(opts.n_obj) + '.npy', data_map)

    if opts.no_psf_gen:
        print '3- Constructing pixel variant PSF from map.'

        # Generate the pixel variant PSF stack.
        psf = psf_gen.pixel_var_psf(data_map.shape, opts.variance)

        # Reshape the PSF stack to data cube format and save it to binary file.
        # Notes: The default stack is the number of map pixels, followed by the
        # PSF shape. The data cube format requires that the default stack be
        # the number of galaxies, followed by the number of pixels per galaxy
        # image and finally the PSF shape (hence the strange indexing below).
        if not opts.full_map_psf:
            index = np.arange(psf.shape[0]).reshape(data_map.shape)
            index = transform.map2cube(index,
                                       layout=layout).reshape(index.size)
            psf = psf[index].reshape([data_cube.shape[0]] +
                                     [np.prod(data_cube.shape[1:])] +
                                     list(psf.shape[-2:]))
            np.save(path + 'deep_galaxy_psf_cube_' + str(opts.n_obj) + '.npy',
                    psf)

        # Just save the PSF in map format to a binary file.
        else:
            np.save(path + 'deep_galaxy_psf_map_' + str(opts.n_obj) + '.npy',
                    psf)


def main():

    get_opts()
    do_stuff()


if __name__ == "__main__":
    main()
