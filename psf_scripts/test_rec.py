#!/Users/sfarrens/Documents/Library/anaconda/bin/python

import numpy as np
import argparse as ap
from functions.shape import Ellipticity

# Script to test the reconstructed images and ellipticities.


def get_opts():

    '''Get script arguments'''

    global opts

    formatter = ap.ArgumentDefaultsHelpFormatter

    parser = ap.ArgumentParser('SCRIPT OPTIONS:', formatter_class=formatter)

    parser.add_argument('-i', '--input', dest='input',
                        required=True, help='Input data file name.')

    parser.add_argument('-c', '--clean_data', dest='clean_data', required=True,
                        help='Clean data file name.')

    parser.add_argument('-r', '--random_seed', dest='random_seed',
                        type=int, required=False, help='Random seed.')

    parser.add_argument('-s', '--sigma_ellip', dest='sigma_ellip',
                        type=float, default=1000.0, required=False,
                        help='Sigma for ellipticities.')

    opts = parser.parse_args()


def perform_test(opts):

    rec_data = np.load(opts.input)

    if not isinstance(opts.random_seed, type(None)):
        np.random.seed(opts.random_seed)
        clean_data = np.load(opts.clean_data)
        clean_data = np.random.permutation(clean_data)[:rec_data.shape[0]]
    else:
        clean_data = np.load(opts.clean_data)[:rec_data.shape[0]]

    clean_ellip = np.array([Ellipticity(x, opts.sigma_ellip).e
                            for x in clean_data])

    rec_ellip = np.array([Ellipticity(x, opts.sigma_ellip).e
                          for x in rec_data])

    px_err = (np.linalg.norm(clean_data - rec_data) /
              np.linalg.norm(clean_data))

    ellip_err = (np.linalg.norm(clean_ellip - rec_ellip) /
                 np.linalg.norm(clean_ellip))

    return (px_err, ellip_err)


def main():

    get_opts()
    print 'Errors:', perform_test(opts)

if __name__ == "__main__":
    main()
