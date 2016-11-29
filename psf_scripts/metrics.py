#! /usr/bin/env python

import os
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from creepy.image.shape import *
from creepy.image.quality import *
from functions.string import extract_num

from psf.transform import *


def get_opts():

    '''Get script arguments'''

    global opts

    formatter = ap.ArgumentDefaultsHelpFormatter

    parser = ap.ArgumentParser('SCRIPT OPTIONS:', formatter_class=formatter)

    parser.add_argument('-i', '--input', dest='input',
                        required=True, help='Input path.')

    parser.add_argument('-c', '--clean_data', dest='clean_data', required=True,
                        help='Clean data file name.')

    parser.add_argument('-a', '--average', dest='average', action='store_true',
                        required=False, help='Averaged results for samples.')

    parser.add_argument('-s', '--sigma', dest='sigma', required=False,
                        default=0.7, type=float,
                        help='Sigma value for weights.')

    opts = parser.parse_args()


def read_files(path, seed=None):

    vals = []

    for file in os.listdir(path):
        if file.endswith('primal.npy'):
            if isinstance(seed, type(None)):
                vals.append(np.load(path + file))
            elif 'rs' + seed in file:
                vals.append(np.load(path + file))

    return np.array(vals)


def add_weights(data):

    return np.array([gaussian_filter(x, sigma=opts.sigma) for set in data
                    for x in set]).reshape(data.shape)


def px_error2(x_og, x_rec):

    return np.linalg.norm(x_rec - x_og) ** 2 / np.linalg.norm(x_og) ** 2


def perform_test(random_seed=None):

    lowr_data = read_files(opts.input + '/lowr/', random_seed)
    sparse_data = read_files(opts.input + '/wave/', random_seed)

    lowr_data = add_weights(lowr_data)
    sparse_data = add_weights(sparse_data)

    if not isinstance(random_seed, type(None)):
        np.random.seed(int(random_seed))
        clean_data = np.load(opts.clean_data)
        clean_data = np.random.permutation(clean_data)[:lowr_data.shape[1]]
    else:
        clean_data = np.load(opts.clean_data)[:lowr_data.shape[1]]

    lowr_px_err = [nmse(clean_data, x) for x in lowr_data]
    sparse_px_err = [nmse(clean_data, x) for x in sparse_data]

    lowr_e_err = [e_error(clean_data, x) for x in lowr_data]
    sparse_e_err = [e_error(clean_data, x) for x in sparse_data]

    return np.array([sparse_px_err, lowr_px_err, sparse_e_err, lowr_e_err])


def make_plots(data1, data2, errors=None, fig_num=1, err_type='px'):

    n_im = extract_num(opts.input, format=int)

    sigma = [0.1, 0.2, 0.5, 0.7, 1.0, 1.5]

    fontsize = 18

    plt.figure(fig_num)

    if isinstance(errors, type(None)):

        plt.plot(sigma, data1, linestyle='-', linewidth=1.5,
                 marker='o', markersize=7, color='#5AB4D1',
                 markeredgecolor='none', label='Sparsity')

        plt.plot(sigma, data2, linestyle='--', linewidth=1.5, marker='o',
                 markersize=7, color='#C67DEE', markeredgecolor='none',
                 label='Low-Rank')

    else:

        plt.errorbar(sigma, data1, yerr=errors[0], linestyle='-',
                     linewidth=1.5, elinewidth=1, marker='o', markersize=7,
                     color='#5AB4D1', markeredgecolor='none', label='Sparsity')

        plt.errorbar(sigma, data2, yerr=errors[1], linestyle='--',
                     linewidth=1.5, elinewidth=1, marker='o', markersize=7,
                     color='#C67DEE', markeredgecolor='none', label='Low-Rank')

    plt.xlabel('$\sigma$', fontsize=fontsize)
    if err_type == 'px':
        plt.ylabel('$P_{err}$', fontsize=fontsize)
        plt.title('Weighted Pixel Error for ' + str(n_im) + ' Galaxy Images')
        file_name = 'pixel_error_' + str(n_im)
    else:
        plt.ylabel('$\epsilon_{err}$', fontsize=fontsize)
        plt.title('Weighted Ellipticity Error')
        file_name = 'ellipticity_error_' + str(n_im)

    plt.ylim(0.0, 1.0)
    plt.legend()

    plt.savefig(file_name)
    print 'Plot saved to:', file_name + '.png'


def run_script():

    # seeds = ['05', '10', '15', '20', '25']
    seeds = ['05', '10', '15', '20', '25', '30', '35', '40', '45', '50']

    if opts.average:

        res = np.array([perform_test(seed) for seed in seeds])
        results = np.average(res, axis=0)
        errors = np.std(res, axis=0)

        make_plots(results[0], results[1], errors=errors[:2])
        make_plots(results[2], results[3], errors=errors[2:], fig_num=2,
                   err_type='e')

    else:

        results = perform_test()

        make_plots(results[0], results[1])
        make_plots(results[2], results[3], fig_num=2, err_type='e')


def main():

    get_opts()
    run_script()

if __name__ == "__main__":
    main()
