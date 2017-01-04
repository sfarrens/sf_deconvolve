#! /usr/bin/env python

import os
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from creepy.image.shape import *
from creepy.image.quality import *
from functions.string import extract_num
from functions.stats import gaussian_kernel

# from psf.transform import *


def get_opts():

    '''Get script arguments'''

    global opts

    formatter = ap.ArgumentDefaultsHelpFormatter

    parser = ap.ArgumentParser('SCRIPT OPTIONS:', formatter_class=formatter)

    parser.add_argument('-i', '--input', dest='input',
                        required=True, help='Input path.')

    parser.add_argument('-c', '--clean_data', dest='clean_data', required=True,
                        help='Clean data file name.')

    parser.add_argument('-n', '--noisy_data', dest='noisy_data',
                        required=False, help='Noisy data file name.')

    parser.add_argument('-p', '--psf', dest='psf', required=False,
                        help='PSF file name.')

    parser.add_argument('-a', '--average', dest='average', action='store_true',
                        required=False, help='Averaged results for samples.')

    parser.add_argument('-s', '--sigma', dest='sigma', required=False,
                        type=float, help='Sigma value for weights.')

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


def read_noisy_files(path, seed=None):

        vals = []

        for file in os.listdir(path):
            if isinstance(seed, type(None)):
                vals.append(np.load(path + file))
            elif 'rs' + seed in file:
                vals.append(np.load(path + file))

        return np.array(vals)


def add_weights_set(data_set, weight):

    return np.array([add_weights(data, weight) for data in
                    data_set]).reshape(data_set.shape)


def add_weights(data, weight):

    return np.array([x * weight for x in data])


def pseudo_inverse(image, kernel, weight=None):

    if isinstance(weight, type(None)):
        weight = np.ones(image.shape)

    y_hat = fftshift(fftn(image))
    h_hat = fftshift(fftn(kernel))
    h_hat_star = np.conj(h_hat)

    res = ((h_hat_star * y_hat) / (h_hat_star * h_hat) * weight)

    return np.real(fftshift(ifftn(ifftshift(res))))


def get_pi(noisy_data_set, psf, weights):

    return np.array([pseudo_inverse(y, h, weights) * weights for y, h in
                    zip(noisy_data_set, psf)])


def perform_test(random_seed=None):

    # READ DATA

    lowr_data = read_files(opts.input + '/lowr/', random_seed)
    sparse_data = read_files(opts.input + '/wave/', random_seed)

    if not isinstance(random_seed, type(None)):
        np.random.seed(int(random_seed))
        clean_data = np.load(opts.clean_data)
        clean_data = np.random.permutation(clean_data)[:lowr_data.shape[1]]
    else:
        clean_data = np.load(opts.clean_data)[:lowr_data.shape[1]]

    # ADD WEIGHTS

    if not isinstance(opts.sigma, type(None)):

        psf = np.load(opts.psf)
        noisy_data = read_noisy_files(opts.noisy_data, random_seed)

        gk = gaussian_kernel(clean_data[0].shape, opts.sigma)

        lowr_data = add_weights_set(lowr_data, gk)
        sparse_data = add_weights_set(sparse_data, gk)
        clean_data = add_weights(clean_data, gk)

        pi_data = np.array([get_pi(nd, psf, gk) for nd in noisy_data])

        c_test = np.array([ellipticity_atoms(x) for x in clean_data])

        crap = []
        for setn in range(6):

            p_test = np.array([ellipticity_atoms(x) for x in pi_data[setn]])
            err_p = np.array([np.linalg.norm(a - b) for a, b in zip(c_test,
                             p_test)])

            crap.append(err_p)

        crap = np.array(crap).ravel()
        print crap.shape

        print np.sum(crap >= 1.0) / float(crap.size) * 100

        exit()

    # CALCULATE ERRORS

    lowr_px_err = [nmse(clean_data, x) for x in lowr_data]
    sparse_px_err = [nmse(clean_data, x) for x in sparse_data]

    lowr_e_err = [e_error(clean_data, x) for x in lowr_data]
    sparse_e_err = [e_error(clean_data, x) for x in sparse_data]

    if not isinstance(opts.sigma, type(None)):

        pi_e_err = [e_error(clean_data, x) for x in pi_data]
        return np.array([sparse_px_err, lowr_px_err, sparse_e_err, lowr_e_err,
                        pi_e_err])

    else:
        return np.array([sparse_px_err, lowr_px_err, sparse_e_err, lowr_e_err])


def make_plots(data1, data2, data3=None, errors=None, fig_num=1,
               err_type='px'):

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
                 label='Low Rank')

        if not isinstance(data3, type(None)):

            plt.plot(sigma, data3, linestyle=':', linewidth=1.5, marker='o',
                     markersize=7, color='#7EBE7E', markeredgecolor='none',
                     label='Pseudo-Inverse')

    else:

        plt.errorbar(sigma, data1, yerr=errors[0], linestyle='-',
                     linewidth=1.5, elinewidth=1, marker='o', markersize=7,
                     color='#5AB4D1', markeredgecolor='none', label='Sparsity')

        plt.errorbar(sigma, data2, yerr=errors[1], linestyle='--',
                     linewidth=1.5, elinewidth=1, marker='o', markersize=7,
                     color='#C67DEE', markeredgecolor='none', label='Low-Rank')

        if not isinstance(data3, type(None)):

            plt.errorbar(sigma, data3, yerr=errors[1], linestyle=':',
                         linewidth=1.5, elinewidth=1, marker='o', markersize=7,
                         color='#7EBE7E', markeredgecolor='none',
                         label='Pseudo-Inverse')

    plt.xlabel('$\sigma$', fontsize=fontsize)

    if err_type == 'px':
        plt.ylabel('$P_{err}$', fontsize=fontsize)
        plt.title('Weighted Pixel Error for ' + str(n_im) + ' Galaxy Images')
        file_name = 'pixel_error_' + str(n_im)

    else:
        plt.ylabel('$\epsilon_{err}$', fontsize=fontsize)
        plt.title('Weighted Ellipticity Error')
        file_name = 'ellipticity_error_' + str(n_im)

    if not isinstance(opts.sigma, type(None)):
        file_name += '_' + str(int(opts.sigma))

    plt.ylim(0.0, 1.0)
    plt.legend()

    plt.savefig(file_name)
    print 'Plot saved to:', file_name + '.png'


def run_script():

    seeds = ['05', '10', '15', '20', '25', '30', '35', '40', '45', '50']

    if opts.average:

        res = np.array([perform_test(seed) for seed in seeds])
        results = np.mean(res, axis=0)
        errors = np.std(res, axis=0)

        make_plots(results[0], results[1], errors=errors[:2])

        if results.shape[0] == 5:
            make_plots(results[2], results[3], results[4], errors=errors[2:],
                       fig_num=2, err_type='e')

        else:
            make_plots(results[2], results[3], errors=errors[2:], fig_num=2,
                       err_type='e')

    else:

        results = perform_test()

        make_plots(results[0], results[1])

        if results.shape[0] == 5:
            make_plots(results[2], results[3], results[4], fig_num=2,
                       err_type='e')

        else:
            make_plots(results[2], results[3], fig_num=2, err_type='e')


def main():

    get_opts()
    run_script()

if __name__ == "__main__":
    main()
