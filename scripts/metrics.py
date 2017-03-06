#! /usr/bin/env python

import os
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from psf.other_methods import pseudo_inverse
from creepy.image.shape import ellipticity_atoms
from creepy.image.quality import nmse, e_error
from functions.string import extract_num
from functions.stats import gaussian_kernel, psnr_stack
from web_code.gaussian_fit import test_fitgaussian


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

    parser.add_argument('-s', '--sigma', dest='sigma', required=False,
                        type=float, help='Sigma value for weights.')

    parser.add_argument('-g', '--pi_kernel', dest='pi_kernel', required=False,
                        type=float, help='Sigma value for pseudo inverse '
                        'weights.')

    parser.add_argument('-m', '--metric', dest='metric', required=False,
                        choices=('mean', 'median'), default='median',
                        help='Metric for averaging results.')

    opts = parser.parse_args()


def read_files(path, seed=None):

    vals = []

    for file in os.listdir(path):
        if file.endswith('primal.npy'):
            if isinstance(seed, type(None)):
                vals.append(np.load(path + file))
            elif 'rand' + seed in file:
                vals.append(np.load(path + file))

    return np.array(vals)


def read_noisy_files(path, seed=None):

        vals = []

        for file in os.listdir(path):
            if 'log' not in file:
                if isinstance(seed, type(None)):
                    vals.append(np.load(path + file))
                elif 'rand' + seed in file:
                    vals.append(np.load(path + file))

        return np.array(vals)


def add_weights_set(data_set, weight):

    return np.array([add_weights(data, weight) for data in
                    data_set]).reshape(data_set.shape)


def add_weights(data, weight):

    return np.array([x * weight for x in data])


def get_pi(noisy_data_set, psf, weights):

    return np.array([pseudo_inverse(y, h, weights) for y, h in
                    zip(noisy_data_set, psf)])


def get_fwhm(data):

    params = np.array([test_fitgaussian(i) for i in data])
    fwhm = 2.355 * np.mean(params[:, -2:], axis=1)

    return fwhm


def get_fwhm_error(data, clean, metric):

    return metric([[np.linalg.norm(a - b) for a, b in zip(clean, c)]
                  for c in data], axis=1)


def perform_test(random_seed=None):

    # READ DATA

    lowr_data = read_files(opts.input + '/lowr_norm/', random_seed)
    sparse_data = read_files(opts.input + '/sparse_norm/', random_seed)

    if not isinstance(random_seed, type(None)):
        np.random.seed(int(random_seed))
        clean_data = np.load(opts.clean_data)
        clean_data = np.random.permutation(clean_data)[:lowr_data.shape[1]]
    else:
        clean_data = np.load(opts.clean_data)[:lowr_data.shape[1]]

    # ADD WEIGHTS

    if not isinstance(opts.sigma, type(None)):

        gk = gaussian_kernel(clean_data[0].shape, opts.sigma, norm='sum')

        lowr_data = add_weights_set(lowr_data, gk)
        sparse_data = add_weights_set(sparse_data, gk)
        clean_data = add_weights(clean_data, gk)

    # CALCULATE FWHM

    clean_fwhm = get_fwhm(clean_data)
    lowr_fwhm = np.array([get_fwhm(x) for x in lowr_data])
    sparse_fwhm = np.array([get_fwhm(x) for x in sparse_data])

    # CALCULATE ERRORS

    if opts.metric == 'median':
        metric = np.median
    elif opts.metric == 'mean':
        metric = np.mean

    lowr_px_err = [nmse(clean_data, x, metric) for x in lowr_data]
    sparse_px_err = [nmse(clean_data, x, metric) for x in sparse_data]

    lowr_e_err = [e_error(clean_data, x, metric) for x in lowr_data]
    sparse_e_err = [e_error(clean_data, x, metric) for x in sparse_data]

    lowr_psnr = [psnr_stack(clean_data, x, metric) for x in lowr_data]
    sparse_psnr = [psnr_stack(clean_data, x, metric) for x in sparse_data]

    lowr_fwhm_error = get_fwhm_error(lowr_fwhm, clean_fwhm, metric)
    sparse_fwhm_error = get_fwhm_error(sparse_fwhm, clean_fwhm, metric)

    res = np.array([sparse_px_err, lowr_px_err, sparse_psnr, lowr_psnr,
                    sparse_fwhm_error, lowr_fwhm_error, sparse_e_err,
                    lowr_e_err])

    # CALCULATE PSEUDO-INVERSE

    if not isinstance(opts.pi_kernel, type(None)):

        psf = np.load(opts.psf)
        noisy_data = read_noisy_files(opts.noisy_data, random_seed)

        gk_pi = gaussian_kernel(clean_data[0].shape, opts.pi_kernel,
                                norm='sum')

        pi_data = np.array([get_pi(nd, psf, gk_pi) for nd in noisy_data])
        if not isinstance(opts.sigma, type(None)):
            pi_data = add_weights_set(pi_data, gk)
        pi_e_err = [e_error(clean_data, x, metric) for x in pi_data]
        res = np.vstack([res, pi_e_err])

    return res


def make_plots(data1, data2, data3=None, errors=None, fig_num=1,
               err_type='px'):

    n_im = extract_num(opts.input, format=int)

    snr = [1.0, 2.0, 3.0, 5.0, 10.0]

    fontsize = 18

    fig = plt.figure(fig_num)

    if isinstance(errors, type(None)):

        plt.plot(snr, data1, linestyle='-', linewidth=1.5,
                 marker='o', markersize=7, color='#5AB4D1',
                 markeredgecolor='none', label='Sparsity')

        plt.plot(snr, data2, linestyle='--', linewidth=1.5, marker='o',
                 markersize=7, color='#C67DEE', markeredgecolor='none',
                 label='Low Rank')

        if not isinstance(data3, type(None)):

            plt.plot(snr, data3, linestyle=':', linewidth=1.5, marker='o',
                     markersize=7, color='#7EBE7E', markeredgecolor='none',
                     label='Pseudo-Inverse')

    else:

        plt.errorbar(snr, data1, yerr=errors[0], linestyle='-',
                     linewidth=1.5, elinewidth=1, marker='o', markersize=7,
                     color='#5AB4D1', markeredgecolor='none', label='Sparsity')

        plt.errorbar(snr, data2, yerr=errors[1], linestyle='--',
                     linewidth=1.5, elinewidth=1, marker='o', markersize=7,
                     color='#C67DEE', markeredgecolor='none', label='Low-Rank')

        if not isinstance(data3, type(None)):

            plt.errorbar(snr, data3, yerr=errors[1], linestyle=':',
                         linewidth=1.5, elinewidth=1, marker='o', markersize=7,
                         color='#7EBE7E', markeredgecolor='none',
                         label='Pseudo-Inverse')

    plt.xlabel('SNR', fontsize=fontsize)
    plt.xscale('log')
    plt.xticks(snr)
    fig.axes[0].get_xaxis().set_major_formatter(ScalarFormatter())

    if err_type == 'px':
        if not isinstance(opts.sigma, type(None)):
            plt.ylabel('$WP_{err}$', fontsize=fontsize)
            plt.title('Weighted Pixel Error for ' + str(n_im) +
                      ' Galaxy Images')
        else:
            plt.ylabel('$P_{err}$', fontsize=fontsize)
            plt.title('Pixel Error for ' + str(n_im) + ' Galaxy Images')
        plt.ylim(0.0, 0.3)
        file_name = 'pixel_error_' + str(n_im)
        plt.yticks([0.0, 0.1, 0.2, 0.3])

    elif err_type == 'psnr':
        plt.ylabel('PSNR', fontsize=fontsize)
        plt.title('PSNR for ' + str(n_im) + ' Galaxy Images')
        plt.ylim(30.0, 40.0)
        file_name = 'psnr_' + str(n_im)

    elif err_type == 'fwhm':
        plt.ylabel('FWHM Error', fontsize=fontsize)
        plt.title('FWHM Error for ' + str(n_im) + ' Galaxy Images')
        plt.ylim(0.0, 0.5)
        file_name = 'fwhm_' + str(n_im)

    elif err_type == 'e':
        plt.ylabel('$\epsilon_{err}$', fontsize=fontsize)
        plt.title('Ellipticity Error for ' + str(n_im) + ' Galaxy Images')
        plt.ylim(0.0, 1.0)
        file_name = 'ellipticity_error_' + str(n_im)

    if not isinstance(opts.sigma, type(None)):
        file_name += '_' + str(int(opts.sigma))

    if not isinstance(opts.pi_kernel, type(None)):
        file_name += '_' + str(int(opts.pi_kernel))

    file_name += '_' + opts.metric

    plt.legend()

    plt.savefig(file_name)
    print 'Plot saved to:', file_name + '.png'


def run_script():

    seeds = ['05', '10', '15', '20', '25', '30', '35', '40', '45', '50']


    if '10000' in opts.input:

        results = perform_test()

        make_plots(results[0], results[1], fig_num=1, err_type='px')
        make_plots(results[2], results[3], fig_num=2, err_type='psnr')
        make_plots(results[4], results[5], fig_num=3, err_type='fwhm')

        if results.shape[0] == 9:
            make_plots(results[6], results[7], results[8], fig_num=4,
                       err_type='e')

        else:
            make_plots(results[6], results[7], fig_num=4, err_type='e')

    else:

        res = np.array([perform_test(seed) for seed in seeds])
        results = np.mean(res, axis=0)
        errors = np.std(res, axis=0)

        make_plots(results[0], results[1], errors=errors[:2], fig_num=1,
                   err_type='px')
        make_plots(results[2], results[3], errors=errors[2:4], fig_num=2,
                   err_type='psnr')
        make_plots(results[4], results[5], errors=errors[4:6], fig_num=3,
                   err_type='fwhm')

        if results.shape[0] == 9:
            make_plots(results[6], results[7], results[8], errors=errors[6:],
                       fig_num=4, err_type='e')

        else:
            make_plots(results[6], results[7], errors=errors[6:], fig_num=4,
                       err_type='e')


def main():

    get_opts()
    run_script()


if __name__ == "__main__":
    main()
