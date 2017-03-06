#! /usr/bin/env Python
# -*- coding: utf-8 -*-

"""COMPARISON SCRIPT

This module is executable and compares the results of the deconvolution script
with the original and observed data.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.1

:Date: 10/01/2017

"""

import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
from matplotlib.colorbar import make_axes
from matplotlib.colors import BoundaryNorm
from psf.convolve import psf_convolve


def get_opts():

    """Get script arguments

    This method sets the script arguments

    """

    global opts

    formatter = ap.ArgumentDefaultsHelpFormatter

    parser = ap.ArgumentParser('SCRIPT OPTIONS:', formatter_class=formatter)

    parser.add_argument('-i', '--input', dest='rec_data',
                        required=True, help='Reconstruction data file name.')

    parser.add_argument('-n', '--noise_data', dest='noise_data',
                        required=True, help='Noisy data file name.')

    parser.add_argument('-c', '--clean_data', dest='clean_data',
                        required=True, help='Clean data file name.')

    parser.add_argument('-p', '--psf', dest='psf',
                        required=True, help='PSF file name.')

    parser.add_argument('-r', '--random_seed', dest='random_seed',
                        type=int, required=False, help='Random seed.')

    parser.add_argument('--obj', dest='obj', required=False, default=0,
                        help='Object stack number.')

    parser.add_argument('--vmin', dest='vmin', required=False, default=0.0005,
                        type=float, help='Minimum pixel value.')

    parser.add_argument('--vmax', dest='vmax', required=False, type=float,
                        help='Maximum pixel value.')

    parser.add_argument('--vmax_mode', dest='vmax_mode', required=False,
                        default='obj', choices=('obj', 'all'),
                        help='Maximum pixel value mode.')

    parser.add_argument('--levels', dest='levels', required=False, type=int,
                        help='Number of colour levels.')

    parser.add_argument('--cmap', dest='cmap', required=False,
                        default='autumn', help='Matplotlib colour map.')

    parser.add_argument('--interp', dest='interp', required=False,
                        default='nearest', help='Interpolation.')

    parser.add_argument('--white', dest='white', required=False,
                        action='store_true',
                        help='Display zero values in white.')

    opts = parser.parse_args()


def make_plot(data, output_file):
    """Make plots

    This method generates the comparison plots

    Parameters
    ----------
    data : np.ndarray
        Input data, 3D array
    output_file : str
        Output file name

    """

    # Set plot parameters.
    titles = ('Clean Data $(X)$', 'Noisy Data $(Y)$',
              'Reconstruction $(\hat{X})$', 'Residual $(|Y-H\hat{X}|)$')

    # Set vmax when mode = 'obj'.
    if opts.vmax_mode == 'obj':
        opts.vmax = np.max(data)

    # Set the colour levels.
    if isinstance(opts.levels, type(None)):
        colourbin = 0.0001
    else:
        colourbin = (opts.vmax - opts.vmin) / opts.levels

    cmap = plt.cm.get_cmap(name=opts.cmap)
    cmap.set_under('k')

    boundaries = np.arange(opts.vmin, opts.vmax, colourbin)
    ticks = np.arange(opts.vmin, opts.vmax + 0.05, 0.05)
    norm = BoundaryNorm(boundaries, cmap.N)

    # Make plot.
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for ax, x, title in zip(axes.flat, data, titles):
        if opts.white:
            x[x == 0.0] = np.nan
        # im = ax.imshow(np.abs(x), norm=norm, cmap=opts.cmap,
        #                interpolation=opts.interp)
        im = ax.imshow(np.abs(x), vmin=opts.vmin, vmax=opts.vmax,
                       cmap=opts.cmap, interpolation=opts.interp)
        ax.set_title(title)
        ax.set_adjustable('box-forced')
    cax, kw = make_axes([ax for ax in axes.flat])
    plt.colorbar(im, ticks=ticks, cax=cax, **kw)

    # Output file.
    plt.savefig(output_file)
    plt.close(fig)
    print 'Output saved to:', output_file


def make_tidy(data):
    """Make tidy

    This method rounds pixel values to 6 decimal places

    Parameters
    ----------
    data : np.ndarray
        Input data, 3D array

    """

    data = np.around(data, decimals=6)

    return data


def run_script():
    """Run script

    This method runs the script

    """

    # Read in data files.
    data_noisy = np.load(opts.noise_data)
    data_rec = np.load(opts.rec_data)
    psf = np.load(opts.psf)

    n_obj = data_rec.shape[0]
    pad = '0' + str(len(str(n_obj)))

    if not isinstance(opts.random_seed, type(None)):
        np.random.seed(opts.random_seed)
        data_clean = np.random.permutation(np.load(opts.clean_data))[:n_obj]
    else:
        data_clean = np.load(opts.clean_data)[:n_obj]

    # Calculate the residual.
    residual = np.abs(data_noisy - psf_convolve(data_rec, psf,
                      psf_type='obj_var'))

    # Create a data list.
    data = [data_clean, data_noisy, data_rec, residual]

    # Tidy up the data list.
    data = np.transpose([make_tidy(x) for x in data], axes=[1, 0, 2, 3])

    # Check if vmax is provided.
    if not isinstance(opts.vmax, type(None)):
        opts.vmax_mode = None

    # Set vmax when vmax_mode = 'all'.
    if opts.vmax_mode == 'all':
        opts.vmax = np.max(data)

    # Make the plot(s).
    if opts.obj == 'all':
        for i in range(n_obj):
            file_name = (opts.rec_data + '_' + format(i, pad) + '_compare.pdf')
            make_plot(data[i], file_name)

    else:
        opts.obj = int(opts.obj)
        file_name = (opts.rec_data + '_' + format(opts.obj, pad) +
                     '_compare.pdf')
        make_plot(data[opts.obj], file_name)


def main():

    get_opts()
    run_script()


if __name__ == "__main__":
    main()
