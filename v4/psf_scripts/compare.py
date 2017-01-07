#! /usr/bin/env Python

#  @file compare.py
#
#  Script to compare an image reconstruction with the original image
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2016
#

import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import make_axes
from matplotlib.colors import BoundaryNorm
from psf.convolve import *
from psf.transform import *
from functions.extra_math import mfactor


##
#  Function to set script argumnets.
#
def get_opts():

    '''Get script arguments'''

    global opts

    format = ap.ArgumentDefaultsHelpFormatter

    parser = ap.ArgumentParser('SCRIPT OPTIONS:', formatter_class=format)

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

    parser.add_argument('--vmin', dest='vmin', required=False, default=0.0,
                        type=float, help='Minimum pixel value.')

    parser.add_argument('--vmax', dest='vmax', required=False, type=float,
                        help='Maximum pixel value.')

    parser.add_argument('--vmax_mode', dest='vmax_mode', required=False,
                        default='obj', choices=('obj', 'all'),
                        help='Maximum pixel value mode.')

    parser.add_argument('--levels', dest='levels', required=False, type=int,
                        help='Number of colour levels.')

    parser.add_argument('--cmap', dest='cmap', required=False,
                        default='gist_stern', help='Matplotlib colour map.')

    parser.add_argument('--interp', dest='interp', required=False,
                        default='nearest', help='Interpolation.')

    parser.add_argument('--white', dest='white', required=False,
                        action='store_true',
                        help='Display zero values in white.')

    opts = parser.parse_args()


##
#  Function to plot the images.
#
#  @param[in] data: Input data array.
#  @param[in] output_file: Output file name.
#
def make_plot(data, output_file):

    # Set plot parameters.
    titles = ('Clean Data $(X)$', 'Noisy Data $(Y)$',
              'Reconstruction $(\hat{X})$', 'Residual $(|Y-M\hat{X}|)$')

    # Set vmax when mode = 'obj'.
    if opts.vmax_mode == 'obj':
        opts.vmax = np.max(data)

    # Set the colour levels.
    if isinstance(opts.levels, type(None)):
        colourbin = 0.05
    else:
        colourbin = (opts.vmax - opts.vmin) / opts.levels

    boundaries = np.arange(opts.vmin, opts.vmax, colourbin)
    norm = BoundaryNorm(boundaries, cm.get_cmap(name=opts.cmap).N)

    # Make plot.
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for ax, x, title in zip(axes.flat, data, titles):
        if opts.white:
            x[x == 0.0] = np.nan
        im = ax.imshow(x, norm=norm, cmap=opts.cmap, interpolation=opts.interp)
        ax.set_title(title)
        ax.set_adjustable('box-forced')
    cax, kw = make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cax, **kw)

    # Output file.
    plt.savefig(output_file)
    plt.close(fig)
    print 'Output saved to:', output_file


##
#  Function to round pixel values to 6 decimal places.
#
#  @param[in] data: Input data.
#
#  @return Rounded data.
#
def make_tidy(data):

    data = np.around(data, decimals=6)

    return data


##
#  Function to run the script.
#
def run_script():

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
                      psf_type='obj_var', data_format='cube'))

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


##
#  Main function.
#
def main():

    get_opts()
    run_script()


if __name__ == "__main__":
    main()
