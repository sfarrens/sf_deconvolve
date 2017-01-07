#!/Users/sfarrens/Documents/Library/anaconda/bin/python

import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
from psf.convolve import *
from psf.transform import *
from functions.extra_math import mfactor


def get_opts():

    '''Get script arguments'''

    global opts

    parser = ap.ArgumentParser('SCRIPT OPTIONS:',
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-r', '--rec_data', dest='rec_data',
                        required=True, help='Reconstruction data file name.')

    parser.add_argument('-n', '--noise_data', dest='noise_data',
                        required=True, help='Noisy data file name.')

    parser.add_argument('-p', '--psf', dest='psf', required=True,
                        help='PSF file name.')

    parser.add_argument('--psf_type', dest='psf_type', required=False,
                        default='fixed', help='PSF fortmat type. ' +
                        '[fixed, obj_var or pix_var]')

    parser.add_argument('-o', '--output', dest='output',
                        required=False, help='Output file name.')

    parser.add_argument('-l', '--layout', dest='layout', type=int, nargs=2,
                        required=False, help='Map layout. [Map format only]')

    parser.add_argument('--data_format', dest='data_format', default='cube',
                        required=False, help='Data format.')

    opts = parser.parse_args()


def make_plots(data):

    '''Make some nice plots'''

    plt.subplot(221)
    plt.imshow(data[0], cmap='gray')
    plt.title('Noisy Data')
    plt.subplot(222)
    plt.imshow(data[1], cmap='gray')
    plt.title('Reconstruction')
    plt.subplot(223)
    plt.imshow(data[2], cmap='gray')
    plt.title('Convolved Reconstruction')
    plt.subplot(224)
    plt.imshow(data[3])
    plt.colorbar()
    plt.title('Residual')

    if isinstance(opts.output, type(None)):
        output_file = opts.rec_data + '_residual_plot.pdf'

    else:
        output_file = opts.output

    plt.savefig(output_file)
    print 'Output saved to:', output_file


def run_scipt():

    '''Run the script'''

    print 'Running Check Residual Script...'

    # Read data files.
    data_rec = np.load(opts.rec_data)
    data_noisy = np.load(opts.noise_data)
    psf = np.load(opts.psf)

    # Convolve PSF with reconstruction.
    data_conv = convolve(data_rec, psf, psf_type=opts.psf_type,
                         data_format=opts.data_format)

    # Convert cube data to map.
    if opts.data_format == 'cube':

        if isinstance(opts.layout, type(None)):
            opts.layout = tuple(mfactor(data_noisy.shape[0]))

        data_rec = cube2map(data_rec, opts.layout)
        data_noisy = cube2map(data_noisy, opts.layout)
        data_conv = cube2map(data_conv, opts.layout)

    # Calculate the residual.
    residual = (data_noisy - data_conv)

    # Return all data.
    return (data_noisy, data_rec, data_conv, residual)


def main():

    get_opts()
    data = run_scipt()
    make_plots(data)


if __name__ == "__main__":
    main()
