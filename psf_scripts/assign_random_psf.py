#! /usr/bin/env Python

import numpy as np
import argparse as ap


##
#  Get global script arguments.
#
def get_opts():

    global opts

    parser = ap.ArgumentParser('SCRIPT OPTIONS:',
                               formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', dest='input', required=True,
                        help='Input PSF stack binary file name.')

    parser.add_argument('-o', '--output', dest='output', required=False,
                        help='Output file name.')

    parser.add_argument('-n', '--n_psf', dest='n_psf', type=int,
                        required=True, help='Number of PSFs.')

    opts = parser.parse_args()


##
#  Run the script.
#
def run_script():

    print 'Running random PSF assignment script.'

    # Read in the PSF stack.
    psf_stack = np.load(opts.input)

    # Select N random indices.
    index = np.random.randint(0, psf_stack.shape[0], opts.n_psf)

    # Create an output file name.
    if isinstance(opts.output, type(None)):
        opts.output = opts.input + '_' + str(opts.n_psf) + '_random_PSFs'

    # Save output to a binary file.
    np.save(opts.output, psf_stack[index])

    print ' - Output saved to: ' + opts.output + '.npy'
    print 'Script completed successfully.'


def main():

    get_opts()
    run_script()


if __name__ == "__main__":
    main()
