#! /usr/bin/env Python
# -*- coding: utf-8 -*-

"""TEST DECONVOLUTION SCRIPT

This module is executable and contains methods for the results of
deconvolution.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 16/01/2016

"""

import numpy as np
import argparse as ap
from psf.tests import test_deconvolution


def get_opts():
    """Get script arguments

    This method sets the script arguments

    """

    global opts

    formatter = ap.ArgumentDefaultsHelpFormatter

    parser = ap.ArgumentParser(add_help=False, usage='%(prog)s [options]',
                               description='Test Deconvolution Script',
                               formatter_class=formatter)
    required = parser.add_argument_group('Required Arguments')
    optional = parser.add_argument_group('Optional Arguments')

    required.add_argument('-i', '--input', dest='input',
                          required=True, help='Input path.')

    required.add_argument('-c', '--clean_data', dest='clean_data',
                          required=True, help='Clean data file name.')

    optional.add_argument('-h', '--help', action='help',
                          help='show this help message and exit')

    optional.add_argument('-r', '--random_seed', dest='random_seed',
                          type=int, required=False, help='Random seed.')

    optional.add_argument('-k', '--kernel', dest='kernel',
                          type=float, required=False,
                          help='Sigma value for Gaussian kernel.')

    optional.add_argument('-m', '--metric', dest='metric',
                          required=False, default='mean',
                          help='Metric for averaging.')

    opts = parser.parse_args()


def run_script():
    """Run script

    This method runs the script

    """

    deconv_data = np.load(opts.input)
    image_errors = test_deconvolution(deconv_data, opts.clean_data,
                                      opts.random_seed, opts.kernel,
                                      opts.metric)

    print ' - Clean Data:', opts.clean_data
    print ' - Random Seed:', opts.random_seed
    print ' - Pixel Error:', image_errors[0]
    print ' - Shape Error:', image_errors[1]
    print ' - PSNR:', image_errors[2]


def main():

    get_opts()
    run_script()


if __name__ == "__main__":
    main()
