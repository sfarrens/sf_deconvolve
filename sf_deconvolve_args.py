# -*- coding: utf-8 -*-

"""PSF DECONVOLUTION ARGUMENTS

This module sets the arguments for deconvolution_script.py.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 12/12/2016

"""

import argparse as ap
from argparse import ArgumentDefaultsHelpFormatter as formatter


class ArgParser(ap.ArgumentParser):

    """Argument Parser

    This class defines a custom argument parser to override the
    defult convert_arg_line_to_args method from argparse.

    """

    def __init__(self, *args, **kwargs):

        super(ArgParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        """Convert argument line to arguments

        This method overrides the default method of argparse. It skips blank
        and comment lines, and allows .ini style formatting.

        Parameters
        ----------
        line : str
            Input argument string

        Yields
        ------
        str
            Argument strings

        """

        line = line.split()
        if line and line[0][0] not in ('#', ';'):
            if line[0][0] != '-':
                line[0] = '--' + line[0]
            if len(line) > 1 and '=' in line[0]:
                line = line[0].split('=') + line[1:]
            for arg in line:
                yield arg


def get_opts():

    """Get script options

    This method sets the PSF deconvolution script options.

    """

    # Set up argument parser

    parser = ArgParser(add_help=False, usage='%(prog)s [options]',
                       description='PSF Deconvolution Script',
                       formatter_class=formatter,
                       fromfile_prefix_chars='@')
    required = parser.add_argument_group('Required Arguments')
    optional = parser.add_argument_group('Optional Arguments')

    # Add arguments

    optional.add_argument('-h', '--help', action='help',
                          help='show this help message and exit')

    optional.add_argument('-v', '--version', action='version',
                          version='%(prog)s v3.2')

    required.add_argument('-i', '--input', dest='input', required=True,
                          help='Input noisy data file name.')

    required.add_argument('-p', '--psf', dest='psf', required=True,
                          help='PSF file name.')

    optional.add_argument('-o', '--output', dest='output',
                          required=False, help='Output file name.')

    optional.add_argument('-k', '--current_res', dest='current_res',
                          required=False,
                          help='Current deconvolution results file name.')

    optional.add_argument('-c', '--clean_data', dest='clean_data',
                          required=False, help='Clean data file name.')

    optional.add_argument('-r', '--random_seed', dest='random_seed',
                          type=int, required=False, help='Random seed.')

    optional.add_argument('--psf_type', dest='psf_type', required=False,
                          default='obj_var',
                          choices=('fixed', 'obj_var'),
                          help='PSF fortmat type. [fixed or obj_var]')

    optional.add_argument('--noise_est', dest='noise_est', type=float,
                          required=False, help='Initial noise estimate.')

    optional.add_argument('-m', '--mode', dest='mode', default='lowr',
                          choices=('all', 'sparse', 'lowr', 'grad'),
                          required=False, help='Option to specify the '
                          'optimisation mode. [all, sparse, lowr or grad]')

    optional.add_argument('--opt_type', dest='opt_type', default='condat',
                          choices=('condat', 'fwbw', 'gfwbw'), required=False,
                          help='Option to specify the optimisation method to'
                          'be implemented. [condat, fwbw or gfwbw]')

    optional.add_argument('--wavelet_type', dest='wavelet_type',
                          required=False, default='1', help='Wavelet type.')

    optional.add_argument('--wave_thresh_factor', dest='wave_tf', type=float,
                          nargs='+', required=False, default=[3.0, 3.0, 4.0],
                          help='Wavelet threshold factor.')

    optional.add_argument('--lowr_thresh_factor', dest='lowr_tf', type=float,
                          required=False, default=1,
                          help='Low rank threshold factor.')

    optional.add_argument('--lowr_type', dest='lowr_type',
                          required=False, default='standard',
                          help='Low rank type. [standard or ngole]')

    optional.add_argument('--lowr_thresh_type', dest='lowr_thresh_type',
                          required=False, default='hard',
                          help='Low rank threshold type. [soft or hard]')

    optional.add_argument('--n_reweights', dest='n_reweights',
                          type=int, required=False, default=1,
                          help='Number of reweightings.')

    optional.add_argument('--n_iter', dest='n_iter',
                          type=int, required=False, default=150,
                          help='Number of iterations.')

    optional.add_argument('--relax', dest='relax',
                          type=float, required=False, default=0.8,
                          help='Relaxation parameter (rho_n).')

    optional.add_argument('--condat_sigma', dest='condat_sigma',
                          type=float, required=False, default=0.5,
                          help='Condat proximal dual parameter.')

    optional.add_argument('--condat_tau', dest='condat_tau',
                          type=float, required=False, default=0.5,
                          help='Condat proximal dual parameter')

    optional.add_argument('--kernel', dest='kernel',
                          type=float, required=False,
                          help='Sigma value for Gaussian kernel.')

    optional.add_argument('--cost_window', dest='cost_window',
                          type=int, required=False, default=1,
                          help='Window to measure cost function.')

    optional.add_argument('--no_pos', dest='no_pos',
                          action='store_false', required=False,
                          help='Option to turn off postivity constraint.')

    optional.add_argument('--no_grad', dest='no_grad',
                          action='store_false', required=False,
                          help='Option to turn off gradinet calculation.')

    # Return the arguments

    return parser.parse_args()
