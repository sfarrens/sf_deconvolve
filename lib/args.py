# -*- coding: utf-8 -*-

"""SF DECONVOLVE ARGUMENTS

This module sets the arguments for sf_deconvolve.py.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 2.4

:Date: 23/10/2017

"""

import argparse as ap
from argparse import ArgumentDefaultsHelpFormatter as formatter
from . import __version__


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


def get_opts(args=None):

    """Get script options

    This method sets the PSF deconvolution script options.

    Returns
    -------
    arguments namespace

    """

    # Set up argument parser
    parser = ArgParser(add_help=False, usage='%(prog)s [options]',
                       description='PSF Deconvolution Script',
                       formatter_class=formatter,
                       fromfile_prefix_chars='@')
    required = parser.add_argument_group('Required Arguments')
    optional = parser.add_argument_group('Optional Arguments')
    init = parser.add_argument_group(' * Initialisation')
    optimisation = parser.add_argument_group(' * Optimisation')
    lowrank = parser.add_argument_group(' * Low-Rank Aproximation')
    sparsity = parser.add_argument_group(' * Sparsity')
    psfest = parser.add_argument_group(' * PSF Estimation')
    shape = parser.add_argument_group(' * Shape Constraint')
    condat = parser.add_argument_group(' * Condat Algorithm')
    testing = parser.add_argument_group(' * Testing')
    hidden = parser.add_argument_group(' * Hidden Options')

    # Add arguments
    optional.add_argument('-h', '--help', action='help',
                          help='show this help message and exit')

    optional.add_argument('-v', '--version', action='version',
                          version='%(prog)s {}'.format(__version__))

    optional.add_argument('-q', '--quiet', action='store_true',
                          help='Suppress verbose.')

    required.add_argument('-i', '--input', required=True,
                          help='Input noisy data file name.')

    required.add_argument('-p', '--psf_file', required=True,
                          help='PSF file name.')

    hidden.add_argument('--psf_type', choices=('fixed', 'obj_var'),
                        default='obj_var', help=ap.SUPPRESS)

    optional.add_argument('-o', '--output', help='Output file name.')

    optional.add_argument('--output_format', choices={'npy', 'fits'},
                          default='npy', help='Output file format.')

    init.add_argument('-k', '--current_res',
                      help='Current deconvolution results file name.')

    hidden.add_argument('--primal', help=ap.SUPPRESS)

    init.add_argument('--noise_est', type=float,
                      help='Initial noise estimate.')

    optimisation.add_argument('-m', '--mode', default='lowr',
                              choices=('all', 'sparse', 'lowr', 'grad'),
                              help='Option to specify the regularisation '
                              'mode.')

    optimisation.add_argument('--opt_type', default='condat',
                              choices=('condat', 'fwbw', 'gfwbw'),
                              help='Option to specify the optimisation method '
                              'to be implemented.')

    optimisation.add_argument('--n_iter', type=int, default=150,
                              help='Number of iterations.')

    optimisation.add_argument('--cost_window', type=int, default=1,
                              help='Window to measure cost function.')

    optimisation.add_argument('--convergence', type=float,
                              default=3e-4, help='Convergence tolerance.')

    optimisation.add_argument('--no_pos', action='store_true',
                              help='Option to turn off postivity constraint.')

    optimisation.add_argument('--no_plots', action='store_true',
                              help='Suppress plots.')

    optimisation.add_argument('--grad_type', default='psf_known',
                              choices=('psf_known', 'psf_unknown', 'shape',
                                       'none'),
                              help='Option to specify the type of gradient.')

    optimisation.add_argument('--convolve_method', default='astropy',
                              choices=('astropy', 'scipy'),
                              help='Option to specify the convolution method.')

    lowrank.add_argument('--lowr_thresh_factor', type=float, default=1,
                         help='Low rank threshold factor.')

    lowrank.add_argument('--lowr_type', choices=('standard', 'ngole'),
                         default='standard', help='Low rank type.')

    lowrank.add_argument('--lowr_thresh_type', choices=('hard', 'soft'),
                         default='hard', help='Low rank threshold type.')

    sparsity.add_argument('--wavelet_type', default='1',
                          help='mr_transform wavelet type.')

    sparsity.add_argument('--wave_thresh_factor', type=float, nargs='+',
                          default=[3.0, 3.0, 4.0],
                          help='Wavelet threshold factor.')

    sparsity.add_argument('--n_reweights', type=int, default=1,
                          help='Number of reweightings.')

    psfest.add_argument('--psf_weights', help='PSF weights file name.')

    psfest.add_argument('--block_size', type=int, default=100,
                        help='Block size for alternating minimisation.')

    psfest.add_argument('--psf_sigma', type=float, nargs='?', const=None,
                        default=0.5, help='Condat proximal dual parameter.')

    psfest.add_argument('--psf_tau', type=float, nargs='?', const=None,
                        default=0.5, help='Condat proximal primal parameter')

    psfest.add_argument('--psf_relax', type=float, default=0.8,
                        help='Relaxation parameter (rho_n).')

    condat.add_argument('--relax', type=float, default=0.8,
                        help='Relaxation parameter (rho_n).')

    condat.add_argument('--condat_sigma', type=float, nargs='?', const=None,
                        default=0.5, help='Condat proximal dual parameter.')

    condat.add_argument('--condat_tau', type=float, nargs='?', const=None,
                        default=0.5, help='Condat proximal primal parameter')

    testing.add_argument('-c', '--clean_data', help='Clean data file name.')

    testing.add_argument('-r', '--random_seed', type=int, help='Random seed.')

    testing.add_argument('--true_psf', help='True PSFs file name.')

    testing.add_argument('--kernel', type=float,
                         help='Sigma value for Gaussian kernel.')

    testing.add_argument('--metric', choices=('mean', 'median'),
                         default='median', help='Metric to average errors.')

    # Return the argument namespace
    return parser.parse_args(args)
