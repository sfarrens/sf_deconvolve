# -*- coding: utf-8 -*-

"""COST FUNCTION

This module the class for the sf_deconvolveCost cost function.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 24/07/2017

"""

import numpy as np
from sf_tools.math.matrix import nuclear_norm
from sf_tools.base.transform import cube2matrix


class sf_deconvolveCost(object):

    """Cost function class for sf_deonvolve

    This class implements the cost function for deconvolution

    Parameters
    ----------
    y : np.ndarray
        Input original data array
    operator : function
        Matrix operator function
    wavelet : class, optional
        Wavelet operator class ("sparse" mode only)
    weights : np.ndarray, optional
        Array of wavelet thresholding weights ("sparse" mode only)
    lambda_lowr : float, optional
        Low-rank regularization parameter ("lowr" mode only)
    lambda_psf : float, optional
        PSF estimate regularization parameter ("psf_unknown" grad_type only)
    mode : str {'lowr', 'sparse'}, optional
        Deconvolution mode (default is "lowr")
    positivity : bool, optional
        Option to test positivity contraint (defult is "True")
    verbose : bool
        Option for verbose output (default is "True")

    """

    def __init__(self, y, grad, wavelet=None, weights=None, lambda_lowr=None,
                 lambda_psf=1, mode='lowr', positivity=True, verbose=True):

        self.y = y
        self.grad = grad
        self.wavelet = wavelet
        self.weights = weights
        self.lambda_lowr = lambda_lowr
        self.lambda_psf = lambda_psf
        self.mode = mode
        self.positivity = positivity
        self.verbose = verbose

    def grad_comp(self, x):
        """Calculate gradient component of the cost

        This method returns the l2 norm error of the difference between the
        original data and the data obtained after optimisation

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float gradient cost component

        """

        l2_norm = np.linalg.norm(self.y - self.grad.H_op(x))

        if self.verbose:
            print ' - L2 NORM (Grad):', l2_norm

        return l2_norm

    def sparse_comp(self, x):
        """Calculate sparsity component of the cost

        This method returns the l1 norm error of the weighted wavelet
        coefficients

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float sparsity cost component

        """

        x = self.weights * self.wavelet.op(x)

        l1_norm = np.sum(np.abs(x))

        if self.verbose:
            print ' - L1 NORM:', l1_norm

        return l1_norm

    def lowr_comp(self, x):
        """Calculate low-rank component of the cost

        This method returns the nuclear norm error of the deconvolved data in
        matrix form

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float low-rank cost component

        """

        x_prime = cube2matrix(x)

        nuc_norm = nuclear_norm(x_prime)

        if self.verbose:
            print ' - NUCLEAR NORM:', nuc_norm

        return self.lambda_lowr * nuc_norm

    def psf_comp(self):
        """Calculate PSF estimation component of the cost

        This method returns the l2 norm error of the difference between the
        initial PSF and the estimated PSF

        Returns
        -------
        float PSF cost component

        """

        l2_norm = np.linalg.norm(self.grad._psf - self.grad._psf0)

        if self.verbose:
            print ' - L2 NORM (PSF):', l2_norm

        return self.lambda_psf * l2_norm

    def calc_cost(self, *args):
        """Get cost function

        This method calculates the cost

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float cost

        """

        x = args[0]

        if self.positivity and self.verbose:
            print ' - MIN(X):', np.min(x)

        cost = 0.5 * self.grad_comp(x) ** 2

        if self.mode in ('sparse', 'all'):
            cost += self.sparse_comp(x)

        elif self.mode in ('lowr', 'all'):
            cost += self.lowr_comp(x)

        if self.grad.grad_type == 'psf_unknown':
            cost += self.psf_comp()

        return cost
