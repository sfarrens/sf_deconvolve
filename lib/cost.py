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
    grad : class
        Gradient operator class
    wavelet : class, optional
        Wavelet operator class ("sparse" mode only)
    weights : np.ndarray, optional
        Array of wavelet thresholding weights ("sparse" mode only)
    lambda_reg : float, optional
        Low-rank regularization parameter ("lowr" mode only)
    mode : str {'lowr', 'sparse'}, optional
        Deconvolution mode (default is "lowr")
    positivity : bool, optional
        Option to test positivity contraint (defult is "True")
    verbose : bool
        Option for verbose output (default is "True")

    """

    def __init__(self, y, grad, wavelet=None, weights=None, lambda_reg=None,
                 mode='lowr', positivity=True, verbose=True):

        self.y = y
        self.grad = grad
        self.wavelet = wavelet
        self.lambda_reg = lambda_reg
        self.mode = mode
        self.positivity = positivity
        self.verbose = verbose
        self.update_weights(weights)

    def update_weights(self, weights):
        """Update weights

        Update the values of the wavelet threshold weights ("sparse" mode only)

        Parameters
        ----------
        weights : np.ndarray
            Array of wavelet thresholding weights

        """

        self.weights = weights

    def l2norm(self, x):
        """Calculate l2 norm

        This method returns the l2 norm error of the difference between the
        original data and the data obtained after optimisation

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float l2 norm value

        """

        l2_norm = np.linalg.norm(self.y - self.grad.op(x))

        if self.verbose:
            print ' - L2 NORM:', l2_norm

        return l2_norm

    def l1norm(self, x):
        """Calculate l1 norm

        This method returns the l1 norm error of the weighted wavelet
        coefficients

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float l1 norm value

        """

        x = self.weights * self.wavelet.op(x)

        l1_norm = np.sum(np.abs(x))

        if self.verbose:
            print ' - L1 NORM:', l1_norm

        return l1_norm

    def nucnorm(self, x):
        """Calculate nuclear norm

        This method returns the nuclear norm error of the deconvolved data in
        matrix form

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float nuclear norm value

        """

        x_prime = cube2matrix(x)

        nuc_norm = nuclear_norm(x_prime)

        if self.verbose:
            print ' - NUCLEAR NORM:', nuc_norm

        return nuc_norm

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

        if self.mode == 'all':
            cost = (0.5 * self.l2norm(x) ** 2 + self.l1norm(x) +
                    self.nucnorm(x))

        elif self.mode == 'sparse':
            cost = 0.5 * self.l2norm(x) ** 2 + self.l1norm(x)

        elif self.mode == 'lowr':
            cost = (0.5 * self.l2norm(x) ** 2 + self.lambda_reg *
                    self.nucnorm(x))

        elif self.mode == 'grad':
            cost = 0.5 * self.l2norm(x) ** 2

        return cost
