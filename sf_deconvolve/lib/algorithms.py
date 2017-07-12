# -*- coding: utf-8 -*-

"""ALGORITHM CLASSES

This module contains classes for defining basic algorithms

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 05/01/2017

"""


import numpy as np
from scipy.linalg import norm


class PowerMethod(object):
    """Power method class

    This method performs implements power method to calculate the spectral
    radius of the input data

    Parameters
    ----------
    operator : class
        Operator class instance
    data_shape : tuple
        Shape of the data array
    auto_run : bool
        Option to automatically calcualte the spectral radius upon
        initialisation

    """

    def __init__(self, operator, data_shape, auto_run=True):

        self.op = operator
        self.data_shape = data_shape
        if auto_run:
            self.get_spec_rad()

    def set_initial_x(self):
        """Set initial value of x

        This method sets the initial value of x to an arrray of random values

        """

        return np.random.random(self.data_shape)

    def get_spec_rad(self, tolerance=1e-6, max_iter=10):
        """Get spectral radius

        This method calculates the spectral radius

        Parameters
        ----------
        tolerance : float, optional
            Tolerance threshold for convergence (default is "1e-6")
        max_iter : int, optional
            Maximum number of iterations

        """

        # Set (or reset) values of x.
        x_old = self.set_initial_x()

        # Iterate until the L2 norm of x converges.
        for i in xrange(max_iter):

            x_new = self.op(x_old) / norm(x_old)

            if(np.abs(norm(x_new) - norm(x_old)) < tolerance):
                print (' - Power Method converged after %d iterations!' %
                       (i + 1))
                break

            elif i == max_iter - 1:
                print (' - Power Method did not converge after %d '
                       'iterations!' % max_iter)

            np.copyto(x_old, x_new)

        self.spec_rad = norm(x_new)
        self.inv_spec_rad = 1.0 / self.spec_rad
