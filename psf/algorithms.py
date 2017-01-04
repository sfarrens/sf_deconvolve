#  @file algorithms.py
#
#  ALGOIRTHM ROUTINES
#
#  Classes for defining algorithms for image reconstruction.
#  Based on work by Yinghao Ge and Fred Ngole.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from scipy.linalg import norm


##
#  Class for perfoming power method to calculate the spectral radius.
#
class PowerMethod():

    ##
    #  Method that initialises the class instance.
    #
    #  @param[in] operator: Input operator.
    #  @param[in] data_shape: Shape of the data.
    #  @param[in] auto_run: Option to automatically get the spectral radius on
    #  initalisaiton.
    #
    def __init__(self, operator, data_shape, auto_run=True):

        self.op = operator
        self.data_shape = data_shape
        if auto_run:
            self.get_spec_rad()

    ##
    #  Method that sets an initial guess for the values of x.
    #
    def set_initial_x(self):

        return np.random.random(self.data_shape)

    ##
    #  Method that initialises the class instance.
    #
    #  @param[in] tolerance: Tolerance for convergence.
    #  @param[in] max_iter: Maximum number of iterations.
    #
    def get_spec_rad(self, tolerance=1e-6, max_iter=150):

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
