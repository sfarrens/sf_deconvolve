#  @file cost.py
#
#  COST FUNCTIONS
#
#  Classes of cost functions for optimization.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from psf.transform import *
from functions.matrix import nuclear_norm
from plotting import *


##
#  Cost function test.
#
class costTest():

    def __init__(self, y, operator):

        self.y = y
        self.op = operator

    def get_cost(self, x):

        return np.linalg.norm(self.y - self.op(x))


##
#  Basic cost function with l2 norm.
#
class costFunction():

    ##
    #  Class initializer.
    #
    #  @param[in] y: Original data.
    #  @param[in] grad: Gradient operator class.
    #  @param[in] wavelet: Wavelet operator class.
    #  @param[in] weights: Input weights.
    #  @param[in] lambda_reg: Regularization factor.
    #  @param[in] print_cost: Option to print cost function value.
    #
    def __init__(self, y, grad, wavelet=None, weights=None,
                 lambda_reg=None, mode='all', data_format='cube',
                 positivity=True, tolerance=1e-4, print_cost=True,
                 live_plotting=False, window=5, total_it=None):

        self.y = y
        self.grad = grad
        self.wavelet = wavelet
        self.lambda_reg = lambda_reg
        self.mode = mode
        self.data_format = data_format
        self.positivity = positivity
        self.update_weights(weights)
        self.cost = 1e6
        self.cost_list = []
        self.x_list = []
        self.tolerance = tolerance
        self.print_cost = print_cost
        self.iteration = 0
        self.liveplot = live_plotting
        self.total_it = total_it

        self.window = window
        self.test_list = []

    ##
    #  Method to update current values of the weights.
    #
    #  @param[in] weights: Input weights.
    #
    def update_weights(self, weights):

        self.weights = weights

    ##
    #  Method to calculate the l2 norm of the reconstruction.
    #
    #  @param[in] x: Input data.
    #
    def l2norm(self, x):

        l2_norm = np.linalg.norm(self.y - self.grad.op(x))

        if self.print_cost:
            print ' - L2 NORM:', l2_norm

        return l2_norm

    ##
    #  Method to calculate the l1 norm of the reconstruction.
    #
    #  @param[in] x: Input data.
    #
    def l1norm(self, x):

        x = self.weights * self.wavelet.op(x)

        l1_norm = np.sum(np.abs(x))

        if self.print_cost:
            print ' - L1 NORM:', l1_norm

        return l1_norm

    ##
    #  Method to calculate the nuclear norm of the reconstruction.
    #
    #  @param[in] x: Input data.
    #
    def nucnorm(self, x):

        if self.data_format == 'map':
            x_prime = map2matrix(x)

        else:
            x_prime = cube2matrix(x)

        nuc_norm = nuclear_norm(x_prime)

        if self.print_cost:
            print ' - NUCLEAR NORM:', nuc_norm

        return nuc_norm

    ##
    #  Method to check for convergence of the cost.
    #
    def check_cost(self, x):

        if self.iteration % (2 * self.window):

            self.x_list.append(x)
            self.test_list.append(self.cost)

            return False

        else:

            self.x_list.append(x)
            self.test_list.append(self.cost)
            x1 = np.average(self.x_list[:self.window], axis=0)
            x2 = np.average(self.x_list[self.window:], axis=0)
            t1 = np.average(self.test_list[:self.window], axis=0)
            t2 = np.average(self.test_list[self.window:], axis=0)
            self.x_list = []
            self.test_list = []

            test = (np.linalg.norm(t1 - t2) / np.linalg.norm(t1))

            if self.print_cost:
                print ' - CONVERGENCE TEST:', test
                print ''

            if self.liveplot:
                # livePlot(x2, x1, self.iteration)
                liveCost(self.cost_list, self.iteration, self.total_it)

            return test <= self.tolerance

    ##
    #  Method to check the residual of the reconstruction.
    #
    def check_residual(self, x):

        self.res = np.std(self.y - self.grad.op(x)) / np.linalg.norm(self.y)

        if self.print_cost:
            print ' - STD RESIDUAL:', self.res

    ##
    #  Method to calculate the cost of the reconstruction.
    #
    #  @param[in] x: Input data.
    #
    def get_cost(self, x):

        if self.print_cost:
            print ' - ITERATION:', self.iteration

        self.iteration += 1
        self.cost_old = self.cost

        self.check_residual(x)

        if self.positivity:
            print ' - MIN(X):', np.min(x)

        if self.mode == 'all':
            self.cost = (0.5 * self.l2norm(x) ** 2 + self.l1norm(x) +
                         self.nucnorm(x))

        elif self.mode == 'wave':
            self.cost = 0.5 * self.l2norm(x) ** 2 + self.l1norm(x)

        elif self.mode == 'lowr':
            self.cost = (0.5 * self.l2norm(x) ** 2 + self.lambda_reg *
                         self.nucnorm(x))

        elif self.mode == 'grad':
            self.cost = 0.5 * self.l2norm(x) ** 2

        self.cost_list.append(self.cost)

        if self.print_cost:
            print ' - Log10 COST:', np.log10(self.cost)
            print ''

        return self.check_cost(x)

    def plot_cost(self):

        plotCost(self.cost_list)
