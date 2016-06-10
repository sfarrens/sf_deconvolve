#  @file optimisation.py
#
#  OPTIMISATION CLASSES
#
#  Classes of optimisation methods.
#
#  REFERENCES:
#  1) Condat, A Primal-Dual Splitting Method for Convex Optimization Involving
#  Lipschitzian, Proximable and Linear Composite Terms, 2013, Journal of
#  Optimization Theory and Applications, 158, 2, 460. (C2013)
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np

##
#  Condat optimisation class.
#
class Condat():

    ##
    #  Class initializer.
    #
    #  @param[in] data_shape: 2D input data shape.
    #  @param[in] grad: Gradient operator.
    #  @param[in] prox: Proximity operator.
    #  @param[in] prox_dual: Proximity dual operator.
    #  @param[in] linear: Linear operator.
    #  @param[in] cost: Cost function.
    #  @param[in] rho: Relaxation parameter.
    #  @param[in] sigma: Proximal dual parameter.
    #  @param[in] tau: Proximal paramater.
    #  @param[in] dual_shape: Shape of the data for dual operator.
    #  @param[in] print_cost: Print cost function at each iteration.
    #  @param[in] auto_iterate: Iterate after initialization.
    #
    def __init__(self, data_shape, grad, prox, prox_dual, linear, cost,
                 rho=0.5, sigma=None, tau=None, dual_shape=None,
                 print_cost=True, auto_iterate=True):

        self.grad = grad
        self.prox = prox
        self.prox_dual = prox_dual
        self.linear = linear
        self.cost = cost
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.x_old = np.ones(data_shape)
        if isinstance(dual_shape, type(None)):
            self.y_old = np.ones(data)
        else:
            self.y_old = np.ones(dual_shape)
        self.print_cost = print_cost
        if auto_iterate:
            self.iterate()

    ##
    #  Method to update the current reconstruction.
    #
    #  Implements equation 9 (algorithm 3.1) from C2013.
    #
    def update(self):

        x_grad = self.grad.get_grad(self.x_old)
        x_temp = (self.x_old - self.tau * x_grad - self.tau *
                  self.linear.adj_op(self.y_old))
        x_prox = self.prox.op(x_temp)

        y_temp = (self.y_old + self.sigma *
                  self.linear.op(2 * x_temp - self.x_old))
        y_prox = (y_temp - self.sigma * self.prox_dual.op(y_temp / self.sigma))

        self.x_new, self.y_new = (self.rho * np.array([x_prox, y_prox]) +
                                  (1 - self.rho) *
                                  np.array([self.x_old, self.y_old]))

        np.copyto(self.x_old, self.x_new)
        np.copyto(self.y_old, self.y_new)

        if self.print_cost:
            print 'COST:', self.cost.get_cost(self.x_new)

    ##
    #  Method to iteratively update the reconstruction.
    #
    def iterate(self, max_iter=150):

        [self.update() for i in range(max_iter)]
