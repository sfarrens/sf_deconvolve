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
#  2) Bauschke et al., Fixed-Point Algorithms for Inverse Problems in Science
#  and Engineering, 2011, Chapter 10. (B2010)
#  3) Raguet et al., Generalized Forward-Backward Splitting, 2012, , (R2012)
#
#  NOTES:
#  * x_old is used in place of x_{n}.
#  * x_new is used in place of x_{n+1}.
#  * x_prox is used in place of \~{x}_{n+1}.
#  * x_temp is used for intermediate operations.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np


##
#  FISTA optimisation class. Used to speed-up convergence.
#
class FISTA():

    ##
    #  Method that initialises the class instance.
    #
    #  @param[in] lambda_init: Initial value of the relaxation parameter.
    #  @param[in] active: Option to activate FISTA convergence speed-up.
    #
    def __init__(self, lambda_init=None, active=True):

        self.lambda_now = lambda_init
        self.t_now = 1.0
        self.t_prev = 1.0
        self.use_speed_up = active

    ##
    #  Method turns off the speed up.
    #
    #  @param turn_on: Option to turn on or off speed-up.
    #
    def speed_switch(self, turn_on=True):

        self.use_speed_up = turn_on

    ##
    #  Method that updates the lambda value.
    #
    #  Implements steps 3 and 4 from algoritm 10.7 in B2010.
    #
    def update_lambda(self):

        self.t_prev = self.t_now
        self.t_now = (1 + np.sqrt(4 * self.t_prev ** 2 + 1)) * 0.5
        self.lambda_now = 1 + (self.t_prev - 1) / self.t_now

    ##
    #  Method that returns the update if the speed up is active.
    #
    def speed_up(self):

        if self.use_speed_up:
            self.update_lambda()


##
#  Forward-backward optimisation class.
#
class ForwardBackward(FISTA):

    ##
    #  Class initialiser.
    #
    #  @param[in] x: Initial guess for the primal variable.
    #  @param[in] grad: Gradient operator class.
    #  @param[in] prox: Proximity operator class.
    #  @param[in] cost: Cost function class.
    #  @param[in] lambda_init: Initial value of the relaxation parameter.
    #  @param[in] lambda_update: Relaxation parameter update method.
    #  @param[in] use_fista: Option to use FISTA speed-up.
    #  @param[in] auto_iterate: Iterate after initialization.
    #
    def __init__(self, x, grad, prox, cost=None, lambda_init=None,
                 lambda_update=None, use_fista=True, auto_iterate=True,
                 indent_level=1):

        FISTA.__init__(self, lambda_init, use_fista)
        self.x_old = x
        self.z_old = np.copy(self.x_old)
        self.grad = grad
        self.prox = prox
        self.cost_func = cost
        self.lambda_update = lambda_update
        self.converge = False
        self.indent = ' ' * indent_level
        if auto_iterate:
            self.iterate()

    ##
    #  Method to update the current reconstruction.
    #
    #  Implements algorithm 10.7 (or 10.5) from B2010.
    #
    def update(self):

        # Step 1 from alg.10.7.
        self.grad.get_grad(self.z_old)
        y_old = self.z_old - self.grad.inv_spec_rad * self.grad.grad

        # Step 2 from alg.10.7.
        self.x_new = self.prox.op(y_old)

        # Steps 3 and 4 from alg.10.7.
        self.speed_up()

        # Step 5 from alg.10.7.
        self.z_new = self.x_old + self.lambda_now * (self.x_new - self.x_old)

        # Test primal variable for convergence.
        if np.sum(np.abs(self.z_old - self.z_new)) <= 1e-6:
            print self.indent + '- converged!'
            self.converge = True

        # Update old values for next iteration.
        np.copyto(self.x_old, self.x_new)
        np.copyto(self.z_old, self.z_new)

        # Update parameter values for next iteration.
        if not isinstance(self.lambda_update, type(None)):
            self.lambda_now = self.lambda_update(self.lambda_now)

        # Test cost function for convergence.
        if not isinstance(self.cost_func, type(None)):
            self.converge = self.cost_func.get_cost(self.z_new)

        if np.all(self.z_new == 0.0):
            raise RuntimeError(self.indent + '- The reconstruction is fucked!')

    ##
    #  Method to iteratively update the reconstruction.
    #
    def iterate(self, max_iter=150):

        for i in xrange(max_iter):
            self.update()

            if self.converge:
                print self.indent + '- Converged!'
                break

        self.x_final = self.z_new


##
#  Forward-backward optimisation class.
#
class GenForwardBackward():

    ##
    #  Class initialiser.
    #
    #  @param[in] x: Initial guess for the primal variable.
    #  @param[in] grad: Gradient operator class.
    #  @param[in] prox_list: List of proximity operator class.
    #  @param[in] cost: Cost function class.
    #  @param[in] lambda_init: Initial value of the relaxation parameter.
    #  @param[in] lambda_update: Relaxation parameter update method.
    #  @param[in] weights: Proximity operator weights.
    #  @param[in] auto_iterate: Iterate after initialization.
    #  @param[in] indent_level: Indentation level.
    #
    def __init__(self, x, grad, prox_list, cost=None, lambda_init=1.0,
                 lambda_update=None, weights=None, auto_iterate=True,
                 indent_level=1, plot=False):

        self.x_old = x
        self.grad = grad
        self.prox_list = np.array(prox_list)
        self.cost_func = cost
        self.lambda_init = lambda_init
        self.lambda_update = lambda_update

        if isinstance(weights, type(None)):
            self.weights = np.repeat(1.0 / self.prox_list.size,
                                     self.prox_list.size)
        else:
            self.weights = np.array(weights)

        # Check weights.
        if np.sum(self.weights) != 1.0:
            raise ValueError('Proximity operator weights must sum to 1.0.'
                             'Current sum of weights = ' +
                             str(np.sum(self.weights)))

        self.z = np.array([self.x_old for i in xrange(self.prox_list.size)])

        self.indent = ' ' * indent_level
        self.plot = plot
        self.converge = False
        if auto_iterate:
            self.iterate()

    ##
    #  Method to update the current reconstruction.
    #
    #  Implements algorithm 1 from R2012.
    #
    def update(self):

        # Calculate gradient for current iteration.
        self.grad.get_grad(self.x_old)

        # Update z values.
        for i in xrange(self.prox_list.size):
            z_temp = (2 * self.x_old - self.z[i] - self.grad.inv_spec_rad *
                      self.grad.grad)
            z_prox = self.prox_list[i].op(z_temp,
                                          extra_factor=self.grad.inv_spec_rad /
                                          self.weights[i])
            self.z[i] += self.lambda_init * (z_prox - self.x_old)

        # Update current reconstruction.
        self.x_new = np.sum((z_i * w_i for z_i, w_i in
                            zip(self.z, self.weights)), axis=0)

        # Update old values for next iteration.
        np.copyto(self.x_old, self.x_new)

        # Update parameter values for next iteration.
        if not isinstance(self.lambda_update, type(None)):
            self.lambda_now = self.lambda_update(self.lambda_now)

        # Test cost function for convergence.
        if not isinstance(self.cost_func, type(None)):
            self.converge = self.cost_func.get_cost(self.x_new)

        if np.all(self.x_new == 0.0):
            raise RuntimeError(self.indent + '- The reconstruction is fucked!')

    ##
    #  Method to iteratively update the reconstruction.
    #
    def iterate(self, max_iter=150):

        for i in xrange(max_iter):
            self.update()

            if self.converge:
                print self.indent + '- Converged!'
                break

        self.x_final = self.x_new
        self.cost_func.plot_cost()


##
#  Condat optimisation class.
#
class Condat():

    ##
    #  Class initialiser.
    #
    #  @param[in] x: Initial guess for the primal variable.
    #  @param[in] y: Initial guess for the dual variable.
    #  @param[in] grad: Gradient operator class.
    #  @param[in] prox: Proximity primal operator class.
    #  @param[in] prox_dual: Proximity dual operator class.
    #  @param[in] linear: Linear operator class.
    #  @param[in] cost: Cost function class.
    #  @param[in] rho: Relaxation parameter.
    #  @param[in] sigma: Proximal dual parameter.
    #  @param[in] tau: Proximal primal paramater.
    #  @param[in] rho_update: Relaxation parameter update method.
    #  @param[in] sigma_update: Proximal dual parameter update method.
    #  @param[in] tau_update: Proximal primal parameter update method.
    #  @param[in] auto_iterate: Iterate after initialization.
    #
    def __init__(self, x, y, grad, prox, prox_dual, linear, cost,
                 rho,  sigma, tau, rho_update=None, sigma_update=None,
                 tau_update=None, auto_iterate=True):

        self.x_old = x
        self.y_old = y
        self.grad = grad
        self.prox = prox
        self.prox_dual = prox_dual
        self.linear = linear
        self.cost_func = cost
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.rho_update = rho_update
        self.sigma_update = sigma_update
        self.tau_update = tau_update
        self.converge = False
        if auto_iterate:
            self.iterate()

    ##
    #  Method to update parameter values.
    #
    def update_param(self):

        # Update relaxation parameter.
        if not isinstance(self.rho_update, type(None)):
            self.rho = self.rho_update(self.rho)

        # Update proximal dual parameter.
        if not isinstance(self.sigma_update, type(None)):
            self.sigma = self.sigma_update(self.sigma)

        # Update proximal primal parameter.
        if not isinstance(self.tau_update, type(None)):
            self.tau = self.tau_update(self.tau)

    ##
    #  Method to update the current reconstruction.
    #
    #  Implements equation 9 (algorithm 3.1) from C2013.
    #
    def update(self):

        # Step 1 from eq.9.
        self.grad.get_grad(self.x_old)

        x_temp = (self.x_old - self.tau * self.grad.grad - self.tau *
                  self.linear.adj_op(self.y_old))
        x_prox = self.prox.op(x_temp)

        # Step 2 from eq.9.
        y_temp = (self.y_old + self.sigma *
                  self.linear.op(2 * x_prox - self.x_old))

        y_prox = (y_temp - self.sigma * self.prox_dual.op(y_temp / self.sigma,
                  extra_factor=(1.0 / self.sigma)))

        # Step 3 from eq.9.
        self.x_new = self.rho * x_prox + (1 - self.rho) * self.x_old
        self.y_new = self.rho * y_prox + (1 - self.rho) * self.y_old

        # self.x_new, self.y_new = (self.rho * np.array([x_prox, y_prox]) +
        #                           (1 - self.rho) *
        #                           np.array([self.x_old, self.y_old]))

        # Update old values for next iteration.
        np.copyto(self.x_old, self.x_new)
        np.copyto(self.y_old, self.y_new)

        # Update parameter values for next iteration.
        self.update_param()

        # Test cost function for convergence.
        self.converge = self.cost_func.get_cost(self.x_new)

    ##
    #  Method to iteratively update the reconstruction.
    #
    def iterate(self, max_iter=150):

        for i in xrange(max_iter):
            self.update()

            if self.converge:
                print ' - Converged!'
                break

        self.x_final = self.x_new
        self.y_final = self.y_new
        self.cost_func.plot_cost()
