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
from psf.transform import *
from psf.noise import *


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

        self.x_old = np.random.random(self.data_shape)

    ##
    #  Method that initialises the class instance.
    #
    #  @param[in] tolerance: Tolerance for convergence.
    #  @param[in] max_iter: Maximum number of iterations.
    #
    def get_spec_rad(self, tolerance=1e-6, max_iter=150):

        # Set (or reset) values of x.
        self.set_initial_x()

        # Iterate until the L2 norm of x converges.
        for i in xrange(max_iter):

            self.x_new = self.op(self.x_old) / norm(self.x_old)

            if(np.abs(norm(self.x_new) - norm(self.x_old)) < tolerance):
                print (' - Power Method converged after %d iterations!' %
                       (i + 1))
                break

            elif i == max_iter - 1:
                print (' - Power Method did not converge after %d '
                       'iterations!' % max_iter)

            np.copyto(self.x_old, self.x_new)

        self.spec_rad = norm(self.x_new)
        self.inv_spec_rad = 1.0 / self.spec_rad


##
#  Class for speeding up algorithm convergence.
#
class SpeedUp():

    ##
    #  Method that initialises the class instance.
    #
    def __init__(self):

        self.t_now = 1.0
        self.t_prev = 1.0
        self.use_speed_up = True

    ##
    #  Method turns off the speed up.
    #
    def speed_switch(self, turn_on=True):

        self.use_speed_up = turn_on

    ##
    #  Method that updates the lambda value.
    #
    def update_lambda(self):

        self.t_prev = self.t_now
        self.t_now = (1 + np.sqrt(4 * self.t_prev ** 2 + 1)) * 0.5
        self.lambda_now = 1 + (self.t_prev - 1) / self.t_now

    ##
    #  Method that returns the update to the input data.
    #
    def speed_up(self, data_now, data_prev):

        if self.use_speed_up:
            self.update_lambda()
            return data_prev + self.lambda_now * (data_now - data_prev)

        else:
            return data_now


##
#  Class for defining the operators of a pixel variant PSF.
#
class LowRankMatrix(SpeedUp):

    ##
    #  Method that initialises the class instance.
    #
    #  @param[in] data_shape: 2D shape of input data.
    #
    def __init__(self, data_shape, layout, tolerance=1e-5):

        SpeedUp.__init__(self)
        self.data_shape = np.array(data_shape)
        self.layout = np.array(layout)
        self.data_rec = np.ones(self.data_shape)
        self.data_rec_prev = np.copy(self.data_rec)
        self.thresh = 0.0
        self.thresh_prev = 0.0
        self.update_thresh = True
        self.tolerance = tolerance

    def update_threshold(self, grad, factor=3):

        u, s, v = np.linalg.svd(map2matrix(grad, self.layout))

        self.thresh = factor * np.median(s)

    def check_threshold(self, iter_num, grad, factor=3, max_iter=50):

        if self.update_thresh:

            self.update_threshold(grad, factor)

            if np.abs(self.thresh - self.thresh_prev) < self.tolerance:
                self.update_thresh = False
                print ' - Threshold converged!'

            elif iter_num == max_iter:
                self.update_thresh = False
                print ' - Threshold stabalised after 50 iterations.'

            self.thresh_prev = self.thresh

    def update(self, grad, inv_spec_rad):

        data_rec_temp = self.data_rec - inv_spec_rad * grad

        data_matrix = map2matrix(data_rec_temp, self.layout)
        data_matrix = svd_threshold(data_matrix, self.thresh,
                                    threshold_type='soft')

        data_rec_temp = matrix2map(data_matrix, self.data_shape)

        np.copyto(self.data_rec_prev, data_rec_temp)

        self.data_rec = self.speed_up(data_rec_temp, self.data_rec_prev)
