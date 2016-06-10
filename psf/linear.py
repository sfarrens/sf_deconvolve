#  @file linear.py
#
#  LINEAR OPERATORS
#
#  Classes of linear operators for optimization.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from wavelet import *


##
#  Wavelet operator class.
#
class Identity():

    ##
    #  Class initializer.
    #
    def __init__(self):

        self.l1norm = 1.0

    ##
    #  Class operator.
    #
    #  @param[in] data: Input data.
    #
    #  @return Input data.
    #
    def op(self, data, **kwargs):

        return data

    ##
    #  Class adjoint operator.
    #
    #  @param[in] data: Input data.
    #
    #  @return Input data.
    #
    def adj_op(self, data):

        return data


##
#  Wavelet operator class.
#
class Wavelet():

    ##
    #  Class initializer.
    #
    #  @param[in] data: Input data.
    #  @param[in] wavelet_levels: Number of wavelet levels.
    #  @param[in] wavelet_opt: Wavelet type option.
    #  @param[in] data_format: Input data format. (map or cube)
    #
    #  @exception ValueError for invalid data format.
    #
    def __init__(self, data, wavelet_levels, wavelet_opt=None,
                 data_format='map'):

        self.y = data
        self.data_format = data_format

        if self.data_format == 'map':
            self.data_shape = data.shape
            n = 1
        elif self.data_format == 'cube':
            self.data_shape = data.shape[-2:]
            n = data.shape[0]
        else:
            raise ValueError('Invalid data type. Options are "map" or "cube".')

        self.filters = get_mr_filters(self.data_shape, wavelet_levels,
                                      wavelet_opt)
        self.l1norm = n * np.sqrt(sum([np.sum(np.abs(filter)) ** 2 for
                                       filter in self.filters]))

    ##
    #  Class operator.
    #
    #  @param[in] data: Input data.
    #
    #  @return Wavelet convolved data.
    #
    def op(self, data):

        if self.data_format == 'map':
            return filter_convolve(data, self.filters)

        elif self.data_format == 'cube':
            return filter_convolve_stack(data, self.filters)

    ##
    #  Class adjoint operator.
    #
    #  @param[in] data: Input data.
    #
    #  @return Wavelet convolved data.
    #
    def adj_op(self, data):

        if self.data_format == 'map':
            return filter_convolve(data, self.filters_rot, filter_rot=True)

        elif self.data_format == 'cube':
            return filter_convolve_stack(data, self.filters, filter_rot=True)

    ##
    #  Method to calculate gradient.
    #
    #  @param[in] data: Input data.
    #
    #  Calculates: Phi (Phi.T X - Y)
    #
    #  @return Gradient step.
    #
    def get_grad(self, data):

        self.grad = self.op(self.adj_op(data) - self.y)
        self.inv_spec_rad = 1.0


##
#  Combined linear operator class.
#
class LinearCombo():

    ##
    #  Class initializer.
    #
    #  @param[in] operators: List of initialised linear operator classes.
    #
    def __init__(self, operators):

        self.operators = operators
        self.l1norm = np.array([operator.l1norm for operator in
                                self.operators])

    ##
    #  Class operator.
    #
    #  @param[in] data: Input data.
    #
    #  @return List of operator outputs.
    #
    def op(self, data):

        res = np.empty(len(self.operators), dtype=np.ndarray)

        for i in range(len(self.operators)):
            res[i] = self.operators[i].op(data)

        return res

        # return np.array([operator.op(data) for operator in self.operators])

    ##
    #  Class adjoint operator.
    #
    #  @param[in] data: Input data.
    #
    #  @return List of adjoint operator outputs.
    #
    def adj_op(self, data):

        return sum([operator.adj_op(x) for x, operator in
                    zip(data, self.operators)])
