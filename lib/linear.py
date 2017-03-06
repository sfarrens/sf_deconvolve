# -*- coding: utf-8 -*-

"""LINEAR OPERATORS

This module contains linear operator classes.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.1

:Date: 04/01/2017

"""

import numpy as np
from wavelet import *
from functions.matrix import rotate
from functions.signal import *


# ##
# #  Directional operator class.
# #
# class Directional():
#
#     ##
#     #  Class initializer.
#     #
#     #  @param[in] data: Input data.
#     #  @param[in] angle_num: Number of angles.
#     #  @param[in] scale_num: Number of scales.
#     #
#     #  @exception ValueError for invalid data format.
#     #
#     def __init__(self, data, angle_num, scale_num):
#
#         self.y = data
#
#         self.data_shape = data.shape[-2:]
#         n = data.shape[0]
#
#         self.get_filters(angle_num, scale_num)
#
#     def get_filters(self, angle_num, scale_num):
#
#         sigma = scale_num
#
#         angles = np.arange(angle_num) * np.pi / angle_num
#
#         shift = (self.y.shape[0] - 1) / 2
#
#         index_matrix = np.tile(np.arange(self.y.shape[0]),
#                                (self.y.shape[0], 1)).T - shift
#
#         def func(x, y):
#
#          return (Gaussian_filter(rotate(index_matrix, x), y, fourier=True) *
#                     mex_hat(rotate(index_matrix, x + np.pi / 2), y))
#
#         self.filters = np.array([func(angle, sigma) for angle in angles])
#
#     # def filter_convolve(self, data):
#
#     # ##
#     # #  Class operator.
#     # #
#     # #  @param[in] data: Input data.
#     # #
#     # #  @return Wavelet convolved data.
#     # #
#     # def op(self, data):
#     #
#     #         return filter_convolve_stack(data, self.filters)
#     #
#     # ##
#     # #  Class adjoint operator.
#     # #
#     # #  @param[in] data: Input data.
#     # #
#     # #  @return Wavelet convolved data.
#     # #
#     # def adj_op(self, data):
#     #
#     #       return filter_convolve_stack(data, self.filters, filter_rot=True)
#     #
#     # ##
#     # #  Method to calculate gradient.
#     # #
#     # #  @param[in] data: Input data.
#     # #
#     # #  Calculates: Phi (Phi.T X - Y)
#     # #
#     # #  @return Gradient step.
#     # #
#     # def get_grad(self, data):
#     #
#     #     self.grad = self.op(self.adj_op(data) - self.y)
#     #     self.inv_spec_rad = 1.0


class Identity(object):
    """Identity operator class

    This is a dummy class that can be used in the optimisation classes

    """

    def __init__(self):

        self.l1norm = 1.0

    def op(self, data, **kwargs):
        """Operator

        Returns the input data unchanged

        Parameters
        ----------
        data : np.ndarray
            Input data array
        **kwargs
            Arbitrary keyword arguments

        Returns
        -------
        np.ndarray input data

        """

        return data

    def adj_op(self, data):
        """Adjoint operator

        Returns the input data unchanged

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray input data

        """

        return data


class Wavelet(object):
    """Wavelet class

    This class defines the wavelet transform operators

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally an array of 2D images
    wavelet_opt: str, optional
        Additional options for `mr_transform`

    """

    def __init__(self, data, wavelet_opt=None):

        self.y = data
        self.data_shape = data.shape[-2:]
        n = data.shape[0]

        self.filters = get_mr_filters(self.data_shape, opt=wavelet_opt)
        self.l1norm = n * np.sqrt(sum((np.sum(np.abs(filter)) ** 2 for
                                       filter in self.filters)))

    def op(self, data):
        """Operator

        This method returns the input data convolved with the wavelet filters

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 2D image

        Returns
        -------
        np.ndarray wavelet convolved data

        """

        return filter_convolve_stack(data, self.filters)

    def adj_op(self, data):
        """Adjoint operator

        This method returns the input data convolved with the wavelet filters
        rotated by 180 degrees

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 3D of wavelet coefficients

        Returns
        -------
        np.ndarray wavelet convolved data

        """

        return filter_convolve_stack(data, self.filters, filter_rot=True)

    ##
    # #  Method to calculate gradient.
    # #
    # #  @param[in] data: Input data.
    # #
    # #  Calculates: Phi (Phi.T X - Y)
    # #
    # #  @return Gradient step.
    # #
    # def get_grad(self, data):
    #
    #     self.grad = self.op(self.adj_op(data) - self.y)
    #     self.inv_spec_rad = 1.0


class LinearCombo(object):
    """Linear combination class

    This class defines a combination of linear transform operators

    Parameters
    ----------
    operators : list
        List of linear operator class instances
    weights : list, optional
        List of weights for combining the linear adjoint operator results

    """

    def __init__(self, operators, weights=None):

        self.operators = operators
        self.weights = weights
        self.l1norm = np.array([operator.l1norm for operator in
                                self.operators])

    def op(self, data):
        """Operator

        This method returns the input data operated on by all of the operators

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 2D image

        Returns
        -------
        np.ndarray linear operation results

        """

        res = np.empty(len(self.operators), dtype=np.ndarray)

        for i in xrange(len(self.operators)):
            res[i] = self.operators[i].op(data)

        return res

    def adj_op(self, data):
        """Adjoint operator

        This method returns the combination of the result of all of the
        adjoint operators. If weights are provided the comibination is the sum
        of the weighted results, otherwise the combination is the mean.

        Parameters
        ----------
        data : np.ndarray
            Input data array, an array of coefficients

        Returns
        -------
        np.ndarray adjoint operation results

        """

        if isinstance(self.weights, type(None)):

            return np.mean([operator.adj_op(x) for x, operator in
                           zip(data, self.operators)], axis=0)

        else:

            return np.sum([weight * operator.adj_op(x) for x, operator,
                          weight in zip(data, self.operators, weights)],
                          axis=0)
