#  @file np_adjust.py
#
#  NUMPY ADJUSTMENT FUNCTIONS
#
#  Functions for making some
#  minor adjustments to the
#  standard output of Numpy
#  functions.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np


##
#  Function rotates the input data by 180 degrees.
#
#  @param[in] data: Input data.
#
#  @return rotated data.
#
def rotate(data):

    return np.rot90(data, 2)


##
#  Function rotates each of the elements of an input data stack by 180 degrees.
#
#  @param[in] data: Input data stack.
#
#  @return rotated data stack.
#
def rotate_stack(data):

    return np.array([rotate(x) for x in data])


##
#  Function converts input data into a numpy array.
#
#  @param[in] data: Input data of any type.
#  @param[in] dtype: Output data type.
#
#  @return Numpy array of data.
#
def data2np(data, dtype=None):

    if not isinstance(data, np.ndarray):
        if isinstance(data, (int, float, str)):
            data = np.array([data], dtype=dtype)
        else:
            data = np.array(data, dtype=dtype)

    return data


##
#  Function pads a 2D array with zeros.
#
#  @param[in] data: 2D Input data.
#  @param[in] shape: 2D Padding shape.
#
#  @return Padded array.
#
def pad2d(data, shape):

    data = data2np(data)
    shape = data2np(shape)

    if shape.ndim == 1:
        shape = np.repeat(shape, 2)

    return np.pad(data, ((shape[0], shape[0]), (shape[1], shape[1])),
                  'constant')


##
#  Function corrects the x-range output
#  from np.histogram for plotting.
#
#  @param[in] vals: x-range from np.histogram
#
#  @return Corrected x-range.
#
def x_bins(vals):

    return (vals[:-1] + vals[1:]) / 2.0


##
#  Function corrects the x-range output from np.histogram for plotting step.
#
#  @param[in] vals: x-range from np.histogram
#
#  @return Corrected x-range step.
#
def x_bins_step(vals):

    return x_bins(vals) + (vals[1] - vals[0]) / 2.0
