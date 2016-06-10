#  @file signal.py
#
#  SIGNAL PROCESSING FUNCTIONS
#
#  Basic functions for signal processing.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np


##
#  Function that tests two operators to see if they are the transpose of each
#  other.
#
#  @param[in] operator: Operator function.
#  @param[in] operator_t: Transpose operator function.
#  @param[in] data_shape: 2D Data shape.
#
def transpose_test(operator, operator_t, x_shape, x_args, y_shape=None,
                   y_args=None):

    if isinstance(y_shape, type(None)):
        y_shape = x_shape

    if isinstance(y_args, type(None)):
        y_args = x_args

    # Generate random arrays.
    x = np.random.ranf(x_shape)
    y = np.random.ranf(y_shape)

    # Calculate <MX, Y>
    mx_y = np.sum(np.multiply(operator(x, *x_args), y))

    # Calculate <X, M.TY>
    x_mty = np.sum(np.multiply(x, operator_t(y, *y_args)))

    # Test the difference between the two.
    print ' - |<MX, Y> - <X, M.TY>| =', np.abs(mx_y - x_mty)
