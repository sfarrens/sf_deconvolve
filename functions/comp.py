#  @file comp.py
#
#  COMPUTATIONAL FUNCTIONS
#
#  Basic indexing functions.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from functools import wraps
import warnings


##
#  Function that checks if input value is a float and if not converts it.
#
#  @param[in] val: Input value.
#
#  @return Float value.
#
def check_float(val):

    if type(val) is float:
        pass
    elif type(val) is int:
        val = float(val)
    elif type(val) is list or type(val) is tuple:
        val = np.array(val, dtype=float)
    elif type(val) is np.ndarray and val.dtype is not 'float64':
        val = val.astype(float)
    else:
        raise ValueError('Invalid input type.')

    return val


##
#  Function that checks if input value is an int and if not converts it.
#
#  @param[in] val: Input value.
#
#  @return Int value.
#
def check_int(val):

    if type(val) is int:
        pass
    elif type(val) is float:
        val = int(val)
    elif type(val) is list or type(val) is tuple:
        val = np.array(val, dtype=int)
    elif type(val) is np.ndarray and val.dtype is not 'int64':
        val = val.astype(int)
    else:
        raise ValueError('Invalid input type.')

    return val


##
#  Decorator that rounds the output of a function to 3 decimal places.
#
#  @param[in] func: Input funciton.
#
#  @return Decorated function.
#
def round3(func):

    '''
    Decorator that rounds the output of a function to 3 decimal places.

    Note: Usese numpy.round() and not native round().
    '''

    # Define function wrapper.
    @wraps(func)
    def wrapper(value):
        return np.round(value, 3)

    return wrapper


##
#  Function that checks if the minimum value is valid.
#
#  @param[in] min_val: Minimum of bin range.
#
#  @exception ValueError if min_val < 0.0.
#
def check_min(min_val):

    '''
    This function raises and exception if the minimum number provided is
    less than zero.
    '''

    if min_val < 0.0:
        raise ValueError('MIN_VAL must be >= 0.0.')


##
#  Function that checks if the minimum and
#  maximum values are valid.
#
#  @param[in] min_val: Minimum of bin range.
#  @param[in] max_val: Maximum of bin range.
#
#  @exception ValueError if min_val > max_val.
#
def check_minmax(min_val, max_val):

    check_min(min_val)

    if min_val > max_val:
        raise ValueError('MIN_VAL must be < MAX_VAL.')


##
#  Function that finds the bin corresponding
#  to a given value.
#
#  @param[in] value: Input value.
#  @param[in] min_value: Minimum of bin range.
#  @param[in] bin_size: Width of bins.
#
def find_bin(value, min_value, bin_size):

    check_min(min_value)

    return np.int(np.floor(np.round((np.array(value) - np.array(min_value)) /
                  np.array(bin_size), 8)))


##
#  Function that finds the number of bins
#  for a given range and bin size.
#
#  @param[in] min_value: Minimum of bin range.
#  @param[in] max_value: Maximum of bin range.
#  @param[in] bin_size: Width of bins.
#
def num_bins(min_value, max_value, bin_size):

    return np.int(np.floor(np.round((np.array(max_value) -
                  np.array(min_value)) / np.array(bin_size), 8)))


##
#  Function that the x-range values for
#  bins in a given range.
#
#  @param[in] n_bins: Number of bins.
#  @param[in] min_value: Minimum of bin range.
#  @param[in] bin_size: Width of bins.
#
def x_vals(n_bins, min_value, bin_size):

    return (np.arange(n_bins) + 0.5) * bin_size + min_value


##
#  Function that the checks if the input value
#  is within a given range.
#
#  @param[in] value: Input value.
#  @param[in] min_value: Minimum of bin range.
#  @param[in] max_value: Maximum of bin range.
#
def within(value, min_value, max_value):

    check_minmax(min_value, max_value)

    return ((np.array(value) >= np.array(min_value)) &
            (np.array(value) < np.array(max_value)))


##
#  Function that sets all NaN values in an
#  array to 1.
#
#  @param[in] array: Input array.
#
def nan2one(array):

    new_array = np.copy(array)

    new_array[np.isnan(new_array)] = 1.0

    return new_array


##
#  Function that sets all NaN values in an
#  array to 0.
#
#  @param[in] array: Input array.
#
def nan2zero(array):

    new_array = np.copy(array)

    new_array[np.isnan(new_array)] = 0.0

    return new_array


##
#  Feature scale data. Ignores division by
#  zero.
#
#  @param[in] data: Input data.
#  @param[in] min_val: Minimum value.
#  @param[in] max_val: Maximum value.
#
#  @exception ValueError if data > max_val.
#
def scale(data, min_val, max_val):

    warnings.simplefilter('ignore')

    data = np.array(data)

    if np.any(data > max_val):
        raise ValueError('DATA must be <= MAX_VAL.')

    check_minmax(min_val, max_val)

    scaled = np.float64(data - min_val) / np.float64(max_val - min_val)

    if isinstance(scaled, float):
        scaled = np.array([scaled])

    return nan2zero(scaled)
