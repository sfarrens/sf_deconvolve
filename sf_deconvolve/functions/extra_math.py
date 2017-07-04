# @file extra_math.py
#
#  EXTRA MATH FUNCTIONS
#
#  Some useful functions for
#  mathematical calculations.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from scipy.integrate import quad


##
#  Function that finds factors of a number (n).
#
#  @param[in] n: Number.
#
#  @return List of factors.
#
def factor(n):

    factors = set()

    for x in range(1, int(np.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(x)
            factors.add(n//x)

    return np.array(sorted(factors))


##
#  Function that finds the middle factor(s) of a number (n).
#
#  @param[in] n: Number.
#
#  @return Middle factor(s).
#
def mfactor(n):

    f = factor(n)

    if f.size % 2:
        return np.repeat(f[f.size / 2], 2)

    else:
        return f[f.size / 2 - 1:f.size / 2 + 1]


##
#  Function that integrates a given
#  function, which has 2 additional
#  arguments, between the specified
#  limits.
#
#  @param[in] func: Function to be integrated.
#  @param[in] lim_low: Lower limit of integration.
#  @param[in] lim_up: Upper limit of integration.
#  @param[in] arg1: 1st additional argument.
#  @param[in] arg2: 2nd additional argument.
#
#  @return Result of the definite integral.
#
def integ_2arg(func, lim_low, lim_up, arg1, arg2):

    return quad(func, lim_low, lim_up, args=(arg1, arg2))[0]


##
#  Vectorized version of integ_2arg. Integral limits
#  can be arrays.
#
#  @param[in] func: Function to be integrated.
#  @param[in] lim_low: Lower limit of integration.
#  @param[in] lim_up: Upper limit of integration.
#  @param[in] arg1: 1st additional argument.
#  @param[in] arg2: 2nd additional argument.
#
#  @return Array of the results of the definite integrals.
#
def vinteg_2arg(func, lim_low, lim_up, arg1, arg2):

    v_integ = np.vectorize(integ_2arg)

    return v_integ(func, lim_low, lim_up, arg1, arg2)


##
#  Function that returns k-values in the
#  range L.
#
#  @param[in] n: Number.
#  @param[in] L: L limit.
#
#  @return k-value.
#
def k_val(n, L):

    return (2.0 * np.pi / L) * np.array(range(n / 2.0) + range(-n / 2.0, 0.0))


##
#  Function that returns the derivative of the
#  specified function to the given order.
#
#  @param[in] func: Function.
#  @param[in] k: k-value
#  @param[in] order: Oder of derivative.
#
#  @return Derivative.
#
def fourier_derivative(func, k, order):

    return np.real(np.fft.ifft((1.j * k) ** order * np.fft.fft(func)))
