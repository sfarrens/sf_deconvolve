#  @file fits.py
#
#  FITTING FUNCTIONS
#
#  Functions for finding best
#  fit to data.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from scipy.odr import *


##
#  Equation of a straight line: y = mx + b
#
#  @param[in] B: Slope (m) and intercept (b).
#  @param[in] x: Data.
#
#  @return Value of y.
#
def linear_fit(B, x):

    return B[0] * x + B[1]


##
#  Equation of a polynomial line: y = a_0 + a_1x + a_2x^2 + ... + a_kx^k
#
#  @param[in] x: Independent data vector.
#  @param[in] a: Polynomial coefficient vector.
#
#  @return Vector of dependent variables y.
#
def polynomial(x, a):

    return sum([(a_i * x ** n) for a_i, n in zip(a, range(a.size))])


##
#  Find coefficients for a polynomial line fit to the input data.
#
#  @param[in] x: Independent data vector.
#  @param[in] y: Dependent data vector.
#  @param[in] k: Number of degrees of freedom. (Default k=1)
#
#  @return Vector of coefficients a.
#
def polynomial_fit(x, y, k=1):

    return least_squares(x_matrix(x, k), y)


##
#  Analytical least squares regression. Returns the values of the coefficients,
#  a, given the input matrix X and the corresponding y values.
#
#  @param[in] X: Independent data matrix.
#  @param[in] y: Dependent data vector.
#
#  @return Vector of coefficients a.
#
def least_squares(X, y):

    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)


##
# Function to define the matrix X for a given vector x corresponding to a
# polynomial with k degrees of freedom.
#
#  @param[in] x: Independent data vector.
#  @param[in] k: Number of degrees of freedom.
#
#  @return Independent variable matrix X.
#
def x_matrix(x, k):

    return np.vstack([x ** n for n in range(k + 1)]).T


##
#  Orthogonal distance regression fit.
#
#  @param[in] x: x data.
#  @param[in] y: y data.
#  @param[in] xerr: x data errors.
#  @param[in] yerr: y data errors.
#  @param[in] fit: Function for fit.
#
#  @return Best fit parameters.
#
def fit_odr(x, y, xerr, yerr, fit):

    model = Model(fit)
    r_data = RealData(x, y, sx=xerr, sy=yerr)
    odr = ODR(r_data, model, beta0=[1.0, 2.0])
    odr_out = odr.run()

    return odr_out.beta
