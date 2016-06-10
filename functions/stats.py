#  @file astro.py
#
#  STATISTICS FUNCTIONS
#
#  Basic statistics functions.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
import scipy.stats as ss


##
#  Function that returns the l2 norm of the 2D autocorrelation of an input
#  array.
#
#  @param[in] data: 2D Input array.
#
#  @return Autocorrelation.
#
#  Returns sum(|FFT_2D(X)| ^ 4) ^ 0.5
#
def autocorr2d(data):

    x = np.abs(np.fft.fft2(data))

    return np.linalg.norm(x ** 2)


##
#  Function that tests the chi^2 goodness
#  of fit.
#
#  @param[in] data_obs: Observed data.
#  @param[in] data_exp: Expected data.
#  @param[in] sigma: Expected data error.
#  @param[in] ddof: Delta degrees of freedom.
#  Default (ddof = 1).
#
#  @return Chi-squared value and probability.
#
#  Degrees of freedom = len(data_obs) - ddof
#
def chi2_gof(data_obs, data_exp, sigma, ddof=1):

    chi2 = np.sum(((data_obs - data_exp) / sigma) ** 2)
    p_val = ss.chi2.cdf(chi2, len(data_obs) - ddof)

    return chi2, p_val


##
#  Function that returns the median absolute deviation (MAD) of an input array.
#
#  @param[in] data: Input data.
#
#  @return MAD.
#
#  MAD = median_i(|X_i - median_j(X_j)|)
#
def mad(data):

    return np.median(np.abs(data - np.median(data)))


##
#  Function that returns the standard deviation (sigma) using the median
#  absolute deviation (MAD).
#
#  @param[in] data: Input data.
#
#  @return Sigma.
#
#  Sigma = 1.4826 * MAD
#
def sigma_mad(data):

    return 1.4826 * mad(data)


##
#  Function that returns median signal-to-noise ratio (SNR) of a data stack for
#  a given value of the standard deviation .
#
#  @param[in] data: Input data stack.
#  @param[in] sigma: Standard deviation.
#
#  @return Median SNR.
#
#  SNR = med(||X_i||_inf) / sigma
#
def med_snr(data, sigma):

    return np.median([np.linalg.norm(x, np.inf) for x in data]) / sigma


##
#  Function that returns the standard deviation for a data stack corresponding
#  to a median signal-to-noise ratio.
#
#  @param[in] data: Input data.
#  @param[in] sigma: Standard deviation.
#
#  @return Median SNR.
#
#  sigma = med(||X_i||_inf) / SNR
#
def sigma_from_snr(data, snr):

    return np.median([np.linalg.norm(x, np.inf) for x in data]) / snr
