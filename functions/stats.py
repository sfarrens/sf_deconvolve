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
from astropy.convolution import Gaussian2DKernel


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


def gaussian(point, mean, sigma, amplitude=None):
    """Gaussian distribution

    Method under development...

    """

    if isinstance(amplitude, type(None)):
        # amplitude = 1 / (sigma * np.sqrt(2 * np.pi))
        amplitude = 1

    val = np.array([((x - mu) / sig) ** 2 for x, mu, sig in
                   zip(point, mean, sigma)])

    return amplitude * np.exp(-0.5 * val)


def gaussian_kernel(data_shape, sigma, norm='max'):
    """Gaussian kernel

    This method produces a Gaussian kerenal of a specified size and dispersion

    Parameters
    ----------
    data_shape : tuple
        Desiered shape of the kernel
    sigma : float
        Standard deviation of the kernel
    norm : str {'max', 'sum'}, optional
        Normalisation of the kerenl (options are 'max' or 'sum')

    Returns
    -------
    np.ndarray kernel

    """

    kernel = np.array(Gaussian2DKernel(sigma, x_size=data_shape[1],
                      y_size=data_shape[0]))

    if norm is 'max':
        return kernel / np.max(kernel)

    elif norm is 'sum':
        return kernel / np.sum(kernel)

    else:
        return kernel


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
#  Function that returns the Mean Squared Error (MSE) between two data sets.
#
#  @param[in] x: First data set.
#  @param[in] y: Second data set.
#
#  @return MSE value.
#
def mse(x, y):

    return np.mean((x - y) ** 2)


##
#  Function that returns the Peak Signa-to-Noise Ratio (PSNR) between and image
#  and a noisy version of that image.
#
#  @param[in] image: Input image.
#  @param[in] noisy_image: Noisy version of input image.
#  @param[in] max_pix: Maximum pixel value.
#
#  @return PSNR value.
#
#  https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
#  PSNR = 20 * log_10(MAX_I) - 10 * log_10(MSE)
#
def psnr2(image, noisy_image, max_pix=255):

    return (20 * np.log10(max_pix) - 10 *
            np.log10(mse(image, noisy_image)))


def psnr(image, recovered_image):
    """Peak Signal-to-Noise Ratio

    This method calculates the PSNR between an image and the recovered version
    of that image

    Parameters
    ----------
    image : np.ndarray
        Input image, 2D array
    recovered_image : np.ndarray
        Recovered image, 2D array

    Returns
    -------
    float PSNR value

    Notes
    -----
    Implements eq.3.7 from _[S2010]

    """

    return (20 * np.log10((image.shape[0] * np.abs(np.max(image) -
            np.min(image))) / np.linalg.norm(image - recovered_image)))


def psnr_stack(images, recoverd_images, metric=np.mean):
    """Peak Signa-to-Noise for stack of images

    This method calculates the PSNRs for a stack of images and the
    corresponding recovered images. By default the metod returns the mean
    value of the PSNRs, but any other metric can be used.

    Parameters
    ----------
    images : np.ndarray
        Stack of images, 3D array
    recovered_images : np.ndarray
        Stack of recovered images, 3D array
    metric : function
        The desired metric to be applied to the PSNR values (default is
        'np.mean')

    Returns
    -------
    float metric result of PSNR values

    Raises
    ------
    ValueError
        For invalid input data dimensions

    """

    if images.ndim != 3 or recoverd_images.ndim != 3:
        raise ValueError('Input data must be a 3D np.ndarray')

    return metric([psnr(i, j) for i, j in zip(images, recoverd_images)])


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
