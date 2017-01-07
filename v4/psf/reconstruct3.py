#  @file reconstruct3.py
#
#  RECONSTRUCTION ROUTINES
#
#  Functions for reconstructing an image from the observation.
#  Based on work by Fred Ngole and Yinghao Ge.
#
#  REFERENCES:
#  1) Condat, A Primal-Dual Splitting Method for Convex Optimization Involving
#  Lipschitzian, Proximable and Linear Composite Terms, 2013, Journal of
#  Optimization Theory and Applications, 158, 2, 460. (C2013)
#  2) Candes, Wakin and Boyd, Enhancing Sparsity by Reweighting l1
#  Minimization, 2008, Journal of Fourier Analysis and Applications,
#  14(5):877-905. (CWB2008)
#
#  NOTES:
#  Minimization problem is min_x 1/2 ||y - fx||_2 + ||lambda * w * phi^* x||_1.
#  Reweighting implemented as w = w (1 / (1 + |x^w|/(n * sigma))).
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from scipy.linalg import norm
from reconstruct import *
from wavelet import *
from noise import denoise
from convolve import pca_convolve
from transform import *
from functions.stats import sigma_mad
from functions.image import *
from functions.signal import transpose_test


def get_sigma_map(data, opt=None):

    # Get wavelet residuals.
    wavelet_residuals = call_mr_transform(data, opt=opt, remove_files=True)

    # Return sigma map.
    return np.array([sigma_mad(a) * np.ones(data.shape) for a in
                     wavelet_residuals])


def weighted_threshold(data, sigma, weights):

    return denoise(data, sigma * weights, 'soft')


def get_noise_est(psf_pcs, psf_coef, filters):

    print ' - Calculating weights.'

    kernel_shape = np.array(psf_pcs[0].shape)
    coef_shape = np.array(psf_coef[0].shape)

    vec_length = np.prod(kernel_shape)

    pcst = [rot_and_roll(x.T) for x in psf_pcs]
    pcs = [rot_and_roll(x) for x in psf_pcs]
    mask = gen_mask(kernel_shape, coef_shape)

    k_pat = kernel_pattern(kernel_shape, mask)

    k_rolls = roll_sequence(kernel_shape)
    m_rolls = roll_sequence(mask.shape)

    res = []
    i = 0
    val = 0

    for i in range(np.prod(coef_shape)):
        for k1 in range(psf_pcs.shape[0]):
            for k2 in range(psf_pcs.shape[0]):
                pkt = roll_2d(pcst[k1],
                              roll_rad=k_rolls[k_pat[i]]).reshape(vec_length)
                pk = roll_2d(pcs[k2],
                             roll_rad=k_rolls[k_pat[i]]).reshape(vec_length)
                mi = roll_2d(mask, roll_rad=m_rolls[i])
                dk1 = psf_coef[k1][mi]
                dk2 = psf_coef[k2][mi]
                val += sum(pkt * dk1 * dk2 * pk)
        res.append(val)

    res = 0.05 ** 2 * np.array(res).reshape(coef_shape)

    res = np.sqrt(convolve_mr_filters(res, filters ** 2))

    return res


##
#  Function that implements the method described in C2013 to reconstruct an
#  image that has been convolved with a pixel variant PSF.
#
#  @param[in] image: Input noisy image.
#  @param[in] psf_pcs: PSF principal components.
#  @param[in] psf_coef: PSF coefficients.
#  @param[in] image_rec: Image reconstruction.
#  @param[in] layout: 2D layout of the data map.
#  @param[in] inv_spec_rad: Inverse spectral radius.
#  @param[in] filters: Wavelet filters.
#  @param[in] wavelet_levels: Number of wavelet levels to use.
#  @param[in] wavelet_opt: Options to pass to mr_transform.
#  @param[in] weights: Regularization weights.
#  @param[in] thresh_factor: Threshold factor.
#  @param[in] n_reweights: Number of reweightings to perform.
#  @param[in] tolerance: Tolerance for convergence.
#  @param[in] max_iter: Maximum number of iterations.
#
#  Implements equation 9 (algorithm 3.1) from C2013 and reweighting scheme from
#  section 4 in CWB2008.
#
#  @return Image reconstruction.
#
def condat(image, psf_pcs, psf_coef, image_rec=None, layout=None,
           inv_spec_rad=None, filters=None, wavelet_levels=None,
           wavelet_opt=None, weights=None, thresh_factor=3, n_reweights=0,
           tolerance=1e-4, max_iter=150):

    print ' - Wavelet Levels:', wavelet_levels
    print ' - Threshold Factor:', thresh_factor
    print ' - Reweights:', n_reweights

    if isinstance(filters, type(None)):
            filters = get_mr_filters(image.shape, wavelet_levels, wavelet_opt)

    filters_rot = rotate_filters(filters)

    x_old = np.ones(image.shape)
    y_old = np.ones([filters.shape[0]] + list(image.shape))

    transpose_test(convolve_mr_filters, deconvolve_mr_filters, x_old.shape,
                   (filters,), y_old.shape, (filters_rot,))

    if isinstance(weights, type(None)):
        error_weights = get_noise_est(psf_pcs, psf_coef, filters)

    if isinstance(inv_spec_rad, type(None)):
        beta = power_method(psf_pcs, psf_coef)
    else:
        beta = 1.0 / inv_spec_rad

    l1norm_filters = sum([norm(filter, 1) for filter in filters])
    tau = 1.0 / (beta + l1norm_filters)
    # tau = (l1norm_filters**2 + inv_spec_rad**2/2)**(-1)
    sigma = tau
    # sigma=1
    rho_n = 0.5

    print ''
    print ' SPEC_RAD:', beta
    print ' SIGMA:', sigma
    print ' TAU:', tau
    print ' TAU/SIGMA TEST:', (1 / tau - sigma * l1norm_filters ** 2 >= beta / 2)
    print ' RHO_n:', rho_n
    print ''
    print ' i COST          L2_NORM       L1_NORM'

    # Set initial weights
    weights = thresh_factor * np.copy(error_weights)

    for j in range(n_reweights + 1):
        for i in range(max_iter):

            grad = grad_step(x_old, image, psf_coef, psf_pcs)
            # weights = sigma * get_sigma_map(grad)

            x_prox = x_old - tau * grad - tau * deconvolve_mr_filters(y_old, filters_rot)
            x_temp = keep_positive(x_prox)

            y_prox = y_old + sigma * convolve_mr_filters(2 * x_temp - x_old, filters)
            y_temp = y_prox - sigma * weighted_threshold(y_prox / sigma, 1 / sigma, weights)

            x_new, y_new = rho_n * np.array([x_temp, y_temp]) + (1 - rho_n) * np.array([x_old, y_old])

            np.copyto(x_old, x_new)
            np.copyto(y_old, y_new)

            l2norm = norm(image - pca_convolve(x_new, psf_pcs, psf_coef), 2)
            l1norm = sum([norm(a, 1) for a in np.multiply(weights, convolve_mr_filters(x_new, filters))])
            cost = 0.5 * l2norm ** 2 + sigma * l1norm

            print '', i, cost, l2norm, l1norm

        # Reweight
        x_wave = convolve_mr_filters(x_new, filters)
        weights *= (1.0 / (1.0 + np.abs(x_wave) / (thresh_factor * error_weights)))
        print ''

    return x_new
