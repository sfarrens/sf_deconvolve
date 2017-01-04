#  @file pca.py
#
#  RECONSTRUCTION ROUTINES
#
#  Functions for reconstructing
#  an image from the observation.
#  Based on work by Yinghao Ge.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from scipy.signal import fftconvolve
from scipy.linalg import norm
from noise import threshold
from convolve import pca_convolve
from transform import *
from reconstruct4 import *


##
#  Function to calculate the action of the matrix M on the data X.
#
#  @param[in] x: Input data.
#  @param[in] m1: Matrix convolution elements.
#  @param[in] m2: Matrix multiplication elements.
#
#  @return MX.
#
def MX(x, m1, m2):

    return sum([np.multiply(fftconvolve(x, a, 'same'), b) for a, b in
                zip(m1, m2)])


##
#  Function to calculate the action of the transpose of the matrix M on the
#  data X.
#
#  @param[in] x: Input data.
#  @param[in] m1: Matrix convolution elements.
#  @param[in] m2: Matrix multiplication elements.
#
#  @return M.TX.
#
def MtX(x, m1, m2):

    return sum([fftconvolve(np.multiply(x, b), np.rot90(a, 2), 'same')
                for a, b in zip(m1, m2)])


##
#  Function to calculate the action of the transpose of the matrix M on the
#  action of the matrix M on the data X.
#
#  @param[in] x: Input data.
#  @param[in] m1: Matrix convolution elements.
#  @param[in] m2: Matrix multiplication elements.
#
#  @return M.TMX.
#
def MtMX(x, m1, m2):

    return MtX(MX(x, m1, m2), m1, m2)


##
#  Power method for iteration.
#
#  @param[in] m1: Matrix convolution elements.
#  @param[in] m2: Matrix multiplication elements.
#  @param[in] tolerance: Convergence tolerance.
#  @param[in] max_iter: Maximum number of iterations.
#
#  @return Spectral radius.
#
def power_method(m1, m2, tolerance=1e-5, max_iter=150):

    # Generate an array of random values.
    x0 = np.random.random(m2.shape[1:])

    # Iterate MtMX until the L2 norm of x converges.
    for i in range(max_iter):

        if i == 0:
            x = MtMX(x0, m1, m2)

        else:
            x = MtMX(x0 / norm(x0, 2), m1, m2)

        if(np.abs(norm(x, 2) - norm(x0, 2)) < tolerance):
            print ' - Power Method converged after %d iterations!' % (i + 1)
            break

        elif i == max_iter - 1:
            print ' - Power Method did not converge after %d iterations!' % \
                   max_iter
        x0 = x

    return norm(x, 2)


##
#  Power method for iteration.
#
#  @param[in] ua: U and A elements of SVD of image reconstruction.
#  @param[in] m1: Matrix convolution elements.
#  @param[in] m2: Matrix multiplication elements.
#  @param[in] shapes: Matrix/Map transform shapes.
#  @param[in] tolerance: Convergence tolerance.
#  @param[in] max_iter: Maximum number of iterations.
#
#  @return Spectral radius.
#
def power_method_UA(ua, m1, m2, shapes, tolerance=1e-5, max_iter=150):

    # Generate an array of random values.
    a0 = np.random.random(ua[1].shape)

    # Iterate MtMX until the difference of the L2 norms converges.
    for i in range(max_iter):
        x0 = matrix2map(np.dot(ua[0], a0 / norm(ua[1], 2)), shapes[0])
        ua[1] = np.dot(ua[0].T, map2matrix(MtMX(x0, m1, m2), shapes[1]))

        if(np.abs(norm(ua[1], 2) - norm(a0, 2)) < tolerance):
            print ' - Power Method converged after %d iterations!' % (i + 1)
            break

        elif i == max_iter - 1:
            print ' - Power Method did not converge after %d iterations!' % \
                   max_iter
        a0 = ua[1]

    return norm(ua[1], 2)


##
#  Function to calculate gradient step.
#
#  @param[in] data_rec: Current data reconstruction estimate.
#  @param[in] data: Observed data.
#  @param[in] psf_coef: PSF coefficients.
#  @param[in] psf_pcs: PSF principal components.
#  @param[in] inv_spec_rad: Inverse spectral radius.
#
#  Calculates: M.T (MX_n - Y)
#
#  @return Gradient step.
#
def grad_step(data_rec, data, psf_coef, psf_pcs):

    part =  MX(data_rec, psf_pcs, psf_coef) - data

    return MtX(part, psf_pcs, psf_coef)


##
#  Function that retains the positive values of the input data.
#
#  @param[in] data: Input data.
#
#  @return Positive values of the data and zeros elsewehere.
#
def keep_positive(data):

    return data * (data > 0)


##
#  Function to speed up gradien descent.
#
#  @param[in] data: Current iteration data.
#  @param[in] data0: Previous iteration data.
#  @param[in] t0: Previous iteration value of t.
#
#  @return Updated data and t values.
#
def speed_up(data, data0, t0):

    t = (1 + np.sqrt(4 * t0 * t0 + 1)) * 0.5
    lambda0 = 1 + (t0 - 1) / t

    return data0 + lambda0 * (data - data0), t


##
#  Function that perfoms gradient descent to
#  reconstruct an image.
#
#  @param[in] image: Input image.
#  @param[in] psf_pcs: PSF principal components.
#  @param[in] psf_coef: PSF coefficients.
#  @param[in] image_rec: Image reconstruction.
#  @param[in] layout: 2D Layout of image map.
#  @param[in] thresh_factor: Constant factor for threshold.
#  @param[in] tolerance: Convergence tolerance.
#  @param[in] max_iter: Maximum number of iterations.
#
#  @return Reconstucted image.
#
def gradient_descent(image, psf_pcs, psf_coef, image_rec=None, layout=None,
                     inv_spec_rad=None, thresh_factor=3, tolerance=1e-4,
                     max_iter=150):

    operator = PixelVariantPSF(psf_pcs, psf_coef)

    if isinstance(image_rec, type(None)):

        # Set inital guess for gradient descent.
        image_rec = np.ones(image.shape)

        # Get the inverse spectral radius
        if isinstance(inv_spec_rad, type(None)):
            inv_spec_rad = 1. / power_method(psf_pcs, psf_coef)

        rec_flag = False

    else:

        # Get SVD of the initial image reconstruction.
        image_matrix = map2matrix(image_rec, layout)
        u, s, v = np.linalg.svd(image_matrix)
        s = np.diag(s)
        s.resize(image_matrix.shape)
        a = np.dot(s, v)
        a0 = np.copy(a)

        # Get the inverse spectral radius
        if isinstance(inv_spec_rad, type(None)):
            inv_spec_rad = 1. / power_method_UA([u, a], psf_pcs, psf_coef,
                                                [image.shape, layout])

        rec_flag = True

    # Set inital values for testing convergence.
    image_rec_xn = np.copy(image_rec)
    image_rec_zn = np.copy(image_rec_xn)
    cost0 = 0.0
    t0 = 1.0
    nuc_norm = 0.0
    thresh = 0.0
    thresh0 = 100.0
    update_thresh = True
    use_speed_up = True

    print '   * 1/rho =', inv_spec_rad

    # Perfom gradient descent to reconstruct image.
    for i in range(max_iter):

        if rec_flag:
            image_matrix = map2matrix(-1. * grad_step(image_rec_zn, image,
                                      psf_coef, psf_pcs), layout)
            a1 = a + inv_spec_rad * np.dot(u.T, image_matrix)
            a1 = np.dot(u.T, keep_positive(np.dot(u, a1)))
            a, t0 = speed_up(a1, a0, t0)
            image_rec_zn = matrix2map(np.dot(u, a), image.shape)
            a0 = np.copy(a1)

        else:
            # Calculate the gradient for this step.
            # grad = grad_step(image_rec_zn, image, psf_coef, psf_pcs)
            grad = operator.grad_step(image_rec_zn, image)

            uu, ss, vv = np.linalg.svd(map2matrix(grad, layout))
            if update_thresh:
                thresh = thresh_factor * np.median(ss)
                if np.abs(thresh - thresh0) < tolerance:
                    update_thresh = False
                    print ' - Threshold converged!'
                else:
                    thresh0 = thresh
            if update_thresh and i == 50:
                update_thresh = False
                print ' - Threshold stabalised after 50 iterations.'

            image_rec_yn = image_rec_zn - inv_spec_rad * grad

            image_matrix = map2matrix(image_rec_yn, layout)
            image_matrix, nuc_normxx = threshold(image_matrix, thresh,
                                                 threshold_type='soft',
                                                 return_nuc_norm=True)
            image_rec_xn_1 = matrix2map(image_matrix, image.shape)

            l2norm0 = norm(image - pca_convolve(image_rec_xn, psf_pcs, psf_coef), 2)
            l2norm1 = norm(image - pca_convolve(image_rec_xn_1, psf_pcs, psf_coef), 2)

            if l2norm1 > l2norm0:
                use_speed_up = False
            if use_speed_up:
                image_rec_zn, t0 = speed_up(image_rec_xn_1, image_rec_xn, t0)
            else:
                image_rec_zn = np.copy(image_rec_xn_1)

            image_rec_xn = np.copy(image_rec_xn_1)

        u3, s3, v3 = np.linalg.svd(map2matrix(image_rec_zn, layout))
        nuc_norm = np.sum(s3)
        l2norm = norm(image - pca_convolve(image_rec_zn, psf_pcs, psf_coef), 2)
        cost = (0.5 * l2norm ** 2 + thresh * nuc_norm)
        # print ' - i:', i, cost, thresh, l2norm

        if np.abs(cost - cost0) < tolerance:
            print ' - Gradient Descent converged after %d iterations!' % \
                  (i + 1)
            print ' - Final cost, thresh, l2norm:', cost, thresh, l2norm
            break

        elif i == max_iter - 1:
            print ' - Gradient Descent did not converge after %d iterations!' \
                  % max_iter
            print ' - Final cost, thresh, l2norm:', cost, thresh, l2norm

        cost0 = cost

    image_rec = image_rec_zn

    return image_rec
