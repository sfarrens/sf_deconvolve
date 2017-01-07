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
from noise_old import threshold
from convolve_old import pca_convolve
from transform_old import *


##
#  Function to calculate M.T * MX.
#
#  @param[in] x: ???
#  @param[in] d: ???
#  @param[in] h: ???
#
#  @return X.
#
def MtMX(x, d, h):

    part = sum([np.multiply(fftconvolve(x, a.T, 'same'), b.T)
                for a, b in zip(h.T, d.T)])

    return sum([fftconvolve(np.multiply(part, b.T), np.rot90(a.T, 2), 'same')
                for a, b in zip(h.T, d.T)])


##
#  Power method for iteration.
#
#  @param[in] d: ???
#  @param[in] h: ???
#  @param[in] tolerance: Convergence tolerance.
#  @param[in] max_iter: Maximum number of iterations.
#
#  @return Spectral radius.
#
def power_method(d, h, tolerance=1e-5, max_iter=250):

    # Generate an array of random values.
    x0 = np.random.random(d.shape[:2])

    # Iterate MtMX until the L2 norm of x converges.
    for i in range(max_iter):
        if i == 0:
            x = MtMX(x0, d, h)
        else:
            x = MtMX(x0 / norm(x0, 2), d, h)
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
#  @param[in] ua: ???
#  @param[in] d: ???
#  @param[in] h: ???
#  @param[in] shapes: Matrix/Map transform shapes.
#  @param[in] tolerance: Convergence tolerance.
#  @param[in] max_iter: Maximum number of iterations.
#
#  @return Spectral radius.
#
def power_method_UA(ua, d, h, shapes, tolerance=1e-5, max_iter=250):

    # Generate an array of random values.
    a0 = np.random.random(ua[1].shape)

    # Iterate MtMX until the difference of the L2 norms converges.
    for i in range(max_iter):
        x0 = matrix2map(np.dot(ua[0], a0 / norm(ua[1], 2)), shapes[0])
        ua[1] = np.dot(ua[0].T, map2matrix(MtMX(x0, d, h), shapes[1]))
        if(np.abs(norm(ua[1], 2) - norm(a0, 2)) < tolerance):
            print ' - Power Method converged after %d iterations!' % (i + 1)
            break
        elif i == max_iter - 1:
            print ' - Power Method did not converge after %d iterations!' % \
                   max_iter
        a0 = ua[1]

    return norm(ua[1], 2)


##
#  Function to calculate M.T _ Y _ MX.
#
#  @param[in] x: ???
#  @param[in] y: ???
#  @param[in] d: ???
#  @param[in] h: ???
#  @param[in] pas: ???
#
#  @return X.
#
def Mt_Y_MX(x, y, d, h, pas):

    part = y - sum([np.multiply(fftconvolve(x, a.T, 'same'), b.T)
                    for a, b in zip(h.T, d.T)])

    temp = x
    if pas == 1:
        temp = 0

    return temp + sum([fftconvolve(np.multiply(part, b.T), np.rot90(a.T, 2),
                       'same') * pas for a, b in zip(h.T, d.T)])


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
#  Function that perfoms gradient descent to
#  reconstruct an image.
#
#  @param[in] image: Input image.
#  @param[in] psf_pcs: PSF principal components.
#  @param[in] psf_coef: PSF coefficients.
#  @param[in] image_rec: Image reconstruction.
#  @param[in] layout: 2D Layout of image map.
#  @param[in] tolerance: Convergence tolerance.
#  @param[in] max_iter: Maximum number of iterations.
#
#  @return Reconstucted image.
#
def gradient_descent(image, psf_pcs, psf_coef, image_rec=None, layout=None,
                     tolerance=1e-5, max_iter=250):

    if isinstance(image_rec, type(None)):

        # Set inital guess for gradient descent.
        image_rec = np.ones(image.shape)

        # Get the inverse spectral radius
        inv_spec_rad = 1. / power_method(psf_coef, psf_pcs)
        rec_flag = False

    else:

        # Get SVD of the initial image reconstruction.
        image_matrix = map2matrix(image_rec, layout)
        u, s, v = np.linalg.svd(image_matrix)
        s = np.diag(s)
        s.resize(image_matrix.shape)
        a = np.dot(s, v)

        # Get the inverse spectral radius
        inv_spec_rad = 1. / power_method_UA([u, a], psf_coef, psf_pcs,
                                            [image.shape, layout])
        rec_flag = True

    # Set inital value for testing convergence.
    diff0 = 0.0

    print '   * 1/rho =', inv_spec_rad

    # Perfom gradient descent to reconstruct image.
    for i in range(max_iter):
        if rec_flag:
            mat = map2matrix(Mt_Y_MX(image_rec, image, psf_coef, psf_pcs, 1.0),
                             layout)
            a += inv_spec_rad * np.dot(u.T, mat)
            a = np.dot(u.T, keep_positive(np.dot(u, a)))
            image_rec = matrix2map(np.dot(u, a), image.shape)
        else:
            image_rec = Mt_Y_MX(image_rec, image, psf_coef, psf_pcs,
                                inv_spec_rad)
            image_rec = matrix2map(threshold(map2matrix(image_rec, layout),
                                   threshold_type='soft'), image.shape)

        diff = norm(image - pca_convolve(image_rec, psf_pcs, psf_coef), 2)

        if np.abs(diff - diff0) < tolerance:
            print ' - Gradient Descent converged after %d iterations!' % \
                  (i + 1)
            break
        elif i == max_iter - 1:
            print ' - Gradient Descent did not converge after %d iterations!' \
                  % max_iter
        diff0 = diff

    return image_rec
