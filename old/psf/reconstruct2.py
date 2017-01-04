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
from scipy.linalg import norm
from functions.matrix import nuclear_norm
from noise import threshold
from convolve import pca_convolve
from transform import *
from reconstruct import *
from operators import *
from algorithms import *


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
    operator.get_spec_rad()
    algorithm = LowRankMatrix(image.shape, layout)

    if isinstance(image_rec, type(None)):

        # Set inital guess for gradient descent.
        image_rec = np.ones(image.shape)

        # Get the inverse spectral radius
        if isinstance(inv_spec_rad, type(None)):
            inv_spec_rad = operator.inv_spec_rad

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
    use_speed_up = True

    print '   * 1/rho =', inv_spec_rad

    # Perfom gradient descent to reconstruct image.
    for i in range(max_iter):

        if rec_flag:
            image_matrix = map2matrix(-1. * operator.grad_step(image_rec_zn,
                                      image), layout)
            a1 = a + inv_spec_rad * np.dot(u.T, image_matrix)
            a1 = np.dot(u.T, keep_positive(np.dot(u, a1)))
            a, t0 = speed_up(a1, a0, t0)
            image_rec_zn = matrix2map(np.dot(u, a), image.shape)
            a0 = np.copy(a1)

        else:
            # Calculate the gradient for this step.
            grad = operator.grad_step(image_rec_zn, image)

            algorithm.check_threshold(i, grad)
            algorithm.update(grad, inv_spec_rad)

            l2norm0 = norm(image - pca_convolve(algorithm.data_rec_prev, psf_pcs, psf_coef), 2)
            l2norm1 = norm(image - pca_convolve(algorithm.data_rec, psf_pcs, psf_coef), 2)

            if l2norm1 > l2norm0:
                use_speed_up = False
            if use_speed_up:
                image_rec_zn, t0 = speed_up(algorithm.data_rec, algorithm.data_rec_prev, t0)
            else:
                image_rec_zn = np.copy(algorithm.data_rec)

        u3, s3, v3 = np.linalg.svd(map2matrix(image_rec_zn, layout))
        nuc_norm = np.sum(s3)
        l2norm = norm(image - pca_convolve(image_rec_zn, psf_pcs, psf_coef), 2)
        cost = (0.5 * l2norm ** 2 + algorithm.thresh * nuc_norm)
        # print ' - i:', i, cost, thresh, l2norm

        if np.abs(cost - cost0) < tolerance:
            print ' - Gradient Descent converged after %d iterations!' % \
                  (i + 1)
            print ' - Final cost, thresh, l2norm:', cost, algorithm.thresh, l2norm
            break

        elif i == max_iter - 1:
            print ' - Gradient Descent did not converge after %d iterations!' \
                  % max_iter
            print ' - Final cost, thresh, l2norm:', cost, algorithm.thresh, l2norm

        cost0 = cost

    image_rec = image_rec_zn

    return image_rec

class LowRankMethod():

    def __init__(self, data, psf_pcs, psf_coef, layout):

        self.data = data
        self.layout = layout
        self.operator = PixelVariantPSF(psf_pcs, psf_coef)
        self.algorithm = LowRankMatrix(data.shape, layout)

    def descent(self, tolerance=1e-4, max_iter=150):

        self.operator.get_spec_rad()
        cost0 = 0.0

        for i in range(max_iter):

            grad = self.operator.grad_step(self.algorithm.data_rec, self.data)
            self.algorithm.check_threshold(i, grad)
            self.algorithm.update(grad, self.operator.inv_spec_rad)

            l2norm0 = norm(self.data - pca_convolve(self.algorithm.data_rec_prev, self.operator.psf_pcs, self.operator.psf_coef), 2)
            l2norm1 = norm(self.data - pca_convolve(self.algorithm.data_rec, self.operator.psf_pcs, self.operator.psf_coef), 2)

            if l2norm1 > l2norm0:
                self.algorithm.speed_switch(False)

            nuc_norm = nuclear_norm(map2matrix(self.algorithm.data_rec,
                                    self.layout))
            l2norm = norm(self.data - pca_convolve(self.algorithm.data_rec,
                          self.operator.psf_pcs, self.operator.psf_coef), 2)
            cost = (0.5 * l2norm ** 2 + self.algorithm.thresh * nuc_norm)

            print ' - i:', i, cost, l2norm, nuc_norm

            if np.abs(cost - cost0) < tolerance:
                print ' - Gradient Descent converged after %d iterations!' % \
                      (i + 1)
                print ' - Final cost, l2norm, nuc_norm:', cost, l2norm, nuc_norm
                break

            elif i == max_iter - 1:
                print ' - Gradient Descent did not converge after %d iterations!' \
                      % max_iter
                print ' - Final cost, l2norm, nuc_norm:', cost, l2norm, nuc_norm

            cost0 = cost

        return self.algorithm.data_rec
