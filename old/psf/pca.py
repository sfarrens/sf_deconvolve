#  @file pca.py
#
#  PCA ROUTINES
#
#  Functions for performing
#  principal component analysis.
#  Based on work by Yinghao Ge.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from scipy.sparse import linalg
from scipy import linalg as linalg2
from functions.matrix import *


##
#  Function that perfoms principal component analysis
#  on an input PSF assuming sparse matrices.
#
#  @param[in] data: Input Data.
#  @param[in] n_values: Number of singular values.
#
#  @return Singular vectors and values (U, S, V).
#
def svd_sparse(data, n_values):

        # Perform SVD
        u, s, v = linalg.svds(data, n_values)

        # Sort the singular values and vectors in descending order.
        index = np.argsort(s)[::-1]

        return u[:, index], s[index], v[index]


##
#  Function that returns the PCA coefficients
#  for an input PSF and its corresponding
#  principal components.
#
#  @param[in] psf: PSF.
#  @param[in] psf_pcs: PSF principal components.
#  @param[in] shape: Desired shape of the output.
#
#  @return PCA coefficients.
#
def get_coef(psf, psf_pcs, shape=None):

    # Calculate the coefficients.
    coef = np.array([np.sum(a * b) for a in psf_pcs for b in psf])

    # Return the reshaped coefficients.
    if shape:
        return coef.reshape([psf_pcs.shape[0]] + list(shape))
    else:
        return coef.reshape(psf_pcs.shape[0], psf.shape[0])


##
#  Function that perfoms principal component analysis on an input PSF to get
#  the principal components.
#
#  @param[in] psf: PSF.
#  @param[in] threshold: PCA threshold.
#  @param[in] svd_type: Type of SVD to perform.
#
#  @return PSF principal components.
#
def psf_pca(psf, threshold, svd='sparse'):

    # Convert PSF cube to column vectors.
    psf = ftl(psf)
    psf_column = psf.reshape(np.prod(psf.shape[:2]), psf.shape[2])

    # Perform either full of sparse SVD.
    if svd is 'full':
        u, s, v = np.linalg.svd(psf_column)
    else:
        u, s, v = svd_sparse(psf_column, psf_column.shape[0] - 1)

    # Find the index of the last required singular value.
    last_pc = (np.array([s[:i].sum() for i in xrange(s.size)]) - threshold *
               s.sum())
    last_pc = np.where(last_pc == min(last_pc[last_pc >= 0.0]))[0][0]

    # Get the principal components of the PSF.
    psf_pcs = u[:, :last_pc].reshape(list(psf.shape[:2]) + [last_pc])

    # Transpose and return the princicpal components.
    return ftr(psf_pcs)


##
#  Function that perfoms principal component analysis
#  in parts on an input PSF.
#
#  @param[in] psf: PSF.
#  @param[in] thresholds: Array of PCA thresholds.
#  @param[in] n_pieces: Number of pieces of the PSF.
#  @param[in] return_coef: Option to return PCA coefficients.
#  @param[in] coef_shape: Shape of ouput coefficients.
#
#  @return PSF principal components and coefficients.
#
def psf_pca_parts(psf, thresholds, n_pieces, return_coef=True,
                  coef_shape=None):

        # Convert inputs to Numpy arrays
        thresholds = np.array(thresholds)

        # Split the PSF into pieces.
        pieces = np.split(psf, np.arange(1, n_pieces) * psf.shape[2] /
                          n_pieces, axis=2)

        # Get the princicpal components for each of the pieces.
        res = [psf_pca(piece, thresholds[0], 'full', False)
               for piece in pieces]
        psf_pcs_pieces = np.vstack([val.T for val in res]).T

        # Get the final princicpal components.
        psf_pcs = psf_pca(psf_pcs_pieces, thresholds[1], 'full', False)

        if return_coef:
            # Get the PCA coefficients.
            return psf_pcs, get_coef(psf, psf_pcs, coef_shape)
        else:
            return psf_pcs
