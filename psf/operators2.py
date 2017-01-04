#  @file operators.py
#
#  OPERATOR CLASSES
#
#  Classes for defining algorithm operators and gradients.
#  Based on work by Yinghao Ge and Fred Ngole.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from scipy.linalg import norm
from algorithms import PowerMethod
from convolve import pca_convolve_stack
from wavelet import *


##
#  Class for defining the basic methods of a gradient operator.
#
class GradBasic():

    ##
    #  Method to calculate the action of the transpose of the matrix M on the
    #  action of the matrix M on the data X.
    #
    #  @param[in] x: Input data.
    #
    #  @return M.TMX.
    #
    def MtMX(self, x):

        return self.MtX(self.MX(x))

    ##
    #  Method to calculate gradient step.
    #
    #  @param[in] x: Input data.
    #
    #  Calculates: M.T (MX - Y)
    #
    #  @return Gradient step.
    #
    def get_grad(self, x):

        self.grad = self.MtX(self.MX(x) - self.y)


##
#  Class for defining a gradient of zero.
#
class GradZero(GradBasic):

    ##
    #  Method to calculate gradient step.
    #
    #  @param[in] x: Input data.
    #
    #  Calculates: M.T (MX - Y)
    #
    #  @return Gradient step.
    #
    def get_grad(self, x):

        self.grad = np.zeros(x.shape)


##
#  Class for defining the operators of a fixed or object variant PSF.
#
class StandardPSF(GradBasic, PowerMethod):

    ##
    #  Method that initialises the class instance.
    #
    #  @param[in] data: 2D Input array. (Noisy image)
    #  @param[in] psf: PSF.
    #  @param[in] psf_type: PSF type. ('fixed' or 'obj_var')
    #
    def __init__(self, data, psf, psf_type='fixed'):

        self.y = data
        self.psf = psf

        if psf_type in ('fixed', 'obj_var'):
            self.psf_type = psf_type
        else:
            raise ValueError('Invalid PSF type. Options are fixed or obj_var')

        PowerMethod.__init__(self, self.MtMX, self.y.shape, auto_run=False)

    ##
    #  Method to calculate the action of the matrix M on the data X.
    #
    #  @param[in] x: Input data.
    #
    #  @return MX.
    #
    def MX(self, x):

        return psf_convolve(x, self.psf, psf_rot=False, psf_type=self.psf_type)

    ##
    #  Method to calculate the action of the transpose of the matrix M on the
    #  data X.
    #
    #  @param[in] x: Input data.
    #
    #  @return M.TX.
    #
    def MtX(self, x):

        return psf_convolve(x, self.psf, psf_rot=True, psf_type=self.psf_type)


##
#  Class for defining the operators of a fixed or object variant PSF with
#  no gradient.
#
class StandardPSFnoGrad(GradZero, StandardPSF):

    pass


##
#  Class for defining the operators of a pixel variant PSF.
#
class PixelVariantPSF(GradBasic, PowerMethod):

    ##
    #  Method that initialises the class instance.
    #
    #  @param[in] data: 2D Input array. (Noisy image)
    #  @param[in] psf_pcs: PSF principal components.
    #  @param[in] psf_coef: PSF coefficients.
    #
    #  @exception ValueError for invalid data format.
    #
    def __init__(self, data, psf_pcs, psf_coef):

        self.y = data
        self.psf_pcs = np.array(psf_pcs)
        self.psf_coef = np.array(psf_coef)

        PowerMethod.__init__(self, self.MtMX, self.psf_coef.shape[1:],
                             auto_run=False)

    ##
    #  Method to calculate the action of the matrix M on the data X.
    #
    #  @param[in] x: Input data.
    #
    #  @return MX.
    #
    def MX(self, x):

        return pca_convolve_stack(x, self.psf_pcs, self.psf_coef,
                                  pcs_rot=False)

    ##
    #  Method to calculate the action of the transpose of the matrix M on the
    #  data X.
    #
    #  @param[in] x: Input data.
    #
    #  @return M.TX.
    #
    def MtX(self, x):

        return pca_convolve_stack(x, self.psf_pcs, self.psf_coef,
                                  pcs_rot=True)
