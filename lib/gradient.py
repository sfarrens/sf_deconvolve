# -*- coding: utf-8 -*-

"""GRADIENT CLASSES

This module contains classses for defining PSF deconvolution specific
gradients.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 19/07/2017

"""

import numpy as np
from sf_tools.signal.gradient import GradBasic
from sf_tools.math.matrix import PowerMethod
from sf_tools.base.transform import cube2matrix, matrix2cube
from sf_tools.image.convolve import psf_convolve, convolve_stack
from sf_tools.image.shape import shape_project


class GradPSF(PowerMethod):
    """Gradient class for PSF convolution

    This class defines the operators for a fixed or object variant PSF

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D observed images (i.e. with noise)
    psf : np.ndarray
        PSF, a single 2D PSF or an array of 2D PSFs
    psf_type : str {'fixed', 'obj_var'}
        PSF type (defualt is 'fixed')

    Notes
    -----
    The properties of `PowerMethod` are inherited in this class

    """

    def __init__(self, data, psf, psf_type='fixed'):

        self._y = np.copy(data)
        self._psf = np.copy(psf)
        self._psf_type = psf_type

        PowerMethod.__init__(self, lambda x: self.Ht_op(self.H_op(x)),
                             self._y.shape)

    def H_op(self, x):
        """H matrix operation

        This method calculates the action of the matrix H on the input data, in
        this case the convolution of the the input data with the PSF

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result

        """

        return psf_convolve(x, self._psf, psf_rot=False,
                            psf_type=self._psf_type)

    def Ht_op(self, x):
        """Ht matrix operation

        This method calculates the action of the transpose of the matrix H on
        the input data, in this case the convolution of the the input data with
        the rotated PSF

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result

        """

        return psf_convolve(x, self._psf, psf_rot=True,
                            psf_type=self._psf_type)

    def _calc_grad(self, x):

        return self.Ht_op(self.H_op(x) - self._y)


class GradKnownPSF(GradPSF):
    """Gradient class for a known PSF

    This class calculates the gradient when the PSF is known

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D observed images (i.e. with noise)
    psf : np.ndarray
        PSF, a single 2D PSF or an array of 2D PSFs
    psf_type : str {'fixed', 'obj_var'}
        PSF type (defualt is 'fixed')

    Notes
    -----
    The properties of `GradPSF` are inherited in this class

    """

    def __init__(self, data, psf, psf_type='fixed'):

        self.grad_type = 'psf_known'
        super(GradKnownPSF, self).__init__(data, psf, psf_type)

    def get_grad(self, x):
        """Get the gradient at the given iteration

        This method calculates the gradient value from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        """

        self.grad = self._calc_grad(x)


class GradUnknownPSF(GradPSF):
    """Gradient class for a unknown PSF

    This class calculates the gradient when the PSF is not fully known

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D observed images (i.e. with noise)
    psf : np.ndarray
        PSF, a single 2D PSF or an array of 2D PSFs
    psf_type : str {'fixed', 'obj_var'}
        PSF type (defualt is 'fixed')
    prox : class
        Proximity operator for PSF update
    beta_reg : float
        Gradient step size
    lambda_reg : float
        Regularisation control parameter

    Notes
    -----
    The properties of `GradPSF` are inherited in this class

    """

    def __init__(self, data, psf, prox, psf_type='fixed', beta_reg=1,
                 lambda_reg=1):

        if not hasattr(prox, 'op'):
            raise ValueError('prox must have "op()" method')

        self.grad_type = 'psf_unknown'
        self._prox = prox
        self._beta_reg = beta_reg
        self._lambda_reg = lambda_reg
        self._psf0 = np.copy(psf)
        super(GradUnknownPSF, self).__init__(data, psf, psf_type)

    def _update_lambda(self):
        """Update the regularisation parameter lambda_reg

        This method implements the update method for lambda_reg

        """

        self._lambda_reg = self._lambda_reg

    def _update_psf(self, x):
        """Update the current estimate of the PSF

        This method calculates the gradient of the PSF and updates the current
        estimate

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        """

        self._update_lambda()

        psf_grad = (convolve_stack(self.H_op(x) - self._y, x,
                    rot_kernel=True) + self._lambda_reg *
                    (self._psf - self._psf0))

        self._psf = self._prox.op(self._psf - self._beta_reg * psf_grad)

    def get_grad(self, x):
        """Get the gradient at the given iteration

        This method calculates the gradient value from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        """

        self._update_psf(x)
        self.grad = self._calc_grad(x)


class GradShape(GradPSF):
    """Gradient class for shape constraint

    This class calculates the gradient including shape constraint

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D observed images (i.e. with noise)
    psf : np.ndarray
        PSF, a single 2D PSF or an array of 2D PSFs
    psf_type : str {'fixed', 'obj_var'}
        PSF type (defualt is 'fixed')
    lambda_reg : float
        Regularisation control parameter

    Notes
    -----
    The properties of `GradPSF` are inherited in this class

    """

    def __init__(self, data, psf, psf_type='fixed', lambda_reg=1):

        self.grad_type = 'shape'
        self._y = np.copy(data)
        self._psf = np.copy(psf)
        self._psf_type = psf_type
        self.lambda_reg = lambda_reg
        self._St = cube2matrix(shape_project(data.shape[1:]))
        self._S = self._St.T

        def func(x):

            HX = cube2matrix(self.H_op(x))

            StSHX = self._St.dot(self._S.dot(HX))

            return (self.Ht_op(self.H_op(x)) + self.lambda_reg *
                    self.Ht_op(matrix2cube(StSHX, x[0].shape)))

        PowerMethod.__init__(self, func, self._y.shape)

    def _calc_shape_grad(self, x):
        """Get the gradient of the shape constraint component

        This method calculates the gradient value of the shape constraint
        component

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        """

        HXY = cube2matrix(self.H_op(x) - self._y)

        StSHXY = self._St.dot(self._S.dot(HXY))

        return self.Ht_op(matrix2cube(StSHXY, x[0].shape))

    def get_grad(self, x):
        """Get the gradient at the given iteration

        This method calculates the gradient value from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        """

        self.grad = (self._calc_grad(x) + self.lambda_reg *
                     self._calc_shape_grad(x))


class GradNone(GradPSF):
    """No gradient class

    This is a dummy class that returns an array of zeroes for the gradient

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D observed images (i.e. with noise)
    psf : np.ndarray
        PSF, a single 2D PSF or an array of 2D PSFs
    psf_type : str {'fixed', 'obj_var'}
        PSF type (defualt is 'fixed')

    """

    def get_grad(self, x):
        """Get the gradient step

        This method returns an array of zeroes

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.zeros array size

        """

        self.grad = np.zeros(x.shape)
