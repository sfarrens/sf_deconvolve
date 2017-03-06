# -*- coding: utf-8 -*-

"""OPERATOR CLASSES

This module contains classses for defining algorithm operators and gradients.
Based on work by Yinghao Ge and Fred Ngole.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.1

:Date: 04/01/2017

"""

import numpy as np
from algorithms import PowerMethod
from convolve import psf_convolve


class GradBasic(object):
    """Basic gradient class

    This class defines the basic methods that will be inherited by specific
    gradient classes

    """

    def MtMX(self, x):
        """M^T M X

        This method calculates the action of the transpose of the matrix M on
        the action of the matrix M on the data X

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray result

        Notes
        -----
        Calculates  M^T (MX)

        """

        return self.MtX(self.MX(x))

    def get_grad(self, x):
        """Get the gradient step

        This method calculates the gradient step from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray gradient value

        Notes
        -----

        Calculates M^T (MX - Y)

        """

        self.grad = self.MtX(self.MX(x) - self.y)


class GradZero(GradBasic):
    """Zero gradient class

    This is a dummy class that returns an array of zeroes for the gradient

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


class StandardPSF(GradBasic, PowerMethod):
    """Standard PSF class

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
    The properties of `GradBasic` and `PowerMethod` are inherited in this class

    """

    def __init__(self, data, psf, psf_type='fixed'):

        self.y = data
        self.psf = psf
        self.psf_type = psf_type

        PowerMethod.__init__(self, self.MtMX, self.y.shape, auto_run=False)

    def MX(self, x):
        """MX

        This method calculates the action of the matrix M on the data X, in
        this case the convolution of the the input data with the PSF

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result

        """

        return psf_convolve(x, self.psf, psf_rot=False, psf_type=self.psf_type)

    def MtX(self, x):
        """MX

        This method calculates the action of the transpose of the matrix M on
        the data X, in this case the convolution of the the input data with the
        rotated PSF

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result

        """

        return psf_convolve(x, self.psf, psf_rot=True, psf_type=self.psf_type)


class StandardPSFnoGrad(GradZero, StandardPSF):
    """No gradient class

    This is a dummy class that inherits `GradZero` and `StandardPSF`

    """

    pass
