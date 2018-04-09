# -*- coding: utf-8 -*-

"""GRADIENT CLASSES

This module contains classses for defining PSF deconvolution specific
gradients.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 19/07/2017

"""

from __future__ import print_function
import numpy as np
from modopt.base.np_adjust import rotate, rotate_stack
from modopt.base.transform import cube2matrix, matrix2cube
from modopt.base.types import check_float, check_npndarray
from modopt.math.matrix import PowerMethod
from modopt.math.convolve import convolve, convolve_stack
from modopt.opt.gradient import GradParent


def psf_convolve(data, psf, psf_rot=False, psf_type='fixed', method='scipy'):
    """Convolve data with PSF

    This method convolves an image with a PSF

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally an array of 2D images
    psf : np.ndarray
        Input PSF array, normally either a single 2D PSF or an array of 2D
        PSFs
    psf_rot: bool
        Option to rotate PSF by 180 degrees
    psf_type : str {'fixed', 'obj_var'}, optional
        PSF type (default is 'fixed')
    method : str {'astropy', 'scipy'}, optional
        Convolution method (default is 'astropy')

        'fixed':
            The PSF is fixed, i.e. it is the same for each image

        'obj_var':
            The PSF is object variant, i.e. it is different for each image

    Returns
    -------
    np.ndarray convolved data

    Raises
    ------
    ValueError
        If `psf_type` is not 'fixed' or 'obj_var'

    """

    if psf_type not in ('fixed', 'obj_var'):
        raise ValueError('Invalid PSF type. Options are "fixed" or "obj_var"')

    if psf_rot and psf_type == 'fixed':
        psf = rotate(psf)

    elif psf_rot:
        psf = rotate_stack(psf)

    if psf_type == 'fixed':
        return np.array([convolve(data_i, psf, method=method) for data_i in
                        data])

    elif psf_type == 'obj_var':

        return convolve_stack(data, psf, method=method)


class GradPSF(GradParent, PowerMethod):
    """Gradient class for PSF convolution

    This class defines the operators for a fixed or object variant PSF

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D observed images (e.g. with noise)
    psf : np.ndarray
        PSF, a single 2D PSF or an array of 2D PSFs
    psf_type : str {'fixed', 'obj_var'}
        PSF type (defualt is 'fixed')
    convolve_method : str {'astropy', 'scipy'}, optional
        Convolution method (default is 'astropy')

    Notes
    -----
    The properties of `PowerMethod` are inherited in this class

    """

    def __init__(self, data, psf, psf_type='fixed', convolve_method='astropy',
                 data_type=float):

        self._grad_data_type = data_type
        self.obs_data = data
        self.op = self._H_op_method
        self.trans_op = self._Ht_op_method
        check_float(psf)
        check_npndarray(psf, writeable=False)
        self._psf = psf
        self._psf_type = psf_type
        self._convolve_method = convolve_method

        PowerMethod.__init__(self, self.trans_op_op, self.obs_data.shape)

    def _H_op_method(self, x):
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
                            psf_type=self._psf_type,
                            method=self._convolve_method)

    def _Ht_op_method(self, x):
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
                            psf_type=self._psf_type,
                            method=self._convolve_method)

    def _calc_grad(self, x):

        return self.trans_op(self.op(x) - self.obs_data)


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
    convolve_method : str {'astropy', 'scipy'}, optional
        Convolution method (default is 'astropy')

    Notes
    -----
    The properties of `GradPSF` are inherited in this class

    """

    def __init__(self, data, psf, psf_type='fixed', convolve_method='astropy'):

        self.grad_type = 'psf_known'
        self.get_grad = self._get_grad_method
        self.cost = self._cost_method
        super(GradKnownPSF, self).__init__(data, psf, psf_type,
                                           convolve_method)

    def _get_grad_method(self, x):
        """Get the gradient at the given iteration

        This method calculates the gradient value from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        """

        self.grad = self._calc_grad(x)

    def _cost_method(self, *args, **kwargs):
        """Calculate gradient component of the cost

        This method returns the l2 norm error of the difference between the
        original data and the data obtained after optimisation

        Returns
        -------
        float gradient cost component

        """

        cost_val = 0.5 * np.linalg.norm(self.obs_data - self.op(args[0])) ** 2

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - DATA FIDELITY (X):', cost_val)

        return cost_val


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
    convolve_method : str {'astropy', 'scipy'}, optional
        Convolution method (default is 'astropy')
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

    def __init__(self, data, psf, prox, psf_type='fixed',
                 convolve_method='astropy', beta_reg=1, lambda_reg=1):

        if not hasattr(prox, 'op'):
            raise ValueError('prox must have "op()" method')

        self.grad_type = 'psf_unknown'
        self.get_grad = self._get_grad_method
        self.cost = self._cost_method
        self._prox = prox
        self._beta_reg = beta_reg
        self._lambda_reg = lambda_reg
        self._psf0 = np.copy(psf)
        self._convolve_method = convolve_method
        super(GradUnknownPSF, self).__init__(data, psf, psf_type,
                                             convolve_method)

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

        psf_grad = (convolve_stack(self.op(x) - self.obs_data, x,
                    rot_kernel=True, method=self._convolve_method) +
                    self._lambda_reg * (self._psf - self._psf0))

        self._psf = self._prox.op(self._psf - self._beta_reg * psf_grad)

    def _get_grad_method(self, x):
        """Get the gradient at the given iteration

        This method calculates the gradient value from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        """

        self._update_psf(x)
        self.grad = self._calc_grad(x)

    def _cost_method(self, *args, **kwargs):
        """Calculate gradient component of the cost

        This method returns the l2 norm error of the difference between the
        original data and the data obtained after optimisation

        Returns
        -------
        float gradient cost component

        """

        cost_val = (0.5 * np.linalg.norm(self.obs_data - self.op(args[0])) ** 2
                    + np.linalg.norm(self._psf - self._psf0) ** 2)

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - DATA FIDELITY + PSF CONSTRAINT (X):', cost_val)

        return cost_val


class GradNone(GradPSF):
    """No gradient class

    This is a dummy class that returns an array of zeroes for the gradient

    """

    def __init__(*args):

        self.grad_type = 'none'
        self.get_grad = self._get_grad_method
        super(GradNone, self).__init__(*args)

    def _get_grad_method(self, x):
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
