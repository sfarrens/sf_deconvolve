# -*- coding: utf-8 -*-

"""PSF DECONVOLUTION MODULE

This module deconvolves a set of galaxy images with a known object-variant PSF.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

"""

from __future__ import print_function
from builtins import range, zip
from scipy.linalg import norm
from modopt.math.stats import sigma_mad
from modopt.opt.algorithms import *
from modopt.opt.cost import costObj
from modopt.opt.linear import *
from modopt.opt.proximity import *
from modopt.opt.reweight import cwbReweight
from modopt.signal.wavelet import *
from gradient import *


def set_noise(data, **kwargs):
    """Set the noise level

    This method calculates the noise standard deviation using the median
    absolute deviation (MAD) of the input data and adds it to the keyword
    arguments.

    Parameters
    ----------
    data : np.ndarray
        Input noisy data (3D array)

    Returns
    -------
    dict Updated keyword arguments

    """

    # It the noise is not already provided calculate it using the MAD
    if isinstance(kwargs['noise_est'], type(None)):
        kwargs['noise_est'] = sigma_mad(data)

    print(' - Noise Estimate:', kwargs['noise_est'])
    if 'log' in kwargs:
        kwargs['log'].info(' - Noise Estimate: ' + str(kwargs['noise_est']))

    return kwargs


def set_grad_op(data, psf, **kwargs):
    """Set the gradient operator

    This method defines the gradient operator class to use and add an instance
    to the keyword arguments.

    Parameters
    ----------
    data : np.ndarray
        Input noisy data (3D array)
    psf : np.ndarray
        PSF data (2D or 3D array)

    Returns
    -------
    dict Updated keyword arguments

    """

    # Set the gradient operator
    if kwargs['grad_type'] == 'psf_known':
        kwargs['grad_op'] = (GradKnownPSF(data, psf,
                             psf_type=kwargs['psf_type'],
                             convolve_method=kwargs['convolve_method']))

    elif kwargs['grad_type'] == 'psf_unknown':
        kwargs['grad_op'] = (GradUnknownPSF(data, psf, Positive(),
                             psf_type=kwargs['psf_type'],
                             convolve_method=kwargs['convolve_method'],
                             beta_reg=kwargs['beta_psf'],
                             lambda_reg=kwargs['lambda_psf']))

    elif kwargs['grad_type'] == 'shape':
        kwargs['grad_op'] = (GradShape(data, psf, psf_type=kwargs['psf_type'],
                             convolve_method=kwargs['convolve_method'],
                             lambda_reg=kwargs['lambda_shape']))

    elif kwargs['grad_type'] == 'none':
        kwargs['grad_op'] = GradNone(data, psf, psf_type=kwargs['psf_type'],
                                     convolve_method=kwargs['convolve_method'])

    print(' - Spectral Radius:', kwargs['grad_op'].spec_rad)
    if 'log' in kwargs:
        kwargs['log'].info(' - Spectral Radius: ' +
                           str(kwargs['grad_op'].spec_rad))

    return kwargs


def set_linear_op(data, **kwargs):
    """Set the gradient operator

    This method defines the gradient operator class to use and add an instance
    to the keyword arguments. It additionally add the l1 norm of the linear
    operator and the wavelet filters (if used) to the kwagrs.

    Parameters
    ----------
    data : np.ndarray
        Input noisy data (3D array)

    Returns
    -------
    dict Updated keyword arguments

    ToDo
    ----
    - Clean up wavelet_filters and l1norm

    """

    # Set the options for mr_transform (for sparsity)
    if kwargs['mode'] in ('all', 'sparse'):
        wavelet_opt = ['-t ' + kwargs['wavelet_type']]
        kwargs['wavelet_filters'] = get_mr_filters(data.shape[-2:],
                                                   wavelet_opt)
        kwargs['linear_l1norm'] = (data.shape[0] * np.sqrt(sum((np.sum(np.abs
                                   (filter_i)) ** 2 for filter_i in
                                   kwargs['wavelet_filters']))))

    # Set the linear operator
    if kwargs['mode'] == 'all':
        kwargs['linear_op'] = LinearCombo([WaveletConvolve(
                                          kwargs['wavelet_filters'],
                                          method=kwargs['convolve_method'],),
                                          Identity()])

    elif kwargs['mode'] in ('lowr', 'grad'):
        kwargs['linear_op'] = Identity()
        kwargs['linear_l1norm'] = 1.0

    elif kwargs['mode'] == 'sparse':
        kwargs['linear_op'] = WaveletConvolve(kwargs['wavelet_filters'],
                                              method=kwargs['convolve_method'],
                                              )

    return kwargs


def set_sparse_weights(data_shape, psf, **kwargs):
    """Set the sparsity weights

    This method defines the weights for thresholding in the sparse domain and
    add them to the keyword arguments. It additionally defines the shape of the
    dual variable.

    Parameters
    ----------
    data_shape : tuple
        Shape of the input data array
    psf : np.ndarray
        PSF data (2D or 3D array)

    Returns
    -------
    dict Updated keyword arguments

    """

    # Convolve the PSF with the wavelet filters
    if kwargs['psf_type'] == 'fixed':

        filter_conv = (filter_convolve(np.rot90(psf, 2),
                       kwargs['wavelet_filters'],
                       method=kwargs['convolve_method']))

        filter_norm = np.array([norm(a) * b * np.ones(data_shape[1:])
                                for a, b in zip(filter_conv,
                                kwargs['wave_thresh_factor'])])

        filter_norm = np.array([filter_norm for i in
                                range(data_shape[0])])

    else:

        filter_conv = (filter_convolve_stack(np.rot90(psf, 2),
                       kwargs['wavelet_filters'],
                       method=kwargs['convolve_method']))

        filter_norm = np.array([[norm(b) * c * np.ones(data_shape[1:])
                                for b, c in zip(a,
                                kwargs['wave_thresh_factor'])]
                                for a in filter_conv])

    # Define a reweighting instance
    kwargs['reweight'] = cwbReweight(kwargs['noise_est'] * filter_norm)

    # Set the shape of the dual variable
    dual_shape = ([kwargs['wavelet_filters'].shape[0]] + list(data_shape))
    dual_shape[0], dual_shape[1] = dual_shape[1], dual_shape[0]
    kwargs['dual_shape'] = dual_shape

    return kwargs


def set_condat_param(**kwargs):
    """Set the Condat-Vu parameters

    This method sets the values of tau and sigma in the Condat-Vu proximal-dual
    splitting algorithm if not already provided. It additionally checks that
    the combination of values will lead to convergence.

    Returns
    -------
    dict Updated keyword arguments

    """

    # Define a method for calculating sigma and/or tau
    def get_sig_tau():
        return 1.0 / (kwargs['grad_op'].spec_rad + kwargs['linear_l1norm'])

    # Calulate tau if not provided
    if isinstance(kwargs['condat_tau'], type(None)):
        kwargs['condat_tau'] = get_sig_tau()

    # Calculate sigma if not provided
    if isinstance(kwargs['condat_sigma'], type(None)):
        kwargs['condat_sigma'] = get_sig_tau()

    print(' - tau:', kwargs['condat_tau'])
    print(' - sigma:', kwargs['condat_sigma'])
    print(' - rho:', kwargs['relax'])
    if 'log' in kwargs:
        kwargs['log'].info(' - tau: ' + str(kwargs['condat_tau']))
        kwargs['log'].info(' - sigma: ' + str(kwargs['condat_sigma']))
        kwargs['log'].info(' - rho: ' + str(kwargs['relax']))

    # Test combination of sigma and tau
    sig_tau_test = (1.0 / kwargs['condat_tau'] - kwargs['condat_sigma'] *
                    kwargs['linear_l1norm'] ** 2 >=
                    kwargs['grad_op'].spec_rad / 2.0)

    print(' - 1/tau - sigma||L||^2 >= beta/2:', sig_tau_test)
    if 'log' in kwargs:
        kwargs['log'].info(' - 1/tau - sigma||L||^2 >= beta/2: ' +
                           str(sig_tau_test))

    return kwargs


def get_lambda(n_images, p_pixel, sigma, spec_rad):
    """Get lambda value

    This method calculates the singular value threshold for low-rank
    regularisation

    Parameters
    ----------
    n_images : int
        Total number of images
    p_pixel : int
        Total number of pixels
    sigma : float
        Noise standard deviation
    spec_rad : float
        The spectral radius of the gradient operator

    Returns
    -------
    float Lambda value

    """

    return sigma * np.sqrt(np.max([n_images + 1, p_pixel])) * spec_rad


def set_lowr_thresh(data_shape, **kwargs):
    """Set the low-rank threshold

    This method sets the value of the low-rank singular value threshold.

    Parameters
    ----------
    data_shape : tuple
        Shape of the input data array

    Returns
    -------
    dict Updated keyword arguments

    """

    if kwargs['lowr_type'] == 'standard':
        kwargs['lambda'] = (kwargs['lowr_thresh_factor'] *
                            get_lambda(data_shape[0], np.prod(data_shape[1:]),
                            kwargs['noise_est'], kwargs['grad_op'].spec_rad))

    elif kwargs['lowr_type'] == 'ngole':
        kwargs['lambda'] = (kwargs['lowr_thresh_factor'] * kwargs['noise_est'])

    print(' - lambda:', kwargs['lambda'])
    if 'log' in kwargs:
        kwargs['log'].info(' - lambda: ' + str(kwargs['lambda']))

    return kwargs


def set_primal_dual(data_shape, **kwargs):
    """Set primal and dual variables

    This method sets the initial values of the primal and dual variables

    Parameters
    ----------
    data_shape : tuple
        Shape of the input data array

    Returns
    -------
    dict Updated keyword arguments

    """

    # Set the initial values of the primal variable if not provided
    if isinstance(kwargs['primal'], type(None)):
        kwargs['primal'] = np.ones(data_shape)

    # Set the initial values of the dual variable
    if kwargs['mode'] == 'all':
        kwargs['dual'] = np.empty(2, dtype=np.ndarray)
        kwargs['dual'][0] = np.ones(kwargs['dual_shape'])
        kwargs['dual'][1] = np.ones(data_shape)

    elif kwargs['mode'] in ('lowr', 'grad'):
        kwargs['dual'] = np.ones(data_shape)

    elif kwargs['mode'] == 'sparse':
        kwargs['dual'] = np.ones(kwargs['dual_shape'])

    print(' - Primal Variable Shape:', kwargs['primal'].shape)
    print(' - Dual Variable Shape:', kwargs['dual'].shape)
    print(' ' + '-' * 70)
    if 'log' in kwargs:
        kwargs['log'].info(' - Primal Variable Shape: ' +
                           str(kwargs['primal'].shape))
        kwargs['log'].info(' - Dual Variable Shape: ' +
                           str(kwargs['dual'].shape))

    return kwargs


def set_prox_op_and_cost(data, **kwargs):
    """Set the proximity operators and cost function

    This method sets the proximity operators and cost function instances.

    Parameters
    ----------
    data : np.ndarray
        Input noisy data (3D array)

    Returns
    -------
    dict Updated keyword arguments

    """

    # Create a list of proximity operators
    kwargs['prox_op'] = []

    # Set the first operator as positivity contraint or simply identity
    if not kwargs['no_pos']:
        kwargs['prox_op'].append(Positivity())

    else:
        kwargs['prox_op'].append(IdentityProx())

    # Add a second proximity operator
    if kwargs['mode'] == 'all':

        kwargs['prox_op'].append(ProximityCombo(
                                 [SparseThreshold(
                                  kwargs['linear_op'].operators[0],
                                  kwargs['reweight'].weights,),
                                  LowRankMatrix(kwargs['lambda'],
                                  thresh_type=kwargs['lowr_thresh_type'],
                                  lowr_type=kwargs['lowr_type'],
                                  operator=kwargs['grad_op'].trans_op)]))

    elif kwargs['mode'] == 'lowr':

        kwargs['prox_op'].append(LowRankMatrix(kwargs['lambda'],
                                 thresh_type=kwargs['lowr_thresh_type'],
                                 lowr_type=kwargs['lowr_type'],
                                 operator=kwargs['grad_op'].trans_op))

        operator_list = [kwargs['grad_op']] + kwargs['prox_op']

    elif kwargs['mode'] == 'sparse':

        kwargs['prox_op'].append(SparseThreshold(kwargs['linear_op'],
                                 kwargs['reweight'].weights))

    elif kwargs['mode'] == 'grad':

        kwargs['prox_op'].append(IdentityProx())

    # Set the cost function
    kwargs['cost_op'] = (costObj([kwargs['grad_op']] + kwargs['prox_op'],
                         tolerance=kwargs['convergence'],
                         cost_interval=kwargs['cost_window'],
                         plot_output=kwargs['output'],
                         verbose=not kwargs['quiet']))

    return kwargs


def set_optimisation(**kwargs):
    """Set the optimisation technique

    This method sets the technique used for opttimising the problem

    Returns
    -------
    dict Updated keyword arguments

    """

    # Initalise an optimisation instance
    if kwargs['opt_type'] == 'fwbw':
        kwargs['optimisation'] = (ForwardBackward(kwargs['primal'],
                                  kwargs['grad_op'], kwargs['prox_op'][1],
                                  kwargs['cost_op'], auto_iterate=False))

    elif kwargs['opt_type'] == 'condat':
        kwargs['optimisation'] = (Condat(kwargs['primal'], kwargs['dual'],
                                  kwargs['grad_op'], kwargs['prox_op'][0],
                                  kwargs['prox_op'][1], kwargs['linear_op'],
                                  kwargs['cost_op'], rho=kwargs['relax'],
                                  sigma=kwargs['condat_sigma'],
                                  tau=kwargs['condat_tau'],
                                  auto_iterate=False))

    elif kwargs['opt_type'] == 'gfwbw':
        kwargs['optimisation'] = (GenForwardBackward(kwargs['primal'],
                                  kwargs['grad_op'], kwargs['prox_op'],
                                  lambda_init=1.0, cost=kwargs['cost_op'],
                                  weights=[0.1, 0.9],
                                  auto_iterate=False))

    return kwargs


def perform_reweighting(**kwargs):
    """Perform reweighting

    This method updates the weights used for thresholding in the sparse domain

    Returns
    -------
    dict Updated keyword arguments

    """

    # Loop through number of reweightings
    for i in range(kwargs['n_reweights']):

        print(' - REWEIGHT:', i + 1)
        print('')

        # Generate the new weights following reweighting persctiption
        kwargs['reweight'].reweight(kwargs['linear_op'].op(
                                    kwargs['optimisation'].x_final))

        # Perform optimisation with new weights
        kwargs['optimisation'].iterate(max_iter=kwargs['n_iter'])

        print('')


def run(data, psf, **kwargs):
    """Run deconvolution

    This method initialises the operator classes and runs the optimisation
    algorithm

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D images
    psf : np.ndarray
        Input PSF array, a single 2D PSF or an array of 2D PSFs

    Returns
    -------
    np.ndarray decconvolved data

    """

    # SET THE NOISE ESTIMATE
    kwargs = set_noise(data, **kwargs)

    # SET THE GRADIENT OPERATOR
    kwargs = set_grad_op(data, psf, **kwargs)

    # SET THE LINEAR OPERATOR
    kwargs = set_linear_op(data, **kwargs)

    # SET THE WEIGHTS IN THE SPARSE DOMAIN
    if kwargs['mode'] in ('all', 'sparse'):
        kwargs = set_sparse_weights(data.shape, psf, **kwargs)

    # SET THE CONDAT-VU PARAMETERS
    if kwargs['opt_type'] == 'condat':
        kwargs = set_condat_param(**kwargs)

    # SET THE LOW-RANK THRESHOLD
    if kwargs['mode'] in ('all', 'lowr'):
        kwargs = set_lowr_thresh(data.shape, **kwargs)

    # SET THE INITIAL PRIMAL AND DUAL VARIABLES
    kwargs = set_primal_dual(data.shape, **kwargs)

    # SET THE PROXIMITY OPERATORS AND THE COST FUNCTION
    kwargs = set_prox_op_and_cost(data, **kwargs)

    # SET THE OPTIMISATION METHOD
    kwargs = set_optimisation(**kwargs)

    # PERFORM OPTIMISATION
    kwargs['optimisation'].iterate(max_iter=kwargs['n_iter'])

    # PERFORM REWEIGHTING FOR SPARSITY
    if kwargs['mode'] in ('all', 'sparse'):
        perform_reweighting(**kwargs)

    # PLOT THE COST FUNCTION
    if not kwargs['no_plots']:
        kwargs['cost_op'].plot_cost()

    # FINISH AND RETURN RESULTS
    if 'log' in kwargs:
        kwargs['log'].info(' - Final iteration number: ' +
                           str(kwargs['cost_op']._iteration))
        kwargs['log'].info(' - Final log10 cost value: ' +
                           str(np.log10(kwargs['cost_op'].cost)))
        kwargs['log'].info(' - Converged: ' +
                           str(kwargs['optimisation'].converge))

    primal_res = kwargs['optimisation'].x_final

    if kwargs['opt_type'] == 'condat':
        dual_res = kwargs['optimisation'].y_final
    else:
        dual_res = None

    if kwargs['grad_type'] == 'psf_unknown':
        psf_res = kwargs['grad_op']._psf
    else:
        psf_res = None

    return primal_res, dual_res, psf_res
