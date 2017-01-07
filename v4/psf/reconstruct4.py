
from scipy.linalg import norm

from operators import *
from algorithms import PowerMethod
from cost import *
from linear import *
from proximity import *
from optimisation import *
from reweight import cwbReweight
from wavelet import filter_convolve, filter_convolve_stack
from psf_gen import single_psf
from transform import cube2matrix


def rec(data, noise_est, layout, primal=None, psf=None, psf_type='fixed',
        psf_pcs=None, psf_coef=None, wavelet_levels=1, wavelet_opt=None,
        wave_thresh_factor=1, lowr_thresh_factor=1, lowr_thresh_type='soft',
        lowr_type='standard', n_reweights=0, n_iter=150, relax=0.5, mode='all',
        grad=True, pos=True, data_format='cube', opt_type='condat', log=None,
        liveplot=False):

    ######
    # SET THE GRADIENT OPERATOR

    if grad:
        if not isinstance(psf, type(None)):
                grad_op = StandardPSF(data, psf, psf_type=psf_type,
                                      data_format=data_format)

        else:
                grad_op = PixelVariantPSF(data, psf_pcs, psf_coef,
                                          data_format=data_format)

    else:
        grad_op = StandardPSFnoGrad(data, psf, psf_type=psf_type,
                                    data_format=data_format)

    grad_op.get_spec_rad(tolerance=1e-6, max_iter=10)
    print ' - Spetral Radius:', grad_op.spec_rad
    log.info(' - Spetral Radius: ' + str(grad_op.spec_rad))

    ######
    # SET THE LINEAR OPERATOR(S)

    if mode == 'all':
        linear_op = LinearCombo([Wavelet(data, wavelet_levels, wavelet_opt,
                                data_format=data_format), Identity()])
        wavelet_filters = linear_op.operators[0].filters
        linear_l1norm = linear_op.operators[0].l1norm

    elif mode in ('lowr', 'grad'):
        linear_op = Identity()
        linear_l1norm = linear_op.l1norm

    elif mode == 'wave':
        linear_op = Wavelet(data, wavelet_levels, wavelet_opt,
                            data_format=data_format)
        wavelet_filters = linear_op.filters
        linear_l1norm = linear_op.l1norm

    ######
    # ESTIMATE THE NOISE IN THE WAVELET DOMAIN

    if wave_thresh_factor.size == 1:
        wave_thresh_factor = np.repeat(wave_thresh_factor, wavelet_levels)

    elif wave_thresh_factor.size != wavelet_levels:
        raise ValueError('The number of wavelet threshold factors does not ' +
                         'match the number of wavelet levels.')

    if mode in ('all', 'wave'):

        if not isinstance(psf, type(None)):

            if psf_type == 'fixed':

                filter_conv = (filter_convolve(np.rot90(psf, 2),
                               wavelet_filters))

                filter_norm = np.array([norm(a) * b * np.ones(data.shape[1:])
                                        for a, b in zip(filter_conv,
                                        wave_thresh_factor)])

                if data_format == 'cube':
                    filter_norm = np.array([filter_norm for i in
                                            xrange(data.shape[0])])

            else:

                filter_conv = (filter_convolve_stack(np.rot90(psf, 2),
                               wavelet_filters))

                filter_norm = np.array([[norm(b) * c * np.ones(data.shape[1:])
                                        for b, c in zip(a, wave_thresh_factor)]
                                        for a in filter_conv])

            weight_est = noise_est * filter_norm

        else:
            weight_est = np.sqrt(filter_convolve_stack(noise_est,
                                 wavelet_filters ** 2))

    ######
    # SET THE WEIGHTS
        rw = cwbReweight(weight_est)

    ######
    # SET THE SHAPE OF THE DUAL VARIABLE
        dual_shape = ([wavelet_filters.shape[0]] +
                      list(data.shape))

        if data_format == 'cube':
            dual_shape[0], dual_shape[1] = dual_shape[1], dual_shape[0]

    ######
    # SET RHO, SIGMA AND TAU FOR CONDAT

    if opt_type == 'condat':

        tau = 1.0 / (grad_op.spec_rad + linear_l1norm)
        sigma = tau
        rho = relax

        print ' - tau:', tau
        print ' - sigma:', sigma
        print ' - rho:', rho
        log.info(' - tau: ' + str(tau))
        log.info(' - sigma: ' + str(sigma))
        log.info(' - rho: ' + str(rho))

        print ' - 1/tau - sigma||L||^2 >= beta/2:', (1.0 / tau - sigma *
                                                     linear_l1norm ** 2 >=
                                                     grad_op.spec_rad / 2.0)

    ######
    # FIND LAMBDA FOR LOW-RANK.

    if mode in ('all', 'lowr'):

        lamb = lowr_thresh_factor * noise_est

        print ' - lambda:', lamb
        log.info(' - lambda: ' + str(lamb))

    ######
    # INITALISE THE PRIMAL AND DUAL VALUES

    # 1 Primal Operator (Positivity or Identity)
    if isinstance(primal, type(None)):
        primal = np.ones(data.shape)

    if mode == 'all':

        # 2 Dual Operators (Wavelet + Threshold and Identity + LowRankMatrix)
        dual = np.empty(2, dtype=np.ndarray)
        dual[0] = np.ones(dual_shape)
        dual[1] = np.ones(data.shape)

    elif mode in ('lowr', 'grad'):

        # 1 Dual Operator (Identity or LowRankMatrix)
        dual = np.ones(data.shape)

    elif mode == 'wave':

        # 1 Dual Operator (Wavelet Threshold)
        dual = np.ones(dual_shape)

    print ' - Primal Variable Shape:', primal.shape
    print ' - Dual Variable Shape:', dual.shape
    print ' ----------------------------------------'

    ######
    # GET THE INITIAL GRADIENT VALUE
    grad_op.get_grad(primal)

    ######
    # SET THE PROXIMITY OPERATORS

    # FOR GFWBW
    prox_list = []

    if pos:
        prox_op = Positive()
        prox_list.append(prox_op)

    else:
        prox_op = Identity()

    use_pos = False

    if opt_type == 'fwbw':
        use_pos = True

    total_n_iter = n_iter * (n_reweights + 1)
    cost_test_window = 2

    if mode == 'all':

        prox_dual_op = ProximityCombo([Threshold(rw.weights / sigma,
                                      positivity=use_pos), LowRankMatrix(lamb,
                                      data_format=data_format,
                                      thresh_type=lowr_thresh_type,
                                      lowr_type=lowr_type, layout=layout,
                                      operator=grad_op.MtX)])

        cost_op = costFunction(data, grad=grad_op,
                               wavelet=linear_op.operators[0],
                               weights=rw.weights,
                               lambda_reg=lamb,
                               mode=mode, data_format=data_format,
                               positivity=pos, live_plotting=liveplot,
                               window=cost_test_window, total_it=total_n_iter)

    elif mode == 'lowr':

        prox_dual_op = LowRankMatrix(lamb, data_format=data_format,
                                     thresh_type=lowr_thresh_type,
                                     lowr_type=lowr_type,
                                     layout=layout, operator=grad_op.MtX)

        prox_list.append(prox_dual_op)

        cost_op = costFunction(data, grad=grad_op, wavelet=None, weights=None,
                               lambda_reg=lamb, mode=mode,
                               data_format=data_format, positivity=pos,
                               live_plotting=liveplot, window=cost_test_window,
                               total_it=total_n_iter)

    elif mode == 'wave':

        prox_dual_op = Threshold(rw.weights / sigma, positivity=use_pos)

        cost_op = costFunction(data, grad=grad_op,
                               wavelet=linear_op,
                               weights=rw.weights,
                               lambda_reg=None,
                               mode=mode, data_format=data_format,
                               live_plotting=liveplot, window=cost_test_window,
                               total_it=total_n_iter)

    elif mode == 'grad':

        prox_dual_op = Identity()
        prox_list.append(prox_dual_op)

        cost_op = costFunction(data, grad=grad_op,
                               wavelet=None,
                               weights=None,
                               lambda_reg=None,
                               mode=mode, data_format=data_format)

    ######
    # PERFORM THE OPTIMISATION
    if opt_type == 'fwbw':
        opt = ForwardBackward(primal, grad_op, prox_dual_op, cost_op,
                              auto_iterate=False)

    elif opt_type == 'condat':
        opt = Condat(primal, dual, grad_op, prox_op, prox_dual_op, linear_op,
                     cost_op, rho=rho, sigma=sigma, tau=tau,
                     auto_iterate=False)

    elif opt_type == 'gfwbw':
        opt = GenForwardBackward(primal, grad_op, prox_list, lambda_init=1.0,
                                 cost=cost_op, weights=[0.1, 0.9],
                                 auto_iterate=False, plot=True)

    opt.iterate(max_iter=n_iter)

    print ''

    ######
    # REWEIGHTING

    if mode in ('all', 'wave'):

        for i in xrange(n_reweights):

            if not opt.converge:

                print ' Reweight:', i + 1
                print ''

                rw.reweight(linear_op.op(opt.x_new)[0])
                if mode == 'all':
                    prox_dual_op.operators[0].update_weights(rw.weights /
                                                             sigma)
                else:
                    prox_dual_op.update_weights(rw.weights / sigma)
                cost_op.update_weights(rw.weights)
                opt.iterate(max_iter=n_iter)
                print ''

    ######
    # FINISH AND RETURN RESULTS

    log.info(' - Final iteration number: ' + str(cost_op.iteration))
    log.info(' - Final log10 cost value: ' + str(np.log10(cost_op.cost)))
    log.info(' - Converged: ' + str(opt.converge))

    if opt_type == 'condat':
        return opt.x_final, opt.y_final

    else:
        return opt.x_final, None
