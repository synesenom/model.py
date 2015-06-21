#!/usr/bin/env python3
#title          : fit.py
#description    : Contains fitting methods.
#author         : Enys Mones
#date           : 2015.06.21
#version        : 0.1
#usage          : python fit.py
#===============================================
from distributions import distribution as dist
from calculation.measures import ks_statistics
from scipy import optimize as op


# Initial values of parameters for fitting
INITIAL_FITTING_PARAMS = {
    dist.DISTRIBUTION_POISSON: [10.0],
    dist.DISTRIBUTION_EXPONENTIAL: [10.0],
    dist.DISTRIBUTION_LOGNORMAL: [1.0, 0.5],
    dist.DISTRIBUTION_WEIBULL: [3.2, 0.8],
    dist.DISTRIBUTION_TRUNCATED_POWER_LAW: [1.2, 50.0],
    dist.DISTRIBUTION_SHIFTED_POWER_LAW: [1.2, 1.0]
}


def fit_mle(distribution, data):
    """
    Fits a given distribution on the data using maximum likelihood estimation.

    :param distribution: distribution to fit.
    :param data: data to use.
    :return: fit results in a dictionary containing:
        parameter values
        log-likelihood
        K-S statistics.
    """
    params = INITIAL_FITTING_PARAMS[distribution]
    nll = lambda x: -dist.log_likelihood(distribution, x, data)
    res = op.minimize(nll, params, method='nelder-mead')
    return {'params': res.x,
            'log-likelihood': float(-res.fun),
            'D': float(ks_statistics(dist.get_sample_cdf(data), dist.cdf(distribution, res.x)))
            }


def fit_ks(distribution, data):
    """
    Fits a given distribution on the data using K-S goodness-of-fit optimization.

    :param distribution: distribution to fit.
    :param data: data to use.
    :return: fit results in a dictionary containing:
        parameter values
        log-likelihood
        K-S statistics.
    """
    params = INITIAL_FITTING_PARAMS[distribution]
    data_max = int(max(data))
    sample_cdf = dist.get_sample_cdf(data)
    ksd = lambda x: ks_statistics(sample_cdf, dist.cdf(distribution, x, domain=data_max))
    res = op.minimize(ksd, params, method='Nelder-Mead')
    return {'params': res.x,
            'log-likelihood': float(dist.log_likelihood(distribution, res.x, data)),
            'D': float(res.fun)
            }