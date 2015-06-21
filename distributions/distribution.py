#!/usr/bin/env python3
#title          : distribution.py
#description    : Imports distributions and defines their default settings.
#author         : Enys Mones
#date           : 2015.06.20
#version        : 0.1
#usage          : python distribution.py
#==============================================================================
import numpy as np
from core import core as co
from distributions.poisson import poisson
from distributions.exponential import exponential
from distributions.lognormal import lognormal
from distributions.weibull import weibull
from distributions.truncated_power_law import truncated_power_law
from distributions.shifted_power_law import shifted_power_law


# Distribution names
DISTRIBUTION_POISSON = 'poisson'
DISTRIBUTION_EXPONENTIAL = 'exponential'
DISTRIBUTION_LOGNORMAL = 'lognormal'
DISTRIBUTION_WEIBULL = 'weibull'
DISTRIBUTION_SHIFTED_POWER_LAW = 'shifted-power-law'
DISTRIBUTION_TRUNCATED_POWER_LAW = 'truncated-power-law'

# List containing all distributions
DISTRIBUTIONS = [
    DISTRIBUTION_POISSON,
    DISTRIBUTION_EXPONENTIAL,
    DISTRIBUTION_SHIFTED_POWER_LAW,
    DISTRIBUTION_TRUNCATED_POWER_LAW,
    DISTRIBUTION_LOGNORMAL,
    DISTRIBUTION_WEIBULL
]


def get_sample_pmf(samples):
    """
    Creates the probability mass function from a sample of values.

    :param samples: sample of values.
    :return: probability mass function
    """
    _sample_pmf = [0] * int(np.max(samples)+1)
    for _s in samples:
        _sample_pmf[int(_s)] += 1
    return np.array(_sample_pmf)/len(samples)


def get_sample_cdf(samples):
    """
    Creates the cumulative distribution from a sample of values.

    :param samples: sample of values.
    :return: cumulative distribution.
    """
    return np.cumsum(get_sample_pmf(samples))


def pmf(distribution, params, domain=co.DEFAULT_PDF_MAX):
    """
    Returns the probability mass function for the given distribution.

    :param distribution: distribution to use.
    :param params: parameters.
    :param domain: domain size.
    :return: probability mass function.
    """
    if distribution == DISTRIBUTION_POISSON:
        return poisson.pmf(params, domain=domain)
    if distribution == DISTRIBUTION_EXPONENTIAL:
        return exponential.pmf(params, domain=domain)
    if distribution == DISTRIBUTION_LOGNORMAL:
        return lognormal.pmf(params, domain=domain)
    if distribution == DISTRIBUTION_WEIBULL:
        return weibull.pmf(params, domain=domain)
    if distribution == DISTRIBUTION_TRUNCATED_POWER_LAW:
        return truncated_power_law.pmf(params, domain=domain)
    if distribution == DISTRIBUTION_SHIFTED_POWER_LAW:
        return shifted_power_law.pmf(params, domain=domain)


def cdf(distribution, params, domain=co.DEFAULT_PDF_MAX):
    """
    Returns the cumulative distribution function of a given distribution.

    :param distribution: distribution to use.
    :param params: parameters.
    :param domain: domain size.
    :return: cumulative distribution function.
    """
    return np.cumsum(pmf(distribution, params, domain=domain))


def sample(distribution, params, size=co.DEFAULT_SAMPLE_SIZE):
    """
    Returns samples from a given distribution.

    :param distribution: distribution to use.
    :param params: parameters.
    :param size: sample size
    :return: numpy array of samples.
    """
    if distribution == DISTRIBUTION_POISSON:
        return poisson.samples(params, size=size)
    if distribution == DISTRIBUTION_EXPONENTIAL:
        return exponential.samples(params, size=size)
    if distribution == DISTRIBUTION_LOGNORMAL:
        return lognormal.samples(params, size=size)
    if distribution == DISTRIBUTION_WEIBULL:
        return weibull.samples(params, size=size)
    if distribution == DISTRIBUTION_TRUNCATED_POWER_LAW:
        return truncated_power_law.samples(params, size=size)
    if distribution == DISTRIBUTION_SHIFTED_POWER_LAW:
        return shifted_power_law.samples(params, size=size)


def log_likelihood(distribution, params, data, nonzero_only=False):
    """
    Returns the log-likelihood of a distribution over a given sample.

    :param distribution: distribution to use.
    :param params: parameters.
    :param data: data to use.
    :param nonzero_only: whether only non-zero data points should be used.
    :return: log-likelihood.
    """
    if distribution == DISTRIBUTION_POISSON:
        return poisson.log_likelihood(params, data)
    if distribution == DISTRIBUTION_EXPONENTIAL:
        return exponential.log_likelihood(params, data, nonzero_only)
    if distribution == DISTRIBUTION_LOGNORMAL:
        return lognormal.log_likelihood(params, data)
    if distribution == DISTRIBUTION_WEIBULL:
        return weibull.log_likelihood(params, data)
    if distribution == DISTRIBUTION_TRUNCATED_POWER_LAW:
        return truncated_power_law.log_likelihood(params, data)
    if distribution == DISTRIBUTION_SHIFTED_POWER_LAW:
        return shifted_power_law.log_likelihood(params, data, nonzero_only)


def get_params(params, distribution):
    """
    Creates a printable message of the parameter values.

    :param params: list containing the parameters.
    :param distribution: distribution to use.
    :return: printable string of the parameter values.
    """
    if distribution == DISTRIBUTION_POISSON:
        return poisson.get_params(params)
    if distribution == DISTRIBUTION_EXPONENTIAL:
        return exponential.get_params(params)
    if distribution == DISTRIBUTION_LOGNORMAL:
        return lognormal.get_params(params)
    if distribution == DISTRIBUTION_WEIBULL:
        return weibull.get_params(params)
    if distribution == DISTRIBUTION_TRUNCATED_POWER_LAW:
        return truncated_power_law.get_params(params)
    if distribution == DISTRIBUTION_SHIFTED_POWER_LAW:
        return shifted_power_law.get_params(params)