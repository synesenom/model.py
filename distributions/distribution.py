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
from distributions.normal import normal


# Distribution names
DISTRIBUTION_POISSON = 'poisson'
DISTRIBUTION_EXPONENTIAL = 'exponential'
DISTRIBUTION_LOGNORMAL = 'lognormal'
DISTRIBUTION_WEIBULL = 'weibull'
DISTRIBUTION_SHIFTED_POWER_LAW = 'shifted-power-law'
DISTRIBUTION_TRUNCATED_POWER_LAW = 'truncated-power-law'
DISTRIBUTION_NORMAL = 'normal'

KEY_CLASS = 'class'
KEY_TEST_PARAMS = 'test-params'
KEY_INITIAL_FIT_PARAMS = 'initial-fit-params'
DISTRIBUTIONS = {
    DISTRIBUTION_POISSON: {KEY_CLASS: poisson,
                           KEY_TEST_PARAMS: [3.4],
                           KEY_INITIAL_FIT_PARAMS: [20.0]},
    DISTRIBUTION_EXPONENTIAL: {KEY_CLASS: exponential,
                               KEY_TEST_PARAMS: [17.0],
                               KEY_INITIAL_FIT_PARAMS: [10.0]},
    DISTRIBUTION_SHIFTED_POWER_LAW: {KEY_CLASS: shifted_power_law,
                                     KEY_TEST_PARAMS: [2.3, 20.7],
                                     KEY_INITIAL_FIT_PARAMS: [1.2, 1.0]},
    DISTRIBUTION_TRUNCATED_POWER_LAW: {KEY_CLASS: truncated_power_law,
                                       KEY_TEST_PARAMS: [2.3, 123.0],
                                       KEY_INITIAL_FIT_PARAMS: [1.2, 50.0]},
    DISTRIBUTION_LOGNORMAL: {KEY_CLASS: lognormal,
                             KEY_TEST_PARAMS: [1.9, 1.1],
                             KEY_INITIAL_FIT_PARAMS: [1.0, 0.5]},
    DISTRIBUTION_WEIBULL: {KEY_CLASS: weibull,
                           KEY_TEST_PARAMS: [0.5, 1.2],
                           KEY_INITIAL_FIT_PARAMS: [3.2, 0.8]},
    DISTRIBUTION_NORMAL: {KEY_CLASS: normal,
                          KEY_TEST_PARAMS: [80.8, 8.9],
                          KEY_INITIAL_FIT_PARAMS: [10.0, 5.0]}
}


def get():
    """
    Simply returns a sorted list of the available distributions.

    :return: sorted list of available distributions.
    """
    return sorted(list(DISTRIBUTIONS.keys()))


def get_sample_pmf(values):
    """
    Creates the probability mass function from a sample of values.

    :param values: sample of values.
    :return: probability mass function as a numpy array.
    """
    return np.histogram(values.astype(int), range(int(np.max(values))))[0] / len(values)


def get_sample_cdf(values):
    """
    Creates the cumulative distribution from a sample of values.

    :param values: sample of values.
    :return: cumulative distribution.
    """
    return np.cumsum(get_sample_pmf(values))


def pmf(distribution, params, domain=co.DEFAULT_PDF_MAX):
    """
    Returns the probability mass function for the given distribution.

    :param distribution: distribution to use.
    :param params: parameters.
    :param domain: domain size.
    :return: probability mass function.
    """
    return DISTRIBUTIONS[distribution][KEY_CLASS].pmf(params, domain=domain)


def cdf(distribution, params, domain=co.DEFAULT_PDF_MAX):
    """
    Returns the cumulative distribution function of a given distribution.

    :param distribution: distribution to use.
    :param params: parameters.
    :param domain: domain size.
    :return: cumulative distribution function.
    """
    return np.cumsum(pmf(distribution, params, domain=domain))


def samples(distribution, params, size=co.DEFAULT_SAMPLE_SIZE):
    """
    Returns samples from a given distribution.

    :param distribution: distribution to use.
    :param params: parameters.
    :param size: sample size
    :return: numpy array of samples.
    """
    return DISTRIBUTIONS[distribution][KEY_CLASS].samples(params, size=size)


def log_likelihood(distribution, params, data, nonzero_only=False):
    """
    Returns the log-likelihood of a distribution over a given sample.

    :param distribution: distribution to use.
    :param params: parameters.
    :param data: data to use.
    :param nonzero_only: whether only non-zero data points should be used.
    :return: log-likelihood.
    """
    return DISTRIBUTIONS[distribution][KEY_CLASS].log_likelihood(params, data, nonzero_only)


def get_params(params, distribution):
    """
    Creates a printable message of the parameter values.

    :param params: list containing the parameters.
    :param distribution: distribution to use.
    :return: printable string of the parameter values.
    """
    return DISTRIBUTIONS[distribution][KEY_CLASS].get_params(params)
