#!/usr/bin/env python3
#title          : meaures.py
#description    : Contains various measures calculated over samples.
#author         : Enys Mones
#date           : 2015.06.21
#version        : 0.1
#usage          : python measures.py
#=====================================================================
import numpy as np
from mpmath import ln


def ks_statistics(data_cdf_, model_cdf_):
    """
    Calculates the Kolmogorov-Smirnov D statistics for two cumulative core.

    :param data_cdf_: cdf of the data.
    :param model_cdf_: cdf of the model.
    :return: K-S D statistics.
    """
    _size_diff = len(data_cdf_) - len(model_cdf_)
    if _size_diff > 0:
        return np.max(np.abs(data_cdf_ - np.append(model_cdf_, np.ones(_size_diff))))
    if _size_diff < 0:
        return np.max(np.abs(np.append(data_cdf_, np.ones(-_size_diff)) - model_cdf_))
    if _size_diff == 0:
        return np.max(np.abs(data_cdf_ - model_cdf_))


def aic_measure(log_likelihood, params_num):
    """
    Returns the Akaike information criterion value, that is
    AIC = -2ln(L) + 2m,
    where L is the likelihood in the optimum and m is the number of parameters in the
    model.

    :param log_likelihood: optimized log-likelihood.
    :param params_num: number of model parameters.
    :return: AIC value.
    """
    return -2*(log_likelihood - params_num)


def bic_measure(log_likelihood, params_num, sample_size):
    """
    Returns the Bayesian information criterion value, that is
    BIC = -2ln(L) + m*ln(n),
    where L is the likelihood in the optimum, m and n are the number of model parameters
    and the sample size.

    :param log_likelihood: optimized log-likelihood.
    :param params_num: number of model parameters.
    :param sample_size: sample size.
    :return: BIC value.
    """
    return -2*log_likelihood + params_num*ln(sample_size)