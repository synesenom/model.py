#!/usr/bin/env python3
#title          : truncated_power_law.py
#description    : Truncated power-law, i.e., a power-law with cutoff
#author         : Enys Mones
#date           : 2015.06.21
#version        : 0.1
#usage          : python truncated_power_law.py
#=====================================================================
import numpy as np
from mpmath import ln, exp, polylog

from core import core as co


class TruncatedPowerLaw(co.RealDistribution):
    """
    Power-law with cutoff:

    TruncatedPowerLaw(x) = C * x^(-gamma) * exp(-x/kappa),

    where C = Li_gamma(exp(-x/kappa)),  with Li_g(x) being the polylogarithm with index g
    If  the cutoff  is very  small or  the  exponent is  too large,  the distribution  is
    approximated by a delta function.  If the exponent is very small, the distribution is
    approximated by a uniform distribution.
    """

    @staticmethod
    def pmf(params, domain=co.DEFAULT_PDF_MAX):
        """
        Probability mass function.

        :param params: two elements list containing the exponent (gamma) and cutoff
        (kappa).
        :param domain: domain size.
        :return: probability mass function.
        """
        if params[0] < co.EPSILON:
            return co.uniform.pmf(None, domain)
        elif params[1] < co.EPSILON:
            return co.delta.pmf([1], domain)
        else:
            c = polylog(params[0], exp(-1/params[1]))
            if c < co.EPSILON:
                return co.delta.pmf([1], domain)
            else:
                x = np.arange(1, domain+1)
                return np.append([0.0], np.power(x, -params[0])*np.exp(-x/params[1])/c)

    @staticmethod
    def samples(params, size=co.DEFAULT_SAMPLE_SIZE, domain=co.DEFAULT_SAMPLE_MAX):
        """
        Returns samples with discrete truncated power-law.

        :param params: two elements list containing the exponent (gamma) and cutoff
        (kappa).
        :param size: number of samples.
        :param domain: domain size.
        :return: numpy array of samples.
        """
        if params[0] < co.EPSILON:
            return co.uniform.samples(None, size)
        elif params[1] < co.EPSILON:
            return co.delta.samples([1], size)
        else:
            if polylog(params[0], exp(-1/params[1])) < co.EPSILON:
                return co.delta.samples([1], size)
            else:
                x = np.arange(1, domain+1)
                return co.generate_discrete_samples(x, np.power(x, -params[0])*np.exp(-x/params[1]), size)

    @staticmethod
    def log_likelihood(params, data, nonzero=False):
        """
        Calculates the log-likelihood of the discrete truncated power-law on the data.

        :param params: two elements list containing the exponent (gamma) and cutoff
        (kappa).
        :param data: input data as a numpy array.
        :param nonzero: unused.
        :return: log-likelihood.
        """
        nonzero_samples = data[np.where(data > 0)]
        if params[0] < co.EPSILON:
            return co.uniform.log_likelihood(None, data)
        elif params[1] < co.EPSILON:
            return co.delta.log_likelihood([1], data)
        else:
            c = polylog(params[0], exp(-1/params[1]))
            if c < co.EPSILON:
                return co.delta.log_likelihood([1], data)
            else:
                return -params[0]*np.sum(np.log(nonzero_samples))\
                       - np.sum(data)/params[1]\
                       - len(data)*ln(c)

    @staticmethod
    def get_params(params):
        return "(gamma, kappa) = (%.5f, %.5f)" % (params[0], params[1])
truncated_power_law = TruncatedPowerLaw()
