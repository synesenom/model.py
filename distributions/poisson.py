#!/usr/bin/env python3
#title          : poisson.py
#description    : Poisson distribution.
#author         : Enys Mones
#date           : 2015.06.19
#version        : 0.1
#usage          : python poisson.py
#=====================================================
import numpy as np
from scipy import stats
from mpmath import ln

from core import core as co


class Poisson(co.RealDistribution):
    """
    Poisson distribution:

    Poisson(x) = lambda^x * exp(-lambda) / x!.

    If lambda is too small, it is replaced by a delta function.
    """

    @staticmethod
    def pmf(params, domain=co.DEFAULT_PDF_MAX):
        """
        Probability mass function.

        :param params: a one element list containing the shape (lambda) parameter.
        :param domain: maximum of the domain.
        :return: probability mass function.
        """
        if params[0] < co.EPSILON:
            return co.delta.pmf([0], domain)
        else:
            return stats.poisson.pmf(np.arange(0, domain+1), params[0])

    @staticmethod
    def samples(params, size=co.DEFAULT_SAMPLE_SIZE, domain=co.DEFAULT_SAMPLE_MAX):
        """
        Returns samples with Poisson distribution.

        :param params: a one element list containing the shape (lambda) parameter.
        :param size: number of samples.
        :param domain: unused.
        :return: samples.
        """
        if params[0] < co.EPSILON:
            return co.delta.samples([0], size)
        else:
            return np.random.poisson(params[0], size)

    @staticmethod
    def log_likelihood(params, data, nonzero=False):
        """
        Calculates the log-likelihood on the data.

        :param params: a one element list containing the shape (lambda) parameter.
        :param data: input data as a numpy array.
        :param nonzero: unused.
        :return: log-likelihood.
        """
        _nonzero_samples = data[np.where(data > 0)]
        if params[0] < co.EPSILON:
            return co.delta.log_likelihood([0], data)
        else:
            return np.sum(data)*ln(params[0])\
                - len(data)*params[0]\
                - len(data) - np.sum(_nonzero_samples*np.log(_nonzero_samples) - _nonzero_samples)

    @staticmethod
    def get_params(params):
        return "lambda = %.5f" % params[0]
poisson = Poisson()