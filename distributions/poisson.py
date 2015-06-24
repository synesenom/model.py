#!/usr/bin/env python3
#title          : poisson.py
#description    : Poisson distribution.
#author         : Enys Mones
#date           : 2015.06.19
#version        : 0.1
#usage          : python poisson.py
#=====================================================
import numpy as np
from scipy import special as sp
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
    def log_likelihood(params, data, nonzero_only=False):
        """
        Calculates the log-likelihood on the data.
        The factorial is approximated by using Stirling's formula.

        :param params: a one element list containing the shape (lambda) parameter.
        :param data: input data as a numpy array.
        :param nonzero_only: whether nonzero element should be considered only.  This is
        used after determining the parameters and comparing to distributions that ignore
        zero values.
        :return: log-likelihood.
        """
        nonzero_samples = data[np.where(data > 0)]
        if params[0] < co.EPSILON:
            return co.delta.log_likelihood([0], data)
        else:
            if nonzero_only:
                _sampes = data[np.where(data > 0)]
            else:
                _sampes = data
            return np.sum(_sampes)*ln(params[0])\
                - len(_sampes)*params[0]\
                - 0.5*len(_sampes)*ln(2*np.pi) - np.sum((0.5+nonzero_samples)*np.log(nonzero_samples)-nonzero_samples)\
                - np.sum(np.log(1+1/(12*nonzero_samples)+1/(288*np.power(nonzero_samples, 2))))

    @staticmethod
    def get_params(params):
        return "lambda = %.5f" % params[0]
poisson = Poisson()