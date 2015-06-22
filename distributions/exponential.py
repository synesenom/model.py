#!/usr/bin/env python3
#title          : exponential.py
#description    : Discrete exponential distribution.
#author         : Enys Mones
#date           : 2015.06.19
#version        : 0.1
#usage          : python exponential.py
#=====================================================
import numpy as np
from mpmath import exp, ln

from core import core as co


class Exponential(co.RealDistribution):
    """
    Discrete exponential distribution:

    Exponential(x) = C * exp(-x/beta),

    where C = (1-exp(-1/beta)).
    If the scale parameter is very small, a delta distribution is used.
    """

    @staticmethod
    def pmf(params, domain=co.DEFAULT_PDF_MAX):
        """
        Probability mass function.

        :param params: single element list containing the scale (beta) parameter.
        :param domain: domain size.
        :return: probability mass function.
        """
        if params[0] < co.EPSILON:
            return co.delta.pmf([0], domain)
        else:
            c = 1-exp(-1/params[0])
            x = np.arange(0, domain+1)
            return np.exp(-x/params[0])*c

    @staticmethod
    def samples(params, size=co.DEFAULT_SAMPLE_SIZE, domain=co.DEFAULT_SAMPLE_MAX):
        """
        Returns samples with discrete exponential distribution.

        :param params: single element list containing the scale (beta) parameter.
        :param size: number of samples.
        :param domain: domain size.
        :return: numpy array of samples.
        """
        if params[0] < co.EPSILON:
            return co.delta.samples([0], size)
        else:
            x = np.arange(0, domain+1)
            return co.generate_discrete_samples(x, np.exp(-x/params[0]), size)

    @staticmethod
    def log_likelihood(params, data, nonzero_only=False):
        """
        Calculates the log-likelihood on the data.

        :param params: single element list containing the scale (beta) parameter.
        :param data: input data as a numpy array.
        :param nonzero_only: whether nonzero element should be considered only.  This is
        used after determining the parameters and comparing to distributions that ignore
        zero values.
        :return: log-likelihood.
        """
        if params[0] < co.EPSILON:
            return co.delta.log_likelihood([0], data)
        else:
            if nonzero_only:
                _samples = data[np.where(data > 0)]
            else:
                _samples = data
            return len(_samples)*ln(1-exp(-1/params[0])) - np.sum(_samples)/params[0]

    @staticmethod
    def get_params(params):
        return "beta = %.5f" % params[0]
exponential = Exponential()