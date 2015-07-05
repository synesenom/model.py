#!/usr/bin/env python3
#title          : normal.py
#description    : Normal (Gaussian) distribution.
#author         : Enys Mones
#date           : 2015.06.21
#version        : 0.1
#usage          : python normal.py
#===================================================================
import numpy as np
from mpmath import ln

from core import core as co


class Normal(co.RealDistribution):
    """
    Normal distribution:

    Normal(x) = exp(-(x-mu)^2 / (2*sigma^2)) / (sigma*sqrt(2*pi)).

    If the location (mu) parameter is too small, a delta function is used. This is only to
    avoid negative means.
    If the shape (sigma) parameter is too small, a delta function is used to avoid
    negative variances.
    """

    @staticmethod
    def pmf(params, domain=co.DEFAULT_PDF_MAX):
        """
        Probability mass function at integer values.

        :param params: two elements list with the location (mu) and shape (sigma)
        parameters.
        :param domain: domain size.
        :return: probability mass function.
        """
        if params[0] < co.EPSILON:
            return co.delta.pmf([0], domain)
        elif params[1] < co.EPSILON:
            return co.delta.pmf([params[0]], domain)
        else:
            x = np.arange(0, domain+1)
            _pmf = np.exp(-0.5*np.power((x-params[0])/params[1], 2))
            return _pmf/np.sum(_pmf)

    @staticmethod
    def samples(params, size=co.DEFAULT_SAMPLE_SIZE, domain=co.DEFAULT_SAMPLE_MAX):
        """
        Returns samples with discrete normal distribution.

        :param params: two elements list with the location (mu) and shape (sigma)
        parameters.
        :param size: number of samples.
        :param domain: unused.
        :return: numpy array of samples.
        """
        if params[0] < co.EPSILON:
            return co.delta.samples([0], size, domain)
        elif params[1] < co.EPSILON:
            return co.delta.samples([params[0]], domain)
        else:
            x = np.arange(0, domain+1)
            p = np.exp(-0.5*np.power((x-params[0])/params[1], 2))
            return co.generate_discrete_samples(x, p, size)

    @staticmethod
    def log_likelihood(params, data, nonzero_only=False):
        """
        Calculates the log-likelihood on the data.

        :param params: two elements list with the location (mu) and shape (sigma)
        parameters.
        :param data: input data as a numpy array.
        :param nonzero_only: whether nonzero element should be considered only.  This is
        used after determining the parameters and comparing to distributions that ignore
        zero values.
        :return: log-likelihood.
        """
        if params[0] < co.EPSILON:
            return co.delta.log_likelihood([0], data)
        elif params[1] < co.EPSILON:
            return co.delta.log_likelihood([params[0]], data)
        else:
            if nonzero_only:
                _samples = data[np.where(data > 0)]
            else:
                _samples = data
            x = np.arange(0, co.DEFAULT_PDF_MAX+1)
            c = np.sum(np.exp(-0.5*np.power((x-params[0])/params[1], 2)))
            return - np.sum(np.power(_samples-params[0], 2))/(2*params[1]**2)\
                - len(_samples)*ln(c)

    @staticmethod
    def get_params(params):
        return "(mu, sigma) = (%.5f, %.5f)" % (params[0], params[1])
normal = Normal()