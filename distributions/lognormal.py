#!/usr/bin/env python3
#title          : lognormal.py
#description    : Log-normal distribution.
#author         : Enys Mones
#date           : 2015.06.21
#version        : 0.1
#usage          : python lognormal.py
#===================================================================
import numpy as np
from mpmath import exp, ln, sqrt

from core import core as co


class Lognormal(co.RealDistribution):
    """
    Log-normal distribution:

    Lognormal(x) = C * exp(-(ln(x)-mu)^2 / (2*sigma^2)) / x,

    where C = 1/(sigma*sqrt(2*pi)).
    If the shape parameter is very small, a delta distribution is used.
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
        if params[1] < co.EPSILON:
            return co.delta.pmf([exp(params[0])], domain)
        else:
            x = np.arange(1, domain+1)
            _pmf = np.append([0.0], np.exp(-0.5*np.power((np.log(x)-params[0])/params[1], 2))/x)
            return _pmf/np.sum(_pmf)

    @staticmethod
    def samples(params, size=co.DEFAULT_SAMPLE_SIZE, domain=co.DEFAULT_SAMPLE_MAX):
        """
        Returns samples with discrete log-normal distribution.

        :param params: two elements list with the location (mu) and shape (sigma)
        parameters.
        :param size: number of samples.
        :param domain: unused.
        :return: numpy array of samples.
        """
        #return np.random.lognormal(params[0], params[1], size)  # FIXME continuous sampling
        if params[1] < co.EPSILON:
            return co.delta.samples([exp(params[0])], size)
        else:
            x = np.arange(1, domain+1)
            p = np.exp(-0.5*np.power((np.log(x)-params[0])/params[1], 2))/x
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
        """
        if nonzero_only:
            nonzero_samples = np.where(data > 0, data, co.EPSILON)
        else:
            nonzero_samples = data[np.where(data > 0)]
        if params[0] < 0:
            return co.delta.log_likelihood(0, data)
        if params[1] < co.EPSILON:
            return co.delta.log_likelihood(exp(params[0]), data)
        else:
            return -np.sum(np.log(nonzero_samples))\
                - np.sum(np.power(np.log(nonzero_samples)-params[0], 2))/(2*params[1]**2)\
                - len(data)*ln(params[1]*sqrt(2*np.pi))
        """  # FIXME continuous log-likelihood
        nonzero_samples = data[np.where(data > 0)]
        if params[0] < co.EPSILON or params[1] < co.EPSILON:
            return co.delta.log_likelihood([0], data)
        else:
            x = np.arange(1, co.DEFAULT_PDF_MAX+1)
            c = np.sum(np.exp(-0.5*np.power((np.log(x)-params[0])/params[1], 2))/x)
            return -np.sum(np.log(nonzero_samples))\
                - np.sum(np.power(np.log(nonzero_samples)-params[0], 2))/(2*params[1]**2)\
                - len(data)*ln(c)

    @staticmethod
    def get_params(params):
        return "(mu, sigma) = (%.5f, %.5f)" % (params[0], params[1])
lognormal = Lognormal()