#!/usr/bin/env python3
#title          : weibull.py
#description    : Weibull distribution.
#author         : Enys Mones
#date           : 2015.06.21
#version        : 0.1
#usage          : python weibull.py
#===================================================================
import numpy as np
from mpmath import ln

from core import core as co


class Weibull(co.RealDistribution):
    """
    Weibull distribution.
    If the shape or scale parameters are very small, a delta distribution is used.
    """

    @staticmethod
    def pmf(params, domain=co.DEFAULT_PDF_MAX):
        """
        Probability mass function of the Weibull distribution taken at integer values.

        :param params: two elements list containing the shape (k) and scale (lambda) parameters.
        :param domain: domain size.
        :return: probability mass function.
        """
        if params[0] < co.EPSILON or params[1] < co.EPSILON:
            return co.delta.pmf([0], domain)
        else:
            if 0 <= params[0] - 1 < co.EPSILON:
                x = np.arange(1, domain+1)
                _pmf = np.append([1/params[1]], np.power(x, params[0]-1)*np.exp(-np.power(x/params[1], params[0])))
            else:
                x = np.arange(1, domain+1)
                _pmf = np.append([0.0], np.power(x, params[0]-1)*np.exp(-np.power(x/params[1], params[0])))
            return _pmf/np.sum(_pmf)

    @staticmethod
    def samples(params, size=co.DEFAULT_SAMPLE_SIZE, domain=co.DEFAULT_SAMPLE_MAX):
        """
        Returns samples with discrete Weibull distribution.

        :param params: two elements list containing the shape (k) and scale (lambda) parameters.
        :param size: number of samples.
        :param domain: domain size.
        :return: numpy array of samples.
        """
        if params[0] < co.EPSILON or params[1] < co.EPSILON:
            return co.delta.samples([0], size)
        else:
            x = np.arange(1, domain+1)
            p = np.power(x, params[0]-1)*np.exp(-np.power(x/params[1], params[0]))
            if 0 <= params[0] - 1 < co.EPSILON:
                return co.generate_discrete_samples(np.append([0], x), np.append([1/params[1]], p), size)
            else:
                return co.generate_discrete_samples(x, p, size)

    @staticmethod
    def log_likelihood(params, data):
        """
        Calculates the log-likelihood of the Weibull distribution on the data.

        :param params: two elements list containing the shape (k) and scale (lambda) parameters.
        :param data: input data as a numpy array.
        :return: log-likelihood.
        """
        nonzero_samples = data[np.where(data > 0)]
        if params[0] < co.EPSILON or params[1] < co.EPSILON:
            return co.delta.log_likelihood([0], data)
        else:
            x = np.arange(1, co.DEFAULT_PDF_MAX+1)
            c = np.sum(np.power(x, params[0]-1)*np.exp(-np.power(x/params[1], params[0])))
            return (params[0]-1) * np.sum(np.log(nonzero_samples))\
                - 1/params[1]**params[0] * np.sum(np.power(data, params[0]))\
                - len(data) * ln(c)

    @staticmethod
    def get_params(params):
        return "(k, lambda) = (%.5f, %.5f)" % (params[0], params[1])
weibull = Weibull()
