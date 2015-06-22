#!/usr/bin/env python3
#title          : shifted_power_law.py
#description    : Shifted power-law.
#author         : Enys Mones
#date           : 2015.06.21
#version        : 0.1
#usage          : python shifted_power_law.py
#=====================================================================
import numpy as np
from mpmath import ln, zeta

from core import core as co


class ShiftedPowerLaw(co.RealDistribution):
    """
    Power-law with a shift in location:

    ShiftedPowerLaw(x) = C * (x + x0)^(-gamma),

    where C = zeta(gamma, x0), with zeta being the Hurwitz zeta function. If the exponent
    and shift are too small,  or the normalizing constant is too large,  the distribution
    is approximated by a delta function.  If the exponent is very small  but the shift is
    finite, the distribution is approximated by a uniform distribution.
    """

    @staticmethod
    def pmf(params, domain=co.DEFAULT_PDF_MAX):
        """
        Probability mass function.

        :param params: two elements list containing the exponent (gamma) and shift (x0).
        :param domain: domain size.
        :return: probability mass function.
        """
        if params[0] < co.EPSILON:
            if params[1] < co.EPSILON:
                return co.delta.pmf([0], domain)
            else:
                return co.uniform.pmf(None, domain)
        else:
            c = float(zeta(params[0], params[1]))
            if c < co.EPSILON:
                return co.delta.pmf([0], domain)
            else:
                return np.power(np.arange(0, domain+1)+params[1], -params[0])/c

    @staticmethod
    def samples(params, size=co.DEFAULT_SAMPLE_SIZE, domain=co.DEFAULT_SAMPLE_MAX):
        """
        Returns samples with discrete shifted power-law.

        :param params: two elements list containing the exponent (gamma) and shift (x0).
        :param size: number of samples.
        :param domain: domain size.
        :return: numpy array of samples.
        """
        if params[0] < co.EPSILON:
            if params[1] < co.EPSILON:
                return co.delta.samples([0], size)
            else:
                return co.uniform.samples(None, size)
        else:
            if float(zeta(params[0], params[1])) < co.EPSILON:
                return co.delta.samples([0], size)
            else:
                x = np.arange(0, co.DEFAULT_SAMPLE_MAX+1)
                return co.generate_discrete_samples(x, np.power(x+params[1], -params[0]), size)

    @staticmethod
    def log_likelihood(params, data, nonzero_only=False):
        """
        Calculates the log-likelihood on the data.

        :param params: two elements list containing the exponent (gamma) and shift (x0).
        :param data: input data as a numpy array.
        :param nonzero_only:  whether nonzero element should be considered only.  This is
        used after determining the parameters  and comparing to distributions that ignore
        zero values.
        :return: log-likelihood.
        """
        if params[0] < co.EPSILON:
            if params[1] < co.EPSILON:
                return co.delta.log_likelihood([0], data)
            else:
                return co.uniform.log_likelihood(None, data)
        else:
            c = float(zeta(params[0], params[1]))
            if c < co.EPSILON:
                return co.delta.log_likelihood([0], data)
            else:
                if nonzero_only:
                    _samples = data[np.where(data > 0)]
                else:
                    _samples = data
                return -params[0]*np.sum(np.log(_samples+params[1])) - len(_samples)*ln(c)

    @staticmethod
    def get_params(params):
        return "(gamma, x0) = (%.5f, %.5f)" % (params[0], params[1])
shifted_power_law = ShiftedPowerLaw()

