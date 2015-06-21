#!/usr/bin/env python3
#title          : core.py
#description    : Abstract base class for core.
#author         : Enys Mones
#date           : 2015.06.19
#version        : 0.1
#usage          : python core.py
#========================================================
import numpy as np
from scipy import stats
from mpmath import ln, sqrt

#############
# CONSTANTS #
#############
# Tolerance for substitution of core.
# For performance and robustness reasons, some core are approximated
# by delta or uniform core, when their parameters approach some critical
# values. EPSILON is used for detecting these cases.
EPSILON = 0.001

# Default domain size for generating probability mass functions.
DEFAULT_PDF_MAX = 10000

# Default domain size for generating random samples.
DEFAULT_SAMPLE_MAX = DEFAULT_PDF_MAX

# Default number of random samples to generate.
# Used mostly for testing.
DEFAULT_SAMPLE_SIZE = 10000


###########
# CLASSES #
###########
class RealDistribution():
    """
    The abstract base class for all the core.

    All core have to implement the following:
    1) Probability mass function for testing and K-S optimization.
    2) Sampling method for testing and p-values of K-S statistics.
    3) Log-likelihood for MLE and information criteria.
    """

    @staticmethod
    def pmf(params, domain=DEFAULT_PDF_MAX):
        """
        Returns the probability mass function.

        :param params: a list containing the parameters.
        :param domain: domain size of the probability mass function.
        :return: probability mass function.
        """
        raise NotImplementedError("Subclass must implement pmf(params, domain).")

    @staticmethod
    def samples(params, size=DEFAULT_SAMPLE_SIZE, domain=DEFAULT_SAMPLE_MAX):
        """
        Returns a give number of samples.

        :param params: a list containing the parameters.
        :param size: number of samples to return.
        :param domain: domain size of the samples to draw from.
        :return: samples in a numpy array.
        """
        raise NotImplementedError("Subclass must implement samples(params, size, domain).")

    @staticmethod
    def log_likelihood(params, data):
        """
        Returns the log-likelihood of the distribution for a given sample.

        :param params: a list containing the parameters.
        :param data: the data over which the log-likelihood should be calculated.
        :return: the log-likelihood.
        """
        raise NotImplementedError("Subclass must implement log_likelihood(params, data).")

    @staticmethod
    def get_params(params):
        """
        Returns a printable string of the distribution parameters.

        :param params: list of parameters.
        :return: printable string in the format of '(<name1>, <name2>, ...) = (<value1>, <value2>, ...)'
        where <nameX> and <valueX> corresponds to the name and value of parameter X.
        """
        return NotImplementedError("Subclass must implement get_params(params).")


class Delta(RealDistribution):
    """
    Dirac delta distribution.
    This distribution is used for approximating other distributions when some parameters
    approach critical values and they can be replaced by a delta function safely.
    """

    @staticmethod
    def pmf(params, domain=DEFAULT_PDF_MAX):
        """
        Probability mass function of a delta distribution.

        :param params: single element list with the location parameter.
        :param domain: domain size.
        :return: probability mass function.
        """
        _pmf = np.append(np.zeros(int(params[0])), [1.0])
        return np.append(_pmf, np.zeros(domain-int(params[0])-1))

    @staticmethod
    def samples(params, size=DEFAULT_SAMPLE_SIZE, domain=DEFAULT_SAMPLE_MAX):
        """
        Generates samples for a delta distribution.

        :param params: single element list with the location parameter.
        :param size: number of samples.
        :param domain: unused.
        :return: numpy array of samples.
        """
        return np.ones(size) * int(params[0])

    @staticmethod
    def log_likelihood(params, data):
        """
        Returns the log-likelihood of a delta distribution.
        The distribution is approximated by a narrow Gaussian.

        :param params: single element list with the location parameter.
        :param data: the data over which the log-likelihood should be calculated.
        :return: log-likelihood.
        """
        return -len(data)*ln(EPSILON*sqrt(2*np.pi)) - 0.5*np.sum(0.5*np.power(data-params[0], 2))/EPSILON**2
delta = Delta()


class Uniform(RealDistribution):
    """
    Uniform distribution.
    Mostly used when other distributions are approximated in case of some of their parameters
    approach critical values where they can be replaced by a uniform distribution safely.
    """

    @staticmethod
    def pmf(params, domain=DEFAULT_PDF_MAX):
        """
        Probability mass function of a uniform distribution.

        :param params: unused.
        :param domain: domain size.
        :return: probability mass function.
        """
        return np.ones(domain+1)/float(domain+1)

    @staticmethod
    def samples(params, size=DEFAULT_SAMPLE_SIZE, domain=DEFAULT_SAMPLE_MAX):
        """
        Generates samples for a uniform distribution.

        :param params: unused.
        :param size: number of samples.
        :param domain: domain size.
        :return: numpy array of samples.
        """
        return np.random.uniform(0, domain, size)

    @staticmethod
    def log_likelihood(params, data):
        """
        Returns the log-likelihood of a uniform distribution.

        :param params: unused.
        :param data: the data over which the log-likelihood should be calculated.
        :return: log-likelihood.
        """
        return -len(data) * ln(float(np.max(data)))
uniform = Uniform()


def generate_discrete_samples(values, probabilities, size=DEFAULT_SAMPLE_SIZE):
    """
    Generates a sample of discrete random variables specified by the probabilities.

    :param values: domain of values.
    :param probabilities: probabilities. Must have the same length as the domain of values.
    :param size: number of samples to return.
    :return: list of samples.
    """
    if len(values) != len(probabilities):
        print("Error: len(x) != len(p)")
        return
    else:
        _random_sampler = stats.rv_discrete(values=(values, probabilities/np.sum(probabilities)))
        return _random_sampler.rvs(size=size)