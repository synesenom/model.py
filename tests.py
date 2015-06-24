#!/usr/bin/env python3
#title          : tests.py
#description    : Contains testing methods for the various functions.
#author         : Enys Mones
#date           : 2015.06.21
#version        : 0.1
#usage          : python tests.py
#=========================================================================
import numpy as np
from mpmath import exp
from core import utils
from distributions import distribution as dist
from calculation import fit
from calculation import measures as me
from calculation.model_selection import print_pmfs


def test_sampling(distribution):
    """
    Tests sampling method for a given distribution.
    During the test, this method generates samples from the specific distribution.
    Results of the test are written in a file called 'TEST-SAMPLING.CSV', which contains the
    theoretical and actual distributions.

    :param distribution: distribution to test.
    """
    print("TESTING: sampling for %s distribution" % distribution)
    params = dist.DISTRIBUTIONS[distribution][dist.KEY_TEST_PARAMS]
    print("  input parameters: %s" % dist.get_params(params, distribution))
    print("  creating pdf")
    test_pmf = dist.pmf(distribution, params)
    print("  generating samples")
    test_sample_pmf = dist.get_sample_pmf(dist.samples(distribution, params))
    test_distribution = []
    for i in range(len(test_pmf)):
        if i < len(test_sample_pmf):
            test_distribution.append([i, test_pmf[i], test_sample_pmf[i]])
        else:
            test_distribution.append([i, test_pmf[i], 0.0])
    test_output = 'TEST-SAMPLE.CSV'
    print("  printing results in %s" % test_output)
    utils.print_csv(test_output, ['value', 'probability', 'relative_frequency'], test_distribution)


def test_fit_mle(distribution):
    """
    Tests MLE fit of a given distribution.
    The test generates samples for all distributions and then performs MLE fit of the specified
    distribution. Should test robustness of MLE calculations and precision of fit on a sample from
    its own distribution.

    :param distribution: distribution to test.
    """
    print("TESTING: MLE fit for %s distribution" % distribution.upper())
    print("  fitting to others")
    for sample_dist in dist.get():
        print("    %s" % sample_dist.upper())
        params = dist.DISTRIBUTIONS[sample_dist][dist.KEY_TEST_PARAMS]
        test_sample = dist.samples(sample_dist, params)
        fit_result = fit.fit_mle(distribution, test_sample)
        print("      input parameters: %s" % dist.get_params(params, sample_dist))
        print("      fit parameters: %s" % dist.get_params(fit_result['params'], distribution))
        print("      log-likelihood: %r" % fit_result['log-likelihood'])
        print("      K-S statistics: %r" % fit_result['D'])


def test_fit_ks(distribution):
    """
    Tests K-S fit of a given distribution.
    The test generates samples for all distributions and then performs K-S fit of the specified
    distribution. Should test robustness of K-S calculations and precision of fit on a sample from
    its own distribution.

    :param distribution: distribution to test.
    """
    print("TESTING: K-S fit for %s distribution" % distribution.upper())
    print("  fitting to others")
    for sample_dist in dist.get():
        print("    %s" % sample_dist.upper())
        params = dist.DISTRIBUTIONS[sample_dist][dist.KEY_TEST_PARAMS]
        test_sample = dist.samples(sample_dist, params)
        fit_results = fit.fit_ks(distribution, test_sample)
        print("      input parameters: %s" % dist.get_params(params, sample_dist))
        print("      fit parameters: %s" % dist.get_params(fit_results['params'], distribution))
        print("      log-likelihood: %r" % fit_results['log-likelihood'])
        print("      K-S statistics: %r" % fit_results['D'])


def test_aic_ms(distribution):
    """
    Tests AIC model selection.
    During the test, this method generates a sample with the specified distribution and then
    calculates AIC for all other distributions (including the tested one). Finally, these are
    compared and the one with the largest AIC weight is chosen.

    :param distribution:
    :return:
    """
    print("TESTING: AIC model selection for %s distribution" % distribution.upper())
    params = dist.DISTRIBUTIONS[distribution][dist.KEY_TEST_PARAMS]
    print("  creating sample")
    test_sample = dist.samples(distribution, params)
    print("  calculating AIC for all distributions")
    fit_results = {}
    aic = {}
    for d in dist.get():
        fit_results[d] = fit.fit_mle(d, test_sample)
        aic[d] = me.aic_measure(dist.log_likelihood(d, fit_results[d]['params'], test_sample, nonzero_only=True),
                                len(fit_results[d]['params']))
    delta_aic = {d: aic[d]-min(aic.values()) for d in aic}
    weights = {d: float(exp(-delta_aic[d]/2)) for d in delta_aic}
    best_model = dist.get()[0]
    print("  input parameters: %s" % dist.get_params(params, distribution))
    for d in dist.get():
        if weights[d] > weights[best_model]:
            best_model = d
        weights[d] /= sum(weights.values())
        print("  %s:" % d.upper())
        print("    %s" % dist.get_params(fit_results[d]['params'], d))
        print("    AIC  = %.0f" % aic[d])
        print("    dAIC = %.0f" % delta_aic[d])
        print("    w    = %r" % weights[d])
    print("  Most likely model: %s" % best_model.upper())
    print_pmfs(test_sample, fit_results, 'TEST-AIC.CSV')


def test_bic_ms(distribution):
    """
    Tests BIC model selection.
    During the test, this method generates a sample with the specified distribution and then
    calculates BIC for all other distributions (including the tested one). Finally, these are
    compared and the one with the largest BIC weight is chosen.

    :param distribution:
    :return:
    """
    print("TESTING: BIC model selection for %s distribution" % distribution.upper())
    params = dist.DISTRIBUTIONS[distribution][dist.KEY_TEST_PARAMS]
    print("  creating sample")
    test_sample = dist.samples(distribution, params)
    print("  calculating BIC for all distributions")
    fit_results = {}
    bic = {}
    for d in dist.get():
        fit_results[d] = fit.fit_mle(d, test_sample)
        bic[d] = me.bic_measure(dist.log_likelihood(d, fit_results[d]['params'], test_sample, nonzero_only=True),
                                len(fit_results[d]['params']), len(test_sample))
    delta_bic = {d: bic[d]-min(bic.values()) for d in bic}
    weights = {d: float(exp(-delta_bic[d]/2)) for d in delta_bic}
    best_model = dist.get()[0]
    print("  input parameters: %s" % dist.get_params(params, distribution))
    for d in dist.get():
        if weights[d] > weights[best_model]:
            best_model = d
        weights[d] /= sum(weights.values())
        print("  %s:" % d.upper())
        print("    %s" % dist.get_params(fit_results[d]['params'], d))
        print("    BIC  = %.0f" % bic[d])
        print("    dBIC = %.0f" % delta_bic[d])
        print("    w    = %r" % weights[d])
    print("  Most likely model: %s" % best_model.upper())
    print_pmfs(test_sample, fit_results, 'TEST-BIC.CSV')


def test_ks_ms(distribution):
    """
    Tests K-S model selection.
    During the test, this method generates a sample with the specified distribution and then
    calculates K-S statistics for all other distributions (including the tested one). Finally, these are
    compared and the one with the lowest K-S weight is chosen.

    :param distribution:
    :return:
    """
    print("TESTING: K-S model selection for %s distribution" % distribution.upper())
    params = dist.DISTRIBUTIONS[distribution][dist.KEY_TEST_PARAMS]
    print("  creating sample")
    test_sample = dist.samples(distribution, params)
    print("  calculating K-S statistics for all distributions")
    print("  input parameters: %s" % dist.get_params(params, distribution))
    fit_results = {}
    best_ksd = 1.0
    best_model = dist.get()[0]
    for d in dist.get():
        print("  %s:" % d.upper())
        fit_results[d] = fit.fit_ks(d, test_sample)
        if fit_results[d]['D'] < best_ksd:
            best_ksd = fit_results[d]['D']
            best_model = d
        print("    %s" % dist.get_params(fit_results[d]['params'], d))
        print("    D = %r" % fit_results[d]['D'])
        params = fit_results[d]['params']
        p = 0
        for r in range(100):
            synthetic_sample = dist.samples(d, params, len(test_sample))
            ksd = me.ks_statistics(dist.get_sample_cdf(synthetic_sample), dist.cdf(d, params, np.max(synthetic_sample)))
            if ksd > fit_results[d]['D']:
                p += 1
        print("    p = %r" % (float(p)/100.0))
    print("  Best fitting model: %s" % best_model.upper())
    print_pmfs(test_sample, fit_results, 'TEST-KS.CSV')