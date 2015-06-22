#!/usr/bin/env python3
#title          : model_selection.py
#description    : Contains methods that perform the actual model selection.
#author         : Enys Mones
#date           : 2015.06.21
#version        : 0.1
#usage          : python model_selection.py
#===========================================================================
import numpy as np
from math import exp, floor
from core import utils
from distributions import distribution as dist
from calculation import fit
from calculation import measures as me


# Model selection methods
MODEL_SELECTION_METHOD_AIC = 'aic'  # Akaike information criterion
MODEL_SELECTION_METHOD_KS = 'ks'  # Kolmogorov-Smirnov test
MODEL_SELECTION_METHOD_BIC = 'bic'  # Bayesian information criterion
MODEL_SELECTION_METHOD_LRT = 'lrt'  # Likelihood ratio test TODO

# Available methods
AVAILABLE_METHODS = [
    MODEL_SELECTION_METHOD_AIC,
    MODEL_SELECTION_METHOD_BIC,
    MODEL_SELECTION_METHOD_KS
]


def print_pmfs(data, fit_results, output_name):
    """
    Prints result distributions in the given output.

    :param data: data used for model selection.
    :param fit_results: fit results for each distribution.
    :param output_name: name of output file.
    """
    print("  printing probability mass functions")
    pmfs = {}
    for d in dist.get():
        pmfs[d] = dist.pmf(d, fit_results[d]['params'], domain=np.max(data)+1)
    output = []
    for i in range(np.max(data)+1):
        row = [float(i), 0.0]
        for d in dist.get():
            row.append(pmfs[d][i])
        output.append(row)
    for s in data:
        output[int(floor(s))][1] += 1.0
    for i in range(len(output)):
        output[i][1] /= float(len(data))
    utils.print_csv(output_name, ['value', 'p_measured'] + dist.get(), output)


def perform_aic_test(data, output_name):
    """
    Performs model selection based on the Akaike information criterion.

    :param data: input data.
    :param output_name: name of output file that contains the probability mass function
    of the original data and the fitted distributions with their optimal parameters.
    """
    print("AIC test")
    print("  number of samples: %i" % len(data))
    print("  fitting distribution")
    fit_results = {}
    npdata = np.array(data)
    for d in dist.get():
        fit_results[d] = fit.fit_mle(d, npdata)
    aic = {d: me.aic_measure(fit_results[d]['log-likelihood'], len(fit_results[d]['params'])) for d in fit_results}
    daic = {d: aic[d] - min(aic.values()) for d in aic}
    weights = {d: exp(-daic[d]/2) for d in daic}
    weights_total = sum(weights.values())
    for d in dist.get():
        weights[d] /= weights_total
        print("  %s:" % d.upper())
        print("    %s" % dist.get_params(fit_results[d]['params'], d))
        print("    AIC  = %.f" % aic[d])
        print("    dAIC = %.f" % daic[d])
        print("    w    = %r" % weights[d])
    print_pmfs(npdata, fit_results, output_name)


def perform_bic_test(data, output_name):
    """
    Performs model selection based on the Bayesian information criterion.

    :param data: input data.
    :param output_name: name of output file that contains the probability mass function
    of the original data and the fitted distributions with their optimal parameters.
    """
    print("BIC test")
    print("  number of samples: %i" % len(data))
    print("  fitting distribution")
    fit_results = {}
    npdata = np.array(data)
    for d in dist.get():
        fit_results[d] = fit.fit_mle(d, npdata)
    bic = {d: me.bic_measure(fit_results[d]['log-likelihood'], len(fit_results[d]['params']), len(npdata)) for d in fit_results}
    dbic = {d: bic[d] - min(bic.values()) for d in bic}
    weights = {d: exp(-dbic[d]/2) for d in dbic}
    weights_total = sum(weights.values())
    for d in dist.get():
        weights[d] /= weights_total
        print("  %s:" % d.upper())
        print("    %s" % dist.get_params(fit_results[d]['params'], d))
        print("    BIC  = %.f" % bic[d])
        print("    dBIC = %.f" % dbic[d])
        print("    w    = %r" % weights[d])
    print_pmfs(npdata, fit_results, output_name)


def perform_ks_test(data, output_name, synthetic_samples_num=100):
    """
    Performs model selection based on the K-S goodness-of-fit statistics.

    :param data: input data.
    :param output_name: name of output file that contains the probability mass function
    of the original data and the fitted distributions with their optimal parameters.
    """
    print("K-S test")
    print("  number of samples: %i" % len(data))
    print("  fitting distribution")
    fit_results = {}
    npdata = np.array(data)
    for d in dist.get():
        print("  %s:" % d.upper())
        fit_results[d] = fit.fit_ks(d, npdata)
        print("    %s" % dist.get_params(fit_results[d]['params'], d))
        print("    D = %r" % fit_results[d]['D'])
        params = fit_results[d]['params']
        p = 0
        for r in range(synthetic_samples_num):
            synthetic_sample = dist.sample(d, params, len(data))
            ksd = me.ks_statistics(dist.get_sample_cdf(synthetic_sample), dist.cdf(d, params, np.max(synthetic_sample)))
            if ksd > fit_results[d]['D']:
                p += 1
        print("    p = %r" % (float(p)/float(synthetic_samples_num)))
    print_pmfs(npdata, fit_results, output_name)
