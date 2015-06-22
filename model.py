#!/usr/bin/env python3
#title          : model.py
#description    : Selects the best model for a sample.
#author         : Enys Mones
#date           : 2015.06.10
#version        : 0.1
#usage          : python model.py
# TODO log-logistic, logistic, gumbel
#===================================================================
from core import args
from core import utils
from distributions import distribution as dist
from calculation import model_selection as ms


a = args.Args(name="model.py",
              desc="Selects the best model for a distribution.")
params = a\
    .add(key='--input', dest='input', default=None,
         help='Input file containing the measured samples.')\
    .add(key='--output', dest='output', default=None,
         help='Output file, results are stored here.')\
    .add(key='--select', dest='select', default=None,
         help='Model selection with the given method to use (%s)' % ', '.join(ms.AVAILABLE_METHODS))\
    .add(key='--test-sampling', dest='test_sampling', default=None,
         help='Test sampling from the given distribution (%s)' % ', '.join(dist.get()))\
    .add(key='--test-mle-fit', dest='test_mle_fit', default=None,
         help='Test MLE fit for the given distribution (%s)' % ', '.join(dist.get()))\
    .add(key='--test-ks-fit', dest='test_ks_fit', default=None,
         help='Test K-S fit for the given distribution (%s)' % ', '.join(dist.get()))\
    .add(key='--test-aic-ms', dest='test_aic_ms', default=None,
         help='Test AIC model selection for the given distribution (%s)' % ', '.join(dist.get()))\
    .add(key='--test-bic-ms', dest='test_bic_ms', default=None,
         help='Test BIC model selection for the given distribution (%s)' % ', '.join(dist.get()))\
    .add(key='--test-ks-ms', dest='test_ks_ms', default=None,
         help='Test K-S model selection for the given distribution (%s)' % ', '.join(dist.get()))\
    .get()

# Testing
if params['test_sampling'] is not None:
    from tests import test_sampling
    test_sampling(params['test_sampling'])
if params['test_mle_fit'] is not None:
    from tests import test_fit_mle
    test_fit_mle(params['test_mle_fit'])
if params['test_ks_fit'] is not None:
    from tests import test_fit_ks
    test_fit_ks(params['test_ks_fit'])
if params['test_aic_ms'] is not None:
    from tests import test_aic_ms
    test_aic_ms(params['test_aic_ms'])
if params['test_bic_ms'] is not None:
    from tests import test_bic_ms
    test_bic_ms(params['test_bic_ms'])
if params['test_ks_ms'] is not None:
    from tests import test_ks_ms
    test_ks_ms(params['test_ks_ms'])

# Calculations
if params['select'] is not None:
    print("reading data")
    data = []
    for row in utils.read_csv(params['input']):
        data.append(float(row[0]))

    if params['select'] == ms.MODEL_SELECTION_METHOD_AIC:
        ms.perform_aic_test(data, params['output'])

    if params['select'] == ms.MODEL_SELECTION_METHOD_BIC:
        ms.perform_bic_test(data, params['output'])

    if params['select'] == ms.MODEL_SELECTION_METHOD_KS:
        ms.perform_ks_test(data, params['output'])
