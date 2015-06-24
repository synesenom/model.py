========
model.py
========
----------------------------------------
Model selection based on various methods
----------------------------------------

:Author: Enys Mones
:Version: 0.1
:License: MIT

This python script (model.py) implements model selection for discrete distributions.
As being motivated by degree distributions in complex networks, currently it supports only distributions over
non-negative integers and the following models:

- Poisson
- exponential
- log-normal
- Weibull
- shifted power-law
- truncated power-law (power-law with cutoff)
- normal


Requirements
============

It requires the following python packages:

- ``csv``
- ``argparse``
- ``mpmath``
- ``scipy``
- ``numpy``


Usage
=====

Passing ``--help`` will print the help menu.


Input files
===========

One-column CSV, with the numbers being the single sample values from the distribution.


Test
====

The script is shipped with some basic testing methods which can be accessed by the corresponding commands when started.


Output file
===========

Distribution of the original data and the optimal theoretical distribution of all models.


TODO
====

More distributions...


Bibliography
============

.. [1] Aaron Causet, Cosma Rohilla Shalizi and M. E. J. Newman: Power-law distributions in empirical data.
	   *SIAM Review* **51** (4) (2009): 661-703.

.. [2] M. P. H. Stumpf and P. J. Ingram: Probability models for degree distributions of protein interaction networks.
	   *Europhys. Lett.* **71** (1) (2005): 152-158.

.. [3] Gideon Schwarz: Estimating the dimension of a model.
	   *The Annals of Statistics* **6** (2) (1978): 461-464.