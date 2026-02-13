ELISA Documentation
====================

**Efficient Likelihood Inference for Stellar Ages**

ELISA is a Python package for Bayesian inference of stellar cluster parameters
using PARSEC isochrone fitting and MCMC sampling.

It infers four fundamental cluster parameters from Gaia photometry:

- **logAge** -- Log10 of the cluster age (yr)
- **[M/H]** -- Metallicity
- **dm** -- Distance modulus (m - M)
- **A_V** -- V-band extinction

ELISA downloads PARSEC isochrones, builds an interpolation grid, and uses
``emcee`` to sample the posterior distribution of these parameters given
observed magnitudes and errors.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api
