.. figure:: _static/images/bird.jpeg
   :width: 30%
   :align: center
   :alt: Bird

ELISA
====================

**Efficient Likelihood Inference for Stellar Ages**

``ELISA`` is a Python package for Bayesian inference of stellar cluster parameters
from Gaia photometry. It provides two complementary inference engines:

- **MCMC isochrone fitting** — infers cluster-level parameters (age, metallicity,
  distance modulus, extinction) by fitting PARSEC isochrones with ``emcee``.
- **Binary-star SBI** — estimates individual star parameters (primary mass,
  mass ratio) using Simulation-Based Inference with neural posterior estimation.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   sbi
   api
