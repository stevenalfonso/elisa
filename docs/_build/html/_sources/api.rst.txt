API Reference
=============

This page documents the classes and functions used to run inferences and
download data. Internal helpers and low-level utilities are omitted.

ElisaClusterInference
---------------------

Main class for MCMC-based cluster parameter inference.

.. autoclass:: elisa.mcmc.elisa.ElisaClusterInference
   :members: download_isochrones, load_isochrone_grid, setup_logposterior,
             run_mcmc, get_gelman_rubin, get_results_summary, plot_isochrone_fit
   :show-inheritance:

Posterior
---------

Log-posterior callable returned by ``setup_logposterior``.

.. autoclass:: elisa.mcmc.posterior.Posterior
   :members: get_initial_params
   :show-inheritance:

ElisaBinary
-----------

Simulation-Based Inference for binary-star parameters.

.. autoclass:: elisa.sbi.binary.ElisaBinary
   :members: set_tracks, set_error_splines, build_posterior, infer, save, load
   :show-inheritance:

StellarPrior
------------

Astrophysically-motivated prior for binary-star SBI.

.. autoclass:: elisa.sbi.priors.StellarPrior
   :members: sample, log_prob
   :show-inheritance:

build_error_splines
-------------------

.. autofunction:: elisa.sbi.binary.build_error_splines

ElisaQuery
----------

Download cluster catalogs and query Gaia photometry.

.. autoclass:: elisa.query.data.ElisaQuery
   :members: load_catalog, gaia_source_id
   :show-inheritance:
