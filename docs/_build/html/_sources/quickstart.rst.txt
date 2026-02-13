Quick Start
===========

The ELISA workflow follows five steps:

1. Download or load an isochrone grid
2. Load observed photometry
3. Configure priors and build the posterior
4. Run MCMC
5. Analyze results

Step 1: Download an Isochrone Grid
----------------------------------

.. code-block:: python

   from elisa import ElisaClusterInference

   elisa_inference = ElisaClusterInference()

   iso_grid = elisa_inference.download_isochrones(
       logage_range=(7.0, 10.0, 0.2),
       MH_range=(-2.5, 1.0, 0.25),
       photsys='gaiaEDR3',
       output_dir='isochrone_data',
       output_name='parsec_gaia_grid',
       save_file=True,
   )

To reload a previously saved grid:

.. code-block:: python

   iso_grid = elisa_inference.load_isochrone_grid(
       'isochrone_data/parsec_gaia_grid.parquet'
   )

Step 2: Load Observed Data
--------------------------

ELISA provides :class:`~elisa.query.data.ElisaQuery` to fetch photometry from
Gaia and load cluster membership catalogs from VizieR.

Available catalogs: ``alfonso-2024``, ``cantat-gaudin-2020``, ``hunt-2023``,
``vasiliev-2021``.

.. code-block:: python

   from elisa.query.data import ElisaQuery

   elisa_query = ElisaQuery()

   df_clusters, df_members = elisa_query.load_catalog(
       'alfonso-2024', load_members=True
   )

   source_ids = df_members[
       df_members['Cluster'] == 'Melotte_22'
   ]['GaiaDR3'].values

   df = elisa_query.gaia_source_id(source_id=source_ids)

Prepare the observed magnitudes and errors as NumPy arrays with shape
``(n_stars, 3)`` for the bands ``[G, BP, RP]``:

.. code-block:: python

   import numpy as np

   df.rename(columns={
       'phot_g_mean_mag': 'Gmag',
       'phot_bp_mean_mag': 'BPmag',
       'phot_rp_mean_mag': 'RPmag',
   }, inplace=True)

   observed_mags = df[['Gmag', 'BPmag', 'RPmag']].values
   observed_errors = np.column_stack([
       df['parallax_error'].values,
       df['parallax_error'].values,
       df['parallax_error'].values,
   ])

Step 3: Set Up the Posterior
----------------------------

Define initial parameter guesses and priors. ELISA supports two prior types:

- ``'uniform'`` with ``(low, high)``
- ``'gaussian'`` with ``(mean, std)``

.. code-block:: python

   init_logAge = 8.1
   init_MH = 0.0
   distance_pc = 136.0
   init_dm = 5 * np.log10(distance_pc / 10)
   init_AV = 0.1

   posterior = elisa_inference.setup_logposterior(
       grid=iso_grid,
       observed_mags=observed_mags,
       observed_errors=observed_errors,
       prior_logAge=(6.0, 10.5),
       prior_MH=(init_MH, 0.3),
       prior_dm=(init_dm, 1.0),
       prior_AV=(init_AV, 0.2),
       prior_type={
           'logAge': 'uniform',
           'MH': 'gaussian',
           'dm': 'gaussian',
           'AV': 'gaussian',
       },
   )

   init_params = posterior.get_initial_params(
       logAge_init=init_logAge,
       MH_init=init_MH,
       dm_init=init_dm,
       AV_init=init_AV,
   )

Step 4: Run MCMC
----------------

.. code-block:: python

   sampler = elisa_inference.run_mcmc(
       log_posterior=posterior,
       init_params=init_params,
       n_walkers=32,
       n_steps=2000,
       progress=True,
   )

``run_mcmc`` also supports parallel execution with ``parallel=True`` and
``n_cores=4``.

Step 5: Analyze Results
-----------------------

**Convergence diagnostics:**

.. code-block:: python

   n_burn = 200
   chains = sampler.get_chain(discard=n_burn).transpose(1, 0, 2)
   R_hat = elisa_inference.get_gelman_rubin(chains)

**Autocorrelation and thinning:**

.. code-block:: python

   tau = sampler.get_autocorr_time(quiet=True)
   burnin = int(2 * np.max(tau))
   thin = int(0.5 * np.min(tau))
   flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

**Results summary:**

.. code-block:: python

   results = elisa_inference.get_results_summary(flat_samples)

**Corner plot:**

.. code-block:: python

   import corner

   labels = ["logAge", "[M/H]", "dm", "A_V"]
   corner.corner(
       flat_samples, labels=labels,
       quantiles=[0.16, 0.5, 0.84], show_titles=True,
   )

**Isochrone fit overlay:**

.. code-block:: python

   fig, ax = elisa_inference.plot_isochrone_fit(
       flat_samples=flat_samples,
       observed_mags=observed_mags,
       n_draws=200,
   )
