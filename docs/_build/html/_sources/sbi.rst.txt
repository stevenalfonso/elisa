Binary Star SBI
===============

ELISA includes a Simulation-Based Inference (SBI) module for estimating binary-star parameters — primary mass :math:`M_1`, mass ratio :math:`q`, and optionally age, metallicity, and distance — from Gaia photometry of cluster members.

The approach uses Sequential Neural Posterior Estimation (SNPE): a forward simulator generates synthetic Gaia observations from a prior over stellar parameters, and a neural density estimator learns the posterior directly, without evaluating a likelihood.

.. note::

   The SBI module requires ``torch`` and ``sbi``. Install them with::

      pip install torch sbi isochrones

Workflow Overview
-----------------

.. code-block:: text

   ElisaBinary()
     ├── set_tracks()                   # load MIST stellar evolution tracks
     ├── set_error_splines(csv_path)    # build photometric noise model
     ├── build_posterior(prior, ...)    # simulate + train neural posterior
     └── infer(posterior, df)           # run inference on observed data

Step 1: Load Stellar Tracks
----------------------------

.. code-block:: python

   from elisa.sbi import ElisaBinary

   elisa = ElisaBinary()
   elisa.set_tracks()

This loads the MIST isochrone grid via the ``isochrones`` package. The tracks are used internally by the forward simulator to compute synthetic magnitudes.

Step 2: Build Error Splines
----------------------------

The simulator adds realistic Gaia photometric noise by interpolating a spline fit to the observed magnitude-versus-error relation. You must provide a CSV file with columns ``Gmag``, ``e_Gmag``, ``BPmag``, ``e_BPmag``, ``RPmag``, ``e_RPmag``.

.. code-block:: python

   elisa.set_error_splines("cluster.csv")

You can also build and inspect the splines directly:

.. code-block:: python

   from elisa.sbi import build_error_splines

   splines = build_error_splines("cluster.csv", s=0.01, k=3)
   # splines['G'], splines['BP'], splines['RP'] are UnivariateSpline objects

Step 3: Define a Prior
-----------------------

The prior must be compatible with ``sbi`` — it needs ``.sample()`` and ``.log_prob()`` methods. ELISA provides ``StellarPrior`` as a convenience, but any compatible prior works.

**Simple uniform prior** (two free parameters: M1, q):

.. code-block:: python

   from sbi.utils import BoxUniform
   import torch

   prior = BoxUniform(
       low=torch.tensor([0.1, 0.0]),
       high=torch.tensor([5.0, 1.0]),
   )

**Astrophysically-motivated prior** using Beta distributions:

.. code-block:: python

   from elisa.sbi import StellarPrior

   prior = StellarPrior(
       M1_bounds=(0.1, 5.0),
       q_bounds=(0.0, 1.0),
       fixed_params={"age": 0.035, "feh": 0.0, "distance": 400.0},
   )

The ``fixed_params`` dict pins cluster-level parameters so they are not inferred. When age, metallicity, and distance are all fixed, the prior covers only ``M1`` and ``q``.

To also infer distance, leave it out of ``fixed_params`` and set ``with_distance=True``:

.. code-block:: python

   prior = StellarPrior(
       M1_bounds=(0.1, 5.0),
       q_bounds=(0.0, 1.0),
       tau_bounds=(0.01, 1.0),
       fe_h_bounds=(-0.5, 0.3),
       distance_bounds=(100.0, 600.0),
       with_distance=True,
   )

The parameter order in the prior must match the inferred parameters in this sequence: ``M1``, ``q``, ``age``, ``[Fe/H]``, ``distance`` (minus any that are fixed).

Step 4: Train the Posterior
----------------------------

.. code-block:: python

   posterior, theta, x = elisa.build_posterior(
       prior=prior,
       fixed_params={"age": 0.035, "feh": 0.0, "distance": 400.0},
       num_simulations=100_000,
       use_apparent_mags=True,
   )

This runs the forward simulator ``num_simulations`` times, filters invalid outputs, and trains an SNPE neural posterior estimator. The simulator outputs are in order ``[BP, RP, G, parallax]`` when ``use_apparent_mags=True``.

**Save the trained posterior to disk:**

.. code-block:: python

   elisa.save(posterior, theta, x, "cluster_posterior.pkl")

**Reload later:**

.. code-block:: python

   elisa, posterior, theta, x = ElisaBinary.load("cluster_posterior.pkl")

Step 5: Run Inference
----------------------

Pass the trained posterior and a DataFrame of observed stars:

.. code-block:: python

   results = elisa.infer(
       posterior=posterior,
       data=df_observed,
       labels=["M1", "q"],
       fixed_params={"age": 0.035, "feh": 0.0, "distance": 400.0},
       num_samples=10_000,
   )

The ``data`` DataFrame must contain Gaia photometry columns. ELISA auto-detects both the short names (``Gmag``, ``BPmag``, ``RPmag``, ``PlxCorr``) and Gaia DR3 raw names (e.g. ``phot_g_mean_mag``, ``parallax``).

The returned DataFrame has one row per star with columns ``{param}_median``, ``{param}_err_lower``, ``{param}_err_upper`` for each inferred parameter.

.. code-block:: python

   print(results[["M1_median", "M1_err_lower", "M1_err_upper"]].head())

Full Example
-------------

.. code-block:: python

   import pandas as pd
   from elisa.sbi import ElisaBinary, StellarPrior

   # Load data
   df = pd.read_csv("pleiades_members.csv")

   # Set up the estimator
   elisa = ElisaBinary()
   elisa.set_tracks()
   elisa.set_error_splines("pleiades_members.csv")

   # Define prior (age, metallicity, distance fixed to known cluster values)
   prior = StellarPrior(
       M1_bounds=(0.1, 3.0),
       q_bounds=(0.0, 1.0),
       fixed_params={"age": 0.125, "feh": 0.0, "distance": 136.0},
   )

   # Train
   posterior, theta, x = elisa.build_posterior(
       prior=prior,
       fixed_params={"age": 0.125, "feh": 0.0, "distance": 136.0},
       num_simulations=50_000,
   )
   elisa.save(posterior, theta, x, "pleiades_posterior.pkl")

   # Infer
   results = elisa.infer(
       posterior=posterior,
       data=df,
       labels=["M1", "q"],
       fixed_params={"age": 0.125, "feh": 0.0, "distance": 136.0},
   )
   print(results.head())
