# ELISA

**E**fficient **L**ikelihood **I**nference for **S**tellar **A**ges

A Python package for Bayesian inference of stellar cluster parameters using PARSEC isochrone fitting and MCMC sampling.

## Overview

ELISA infers four fundamental cluster parameters from Gaia photometry:

| Parameter | Description |
|-----------|-------------|
| `logAge` | Log10 of the cluster age (yr) |
| `[M/H]` | Metallicity |
| `dm` | Distance modulus (m - M) |
| `A_V` | V-band extinction |

It downloads PARSEC isochrones, builds an interpolation grid, and uses `emcee` to sample the posterior distribution of these parameters given observed magnitudes and errors.

## Installation

### Requirements

- Python >= 3.9

### From source

```bash
git clone https://github.com/steven/elisa.git
cd elisa
pip install -e .
```

### Optional dependencies

```bash
# Corner plots and Jupyter support
pip install -e ".[all]"

# Development tools (pytest, black, ruff)
pip install -e ".[dev]"
```

## Usage Guide

The workflow follows five steps:

1. Download or load an isochrone grid
2. Load observed photometry
3. Configure priors and build the posterior
4. Run MCMC
5. Analyze results

### 1. Download an Isochrone Grid

```python
from elisa import ElisaClusterInference

elisa_inference = ElisaClusterInference()

iso_grid = elisa_inference.download_isochrones(
    logage_range=(7.0, 10.0, 0.2),       # (min, max, step) in log(age/yr)
    MH_range=(-2.5, 1.0, 0.25),          # (min, max, step) in [M/H]
    photsys='gaiaEDR3',
    output_dir='isochrone_data',
    output_name='parsec_gaia_grid',
    save_file=True                        # saves .parquet + metadata
)
```

To reload a previously saved grid:

```python
iso_grid = elisa_inference.load_isochrone_grid('isochrone_data/parsec_gaia_grid.parquet')
```

### 2. Load Observed Data

ELISA provides `ElisaQuery` to fetch photometry from Gaia and load cluster membership catalogs from VizieR.

**Available catalogs:** `alfonso-2024`, `cantat-gaudin-2020`, `hunt-2023`, `vasiliev-2021`

```python
from elisa.query.data import ElisaQuery

elisa_query = ElisaQuery()

# Load a cluster catalog with member lists
df_clusters, df_members = elisa_query.load_catalog('alfonso-2024', load_members=True)

# Get Gaia source IDs for a specific cluster
source_ids = df_members[df_members['Cluster'] == 'Melotte_22']['GaiaDR3'].values

# Query Gaia for full photometry
df = elisa_query.gaia_source_id(source_id=source_ids)
```

Prepare the observed magnitudes and errors as NumPy arrays with shape `(n_stars, 3)` for the bands `[G, BP, RP]`:

```python
import numpy as np

df.rename(columns={
    'phot_g_mean_mag': 'Gmag',
    'phot_bp_mean_mag': 'BPmag',
    'phot_rp_mean_mag': 'RPmag'
}, inplace=True)

observed_mags = df[['Gmag', 'BPmag', 'RPmag']].values
observed_errors = np.column_stack([
    df['parallax_error'].values,   # proxy for G error
    df['parallax_error'].values,   # proxy for BP error
    df['parallax_error'].values    # proxy for RP error
])
```

### 3. Set Up the Posterior

Define initial parameter guesses and priors. ELISA supports two prior types per parameter:
- `'uniform'` with `(low, high)`
- `'gaussian'` with `(mean, std)`

```python
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
        'AV': 'gaussian'
    }
)

init_params = posterior.get_initial_params(
    logAge_init=init_logAge,
    MH_init=init_MH,
    dm_init=init_dm,
    AV_init=init_AV
)
```

**(Optional) Refine initial parameters with MLE:**

```python
from scipy.optimize import minimize

result = minimize(
    lambda p: -posterior(p), init_params,
    method='Nelder-Mead',
    options={'maxiter': 10000, 'xatol': 1e-4, 'fatol': 1e-4}
)
init_params = result.x
```

### 4. Run MCMC

```python
sampler = elisa_inference.run_mcmc(
    log_posterior=posterior,
    init_params=init_params,
    n_walkers=32,
    n_steps=2000,
    progress=True
)
```

`run_mcmc` also supports parallel execution with `parallel=True` and `n_cores=4`.

### 5. Analyze Results

**Convergence diagnostics:**

```python
n_burn = 200
chains = sampler.get_chain(discard=n_burn).transpose(1, 0, 2)
R_hat = elisa_inference.get_gelman_rubin(chains)
# R-hat values close to 1.0 indicate convergence
```

**Autocorrelation and thinning:**

```python
tau = sampler.get_autocorr_time(quiet=True)
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
```

**Results summary:**

```python
results = elisa_inference.get_results_summary(flat_samples)
# Prints median +/- 1-sigma for each parameter, plus derived distance (pc) and age (Myr)
```

**Corner plot:**

```python
import corner

labels = ["logAge", "[M/H]", "dm", "A_V"]
corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
```

**Isochrone fit overlay:**

```python
fig, ax = elisa_inference.plot_isochrone_fit(
    flat_samples=flat_samples,
    observed_mags=observed_mags,
    n_draws=200
)
```

## API Reference

### `ElisaClusterInference`

| Method | Description |
|--------|-------------|
| `download_isochrones(logage_range, MH_range, photsys, ...)` | Download a PARSEC isochrone grid |
| `load_isochrone_grid(filepath)` | Load a saved `.parquet` or `.csv` grid |
| `setup_logposterior(grid, observed_mags, observed_errors, ...)` | Build the posterior with priors |
| `run_mcmc(log_posterior, init_params, ...)` | Run emcee MCMC sampling |
| `get_gelman_rubin(chain)` | Compute R-hat convergence diagnostic |
| `get_results_summary(flat_samples)` | Print and return median/percentile summaries |
| `plot_isochrone_fit(flat_samples, observed_mags, ...)` | Plot posterior isochrones over observed CMD |

### `ElisaQuery`

| Method | Description |
|--------|-------------|
| `load_catalog(catalog_name, load_members)` | Load cluster catalogs from VizieR |
| `gaia_source_id(source_id, ...)` | Query Gaia DR3 by source ID |

## References

- PARSEC isochrones: [Bressan et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.427..127B)
- emcee: [Foreman-Mackey et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F)
- Likelihood formulation: [von Hippel et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006ApJ...645.1436V)

## License

MIT License - see [LICENSE](LICENSE) for details.
