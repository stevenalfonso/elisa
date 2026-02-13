"""Shared fixtures for ELISA tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_grid():
    """Create a small synthetic isochrone grid for testing.

    Returns a DataFrame mimicking PARSEC output with 2 ages, 2 metallicities,
    and a handful of masses per isochrone.
    """
    rows = []
    ages = [8.0, 9.0]
    metallicities = [0.0, 0.5]
    masses = [0.5, 1.0, 2.0, 5.0]

    for age in ages:
        for mh in metallicities:
            for m in masses:
                # Simple magnitude model: brighter for higher mass, older = slightly fainter
                G = 5.0 - 2.5 * np.log10(m**3.5) + 0.1 * (age - 8.0) + 0.05 * mh
                BP = G + 0.3 + 0.5 / m
                RP = G - 0.2 - 0.1 / m
                rows.append({
                    'logAge': age,
                    'MH': mh,
                    'Mini': m,
                    'Mass': m * 0.99,
                    'label': 1,
                    'logTe': 3.7 + 0.1 * np.log10(m),
                    'logg': 4.5 - np.log10(m),
                    'logL': -0.5 + 3.5 * np.log10(m),
                    'Gmag': G,
                    'G_BPmag': BP,
                    'G_RPmag': RP,
                })

    return pd.DataFrame(rows)


@pytest.fixture
def observed_data(synthetic_grid):
    """Generate synthetic observed photometry from the grid.

    Picks a single isochrone (logAge=8.0, MH=0.0), applies a distance modulus,
    and adds small Gaussian noise.
    """
    rng = np.random.default_rng(42)
    iso = synthetic_grid[
        (synthetic_grid['logAge'] == 8.0) & (synthetic_grid['MH'] == 0.0)
    ].copy()

    dm = 10.0  # distance modulus
    n_stars = len(iso)

    mags = iso[['Gmag', 'G_BPmag', 'G_RPmag']].values + dm
    errors = np.full((n_stars, 3), 0.05)
    noise = rng.normal(0, 0.05, mags.shape)

    observed_mags = mags + noise
    return observed_mags, errors
