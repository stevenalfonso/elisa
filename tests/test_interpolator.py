"""Tests for elisa.mcmc.interpolator."""

import numpy as np
import pytest

from elisa.mcmc.interpolator import IsochroneInterpolator


class TestIsochroneInterpolator:
    def test_init_from_dataframe(self, synthetic_grid):
        interp = IsochroneInterpolator(grid=synthetic_grid)
        assert interp.age_min == 8.0
        assert interp.age_max == 9.0
        assert interp.MH_min == 0.0
        assert interp.MH_max == 0.5

    def test_init_requires_grid_or_path(self):
        with pytest.raises(ValueError, match="Either 'grid' or 'grid_path'"):
            IsochroneInterpolator()

    def test_missing_band_raises(self, synthetic_grid):
        with pytest.raises(ValueError, match="Band .* not found"):
            IsochroneInterpolator(grid=synthetic_grid, photometry_bands=['Jmag'])

    def test_get_magnitudes_on_grid_point(self, synthetic_grid):
        interp = IsochroneInterpolator(grid=synthetic_grid)
        mags = interp.get_magnitudes(mass=1.0, logAge=8.0, MH=0.0)
        assert mags is not None
        assert mags.shape == (3,)
        assert np.all(np.isfinite(mags))

    def test_get_magnitudes_interpolated_age(self, synthetic_grid):
        interp = IsochroneInterpolator(grid=synthetic_grid)
        mags = interp.get_magnitudes(mass=1.0, logAge=8.5, MH=0.0)
        assert mags is not None
        # Should be between the two grid-point values
        mags_lo = interp.get_magnitudes(mass=1.0, logAge=8.0, MH=0.0)
        mags_hi = interp.get_magnitudes(mass=1.0, logAge=9.0, MH=0.0)
        for i in range(3):
            lo, hi = sorted([mags_lo[i], mags_hi[i]])
            assert lo <= mags[i] <= hi

    def test_get_magnitudes_out_of_range(self, synthetic_grid):
        interp = IsochroneInterpolator(grid=synthetic_grid)
        # Age below grid
        assert interp.get_magnitudes(mass=1.0, logAge=5.0, MH=0.0) is None
        # MH above grid
        assert interp.get_magnitudes(mass=1.0, logAge=8.0, MH=2.0) is None

    def test_get_magnitudes_vectorized_shape(self, synthetic_grid):
        interp = IsochroneInterpolator(grid=synthetic_grid)
        masses = np.array([0.5, 1.0, 2.0])
        result = interp.get_magnitudes_vectorized(masses, logAge=8.0, MH=0.0)
        assert result.shape == (3, 3)

    def test_get_magnitudes_vectorized_out_of_range(self, synthetic_grid):
        interp = IsochroneInterpolator(grid=synthetic_grid)
        masses = np.array([1.0, 2.0])
        result = interp.get_magnitudes_vectorized(masses, logAge=5.0, MH=0.0)
        assert np.all(np.isnan(result))

    def test_get_magnitudes_vectorized_matches_scalar(self, synthetic_grid):
        interp = IsochroneInterpolator(grid=synthetic_grid)
        masses = np.array([0.5, 1.0, 2.0])
        vec = interp.get_magnitudes_vectorized(masses, logAge=8.0, MH=0.0)
        for i, m in enumerate(masses):
            scalar = interp.get_magnitudes(m, logAge=8.0, MH=0.0)
            np.testing.assert_allclose(vec[i], scalar, atol=1e-10)

    def test_get_max_mass_at_age(self, synthetic_grid):
        interp = IsochroneInterpolator(grid=synthetic_grid)
        max_mass = interp.get_max_mass_at_age(logAge=8.0, MH=0.0)
        assert max_mass == 5.0

    def test_get_min_mass(self, synthetic_grid):
        interp = IsochroneInterpolator(grid=synthetic_grid)
        assert interp.get_min_mass() == 0.5
