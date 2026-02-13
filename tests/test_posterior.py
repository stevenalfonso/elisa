"""Tests for elisa.mcmc.posterior."""

import numpy as np
import pytest

from elisa.mcmc.posterior import Posterior
from elisa.mcmc.interpolator import IsochroneInterpolator
from elisa.mcmc.extinction import Extinction


@pytest.fixture
def posterior_uniform(synthetic_grid, observed_data):
    observed_mags, observed_errors = observed_data
    interp = IsochroneInterpolator(grid=synthetic_grid)
    ext = Extinction()
    return Posterior(
        observed_mags=observed_mags,
        observed_errors=observed_errors,
        interpolator=interp,
        extinction_law=ext,
        prior_logAge=(7.0, 10.0),
        prior_MH=(-1.0, 1.0),
        prior_dm=(5.0, 15.0),
        prior_AV=(0.0, 3.0),
        prior_type='uniform',
    )


@pytest.fixture
def posterior_gaussian(synthetic_grid, observed_data):
    observed_mags, observed_errors = observed_data
    interp = IsochroneInterpolator(grid=synthetic_grid)
    ext = Extinction()
    return Posterior(
        observed_mags=observed_mags,
        observed_errors=observed_errors,
        interpolator=interp,
        extinction_law=ext,
        prior_logAge=(8.0, 0.5),
        prior_MH=(0.0, 0.3),
        prior_dm=(10.0, 1.0),
        prior_AV=(0.1, 0.2),
        prior_type='gaussian',
    )


@pytest.fixture
def posterior_mixed(synthetic_grid, observed_data):
    observed_mags, observed_errors = observed_data
    interp = IsochroneInterpolator(grid=synthetic_grid)
    ext = Extinction()
    return Posterior(
        observed_mags=observed_mags,
        observed_errors=observed_errors,
        interpolator=interp,
        extinction_law=ext,
        prior_logAge=(7.0, 10.0),
        prior_MH=(0.0, 0.3),
        prior_dm=(10.0, 1.0),
        prior_AV=(0.1, 0.2),
        prior_type={'logAge': 'uniform', 'MH': 'gaussian', 'dm': 'gaussian', 'AV': 'gaussian'},
    )


# ---------------------------------------------------------------------------
# Prior tests
# ---------------------------------------------------------------------------

class TestLogPrior:
    def test_uniform_in_bounds(self, posterior_uniform):
        params = np.array([8.0, 0.0, 10.0, 0.1])
        lp = posterior_uniform.log_prior(params)
        assert np.isfinite(lp)

    def test_uniform_out_of_bounds_age(self, posterior_uniform):
        params = np.array([5.0, 0.0, 10.0, 0.1])
        assert posterior_uniform.log_prior(params) == -np.inf

    def test_uniform_negative_AV(self, posterior_uniform):
        params = np.array([8.0, 0.0, 10.0, -0.1])
        assert posterior_uniform.log_prior(params) == -np.inf

    def test_gaussian_peak(self, posterior_gaussian):
        params = np.array([8.0, 0.0, 10.0, 0.1])
        lp_peak = posterior_gaussian.log_prior(params)
        params_off = np.array([9.0, 0.5, 12.0, 1.0])
        lp_off = posterior_gaussian.log_prior(params_off)
        assert lp_peak > lp_off

    def test_mixed_prior_works(self, posterior_mixed):
        params = np.array([8.0, 0.0, 10.0, 0.1])
        lp = posterior_mixed.log_prior(params)
        assert np.isfinite(lp)


# ---------------------------------------------------------------------------
# Full posterior (__call__)
# ---------------------------------------------------------------------------

class TestPosteriorCall:
    def test_finite_at_good_params(self, posterior_uniform):
        params = np.array([8.0, 0.0, 10.0, 0.1])
        lp = posterior_uniform(params)
        assert np.isfinite(lp)

    def test_neginf_at_bad_params(self, posterior_uniform):
        params = np.array([5.0, 0.0, 10.0, 0.1])
        assert posterior_uniform(params) == -np.inf

    def test_wrong_param_count_raises(self, posterior_uniform):
        with pytest.raises(ValueError, match="Expected 4"):
            posterior_uniform(np.array([8.0, 0.0]))

    def test_better_params_higher_posterior(self, posterior_uniform):
        """The generating parameters should score better than far-off ones."""
        good = np.array([8.0, 0.0, 10.0, 0.1])
        bad = np.array([9.0, 0.4, 7.0, 2.0])
        lp_good = posterior_uniform(good)
        lp_bad = posterior_uniform(bad)
        # Both should be finite for a fair comparison
        if np.isfinite(lp_bad):
            assert lp_good > lp_bad


# ---------------------------------------------------------------------------
# get_initial_params
# ---------------------------------------------------------------------------

class TestGetInitialParams:
    def test_returns_correct_values(self, posterior_uniform):
        params = posterior_uniform.get_initial_params(
            logAge_init=8.5, MH_init=0.1, dm_init=12.0, AV_init=0.3
        )
        np.testing.assert_array_equal(params, [8.5, 0.1, 12.0, 0.3])

    def test_defaults_from_prior(self, posterior_uniform):
        params = posterior_uniform.get_initial_params()
        assert len(params) == 4
        assert params[3] > 0  # AV clipped to > 0

    def test_shape(self, posterior_uniform):
        params = posterior_uniform.get_initial_params()
        assert params.shape == (4,)


# ---------------------------------------------------------------------------
# estimate_masses
# ---------------------------------------------------------------------------

class TestEstimateMasses:
    def test_output_shape(self, posterior_uniform):
        masses = posterior_uniform.estimate_masses(8.0, 0.0, 10.0, 0.1)
        assert masses.shape == (posterior_uniform.n_stars,)

    def test_masses_within_bounds(self, posterior_uniform):
        masses = posterior_uniform.estimate_masses(8.0, 0.0, 10.0, 0.1)
        assert np.all(masses >= posterior_uniform.bounds['mass'][0])
        assert np.all(masses <= posterior_uniform.bounds['mass'][1])
