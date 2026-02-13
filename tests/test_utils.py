"""Tests for elisa.mcmc.utils (priors and likelihood)."""

import numpy as np
import pytest

from elisa.mcmc.utils import (
    log_likelihood,
    log_prior_gaussian,
    log_prior_uniform,
    log_prior_imf_chabrier,
    log_prior_imf_chabrier_vectorized,
)


# ---------------------------------------------------------------------------
# log_prior_gaussian
# ---------------------------------------------------------------------------

class TestLogPriorGaussian:
    def test_peak_at_mean(self):
        assert log_prior_gaussian(0.0, 0.0, 1.0) == 0.0

    def test_symmetric(self):
        lp_pos = log_prior_gaussian(1.0, 0.0, 1.0)
        lp_neg = log_prior_gaussian(-1.0, 0.0, 1.0)
        assert lp_pos == pytest.approx(lp_neg)

    def test_decreases_away_from_mean(self):
        lp_close = log_prior_gaussian(0.1, 0.0, 1.0)
        lp_far = log_prior_gaussian(3.0, 0.0, 1.0)
        assert lp_close > lp_far


# ---------------------------------------------------------------------------
# log_prior_uniform
# ---------------------------------------------------------------------------

class TestLogPriorUniform:
    def test_inside_bounds(self):
        assert log_prior_uniform(5.0, 0.0, 10.0) == 0.0

    def test_at_bounds(self):
        assert log_prior_uniform(0.0, 0.0, 10.0) == 0.0
        assert log_prior_uniform(10.0, 0.0, 10.0) == 0.0

    def test_outside_bounds(self):
        assert log_prior_uniform(-0.1, 0.0, 10.0) == -np.inf
        assert log_prior_uniform(10.1, 0.0, 10.0) == -np.inf


# ---------------------------------------------------------------------------
# log_prior_imf_chabrier
# ---------------------------------------------------------------------------

class TestLogPriorIMF:
    def test_negative_mass(self):
        assert log_prior_imf_chabrier(-1.0) == -np.inf

    def test_zero_mass(self):
        assert log_prior_imf_chabrier(0.0) == -np.inf

    def test_low_mass_finite(self):
        assert np.isfinite(log_prior_imf_chabrier(0.5))

    def test_high_mass_finite(self):
        assert np.isfinite(log_prior_imf_chabrier(5.0))

    def test_higher_mass_lower_prior(self):
        """IMF should favor lower masses at the high-mass end."""
        lp_2 = log_prior_imf_chabrier(2.0)
        lp_10 = log_prior_imf_chabrier(10.0)
        assert lp_2 > lp_10

    def test_vectorized_matches_scalar(self):
        masses = np.array([0.3, 0.5, 1.0, 2.0, 5.0])
        scalar_sum = sum(log_prior_imf_chabrier(m) for m in masses)
        vec_sum = log_prior_imf_chabrier_vectorized(masses)
        assert vec_sum == pytest.approx(scalar_sum)

    def test_vectorized_negative_mass(self):
        assert log_prior_imf_chabrier_vectorized(np.array([1.0, -0.5])) == -np.inf


# ---------------------------------------------------------------------------
# log_likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    def test_perfect_match(self):
        """Identical observations and predictions should give the highest likelihood."""
        mags = np.array([[10.0, 11.0, 9.5]])
        errors = np.array([[0.05, 0.05, 0.05]])
        ll = log_likelihood(mags, errors, mags)
        assert np.isfinite(ll)

    def test_worse_with_offset(self):
        mags = np.array([[10.0, 11.0, 9.5]])
        errors = np.array([[0.05, 0.05, 0.05]])
        predicted_close = mags + 0.01
        predicted_far = mags + 1.0
        ll_close = log_likelihood(mags, errors, predicted_close)
        ll_far = log_likelihood(mags, errors, predicted_far)
        assert ll_close > ll_far

    def test_nan_prediction_returns_neginf(self):
        mags = np.array([[10.0, 11.0, 9.5]])
        errors = np.array([[0.05, 0.05, 0.05]])
        pred = np.array([[10.0, np.nan, 9.5]])
        assert log_likelihood(mags, errors, pred) == -np.inf

    def test_error_floor_applied(self):
        """Tiny errors should be floored, preventing extreme log-likelihood values."""
        mags = np.array([[10.0, 11.0, 9.5]])
        tiny_errors = np.array([[1e-10, 1e-10, 1e-10]])
        pred = mags + 0.01
        ll = log_likelihood(mags, tiny_errors, pred, error_floor=0.02)
        assert np.isfinite(ll)

    def test_multiple_stars(self):
        n = 50
        rng = np.random.default_rng(0)
        mags = rng.uniform(8, 16, (n, 3))
        errors = np.full((n, 3), 0.05)
        ll = log_likelihood(mags, errors, mags)
        assert np.isfinite(ll)
