"""Tests for elisa.mcmc.elisa (ElisaClusterInference)."""

import numpy as np
import pytest

from elisa import ElisaClusterInference


# ---------------------------------------------------------------------------
# Gelman-Rubin diagnostic
# ---------------------------------------------------------------------------

class TestGelmanRubin:
    def test_converged_chains(self):
        """Identical distributions across chains should give R-hat ~ 1."""
        rng = np.random.default_rng(0)
        chains = rng.normal(size=(4, 500, 2))
        elisa = ElisaClusterInference()
        R = elisa.get_gelman_rubin(chains)
        assert R.shape == (2,)
        np.testing.assert_allclose(R, 1.0, atol=0.1)

    def test_diverged_chains(self):
        """Chains with very different means should give R-hat >> 1."""
        rng = np.random.default_rng(0)
        chains = np.zeros((4, 500, 1))
        for i in range(4):
            chains[i, :, 0] = rng.normal(loc=i * 10, scale=0.1, size=500)
        elisa = ElisaClusterInference()
        R = elisa.get_gelman_rubin(chains)
        assert R[0] > 2.0

    def test_single_parameter(self):
        rng = np.random.default_rng(0)
        chains = rng.normal(size=(8, 200, 1))
        elisa = ElisaClusterInference()
        R = elisa.get_gelman_rubin(chains)
        assert R.shape == (1,)


# ---------------------------------------------------------------------------
# get_results_summary
# ---------------------------------------------------------------------------

class TestGetResultsSummary:
    def test_keys_present(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=[8.0, 0.0, 10.0, 0.1], scale=0.01, size=(500, 4))
        samples[:, 3] = np.abs(samples[:, 3])  # AV positive
        elisa = ElisaClusterInference()
        results = elisa.get_results_summary(samples)
        for key in ['logAge', 'MH', 'dm', 'AV', 'distance_pc', 'age_myr']:
            assert key in results

    def test_median_close_to_input(self):
        rng = np.random.default_rng(0)
        true_vals = [8.0, 0.0, 10.0, 0.1]
        samples = rng.normal(loc=true_vals, scale=0.01, size=(1000, 4))
        samples[:, 3] = np.abs(samples[:, 3])
        elisa = ElisaClusterInference()
        results = elisa.get_results_summary(samples)
        for i, name in enumerate(['logAge', 'MH', 'dm', 'AV']):
            assert results[name]['median'] == pytest.approx(true_vals[i], abs=0.05)

    def test_errors_positive(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=[8.0, 0.0, 10.0, 0.1], scale=0.1, size=(500, 4))
        elisa = ElisaClusterInference()
        results = elisa.get_results_summary(samples)
        for name in ['logAge', 'MH', 'dm', 'AV']:
            assert results[name]['err_low'] > 0
            assert results[name]['err_high'] > 0

    def test_derived_distance(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=[8.0, 0.0, 10.0, 0.1], scale=0.001, size=(500, 4))
        elisa = ElisaClusterInference()
        results = elisa.get_results_summary(samples)
        # dm=10 => d = 10^(10/5 + 1) = 1000 pc
        assert results['distance_pc'] == pytest.approx(1000.0, rel=0.01)


# ---------------------------------------------------------------------------
# setup_logposterior
# ---------------------------------------------------------------------------

class TestSetupLogposterior:
    def test_returns_callable(self, synthetic_grid, observed_data):
        observed_mags, observed_errors = observed_data
        elisa = ElisaClusterInference()
        posterior = elisa.setup_logposterior(
            grid=synthetic_grid,
            observed_mags=observed_mags,
            observed_errors=observed_errors,
            prior_logAge=(7.0, 10.0),
            prior_MH=(0.0, 0.3),
            prior_dm=(10.0, 1.0),
            prior_AV=(0.1, 0.2),
            prior_type='gaussian',
        )
        params = np.array([8.0, 0.0, 10.0, 0.1])
        result = posterior(params)
        assert np.isfinite(result)

    def test_empty_grid_raises(self, observed_data):
        import pandas as pd
        observed_mags, observed_errors = observed_data
        elisa = ElisaClusterInference()
        with pytest.raises(ValueError, match="empty"):
            elisa.setup_logposterior(
                grid=pd.DataFrame(),
                observed_mags=observed_mags,
                observed_errors=observed_errors,
            )


# ---------------------------------------------------------------------------
# run_mcmc (short smoke test)
# ---------------------------------------------------------------------------

class TestRunMCMC:
    def test_smoke(self, synthetic_grid, observed_data):
        """Short MCMC run to verify it produces a sampler with correct shape."""
        observed_mags, observed_errors = observed_data
        elisa = ElisaClusterInference()
        posterior = elisa.setup_logposterior(
            grid=synthetic_grid,
            observed_mags=observed_mags,
            observed_errors=observed_errors,
            prior_logAge=(7.0, 10.0),
            prior_MH=(-1.0, 1.0),
            prior_dm=(5.0, 15.0),
            prior_AV=(0.0, 3.0),
            prior_type='uniform',
        )
        init_params = posterior.get_initial_params(
            logAge_init=8.0, MH_init=0.0, dm_init=10.0, AV_init=0.1
        )
        sampler = elisa.run_mcmc(
            log_posterior=posterior,
            init_params=init_params,
            n_walkers=16,
            n_steps=20,
            progress=False,
        )
        chain = sampler.get_chain()
        assert chain.shape == (20, 16, 4)
