"""Tests for elisa.mcmc.extinction."""

import numpy as np
import pytest

from elisa.mcmc.extinction import (
    Extinction,
    distance_modulus_to_distance,
    distance_to_distance_modulus,
)


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

class TestDistanceConversions:
    def test_roundtrip(self):
        d = 136.0
        dm = distance_to_distance_modulus(d)
        d_back = distance_modulus_to_distance(dm)
        assert d_back == pytest.approx(d)

    def test_known_value(self):
        # 10 pc => dm = 0
        assert distance_to_distance_modulus(10.0) == pytest.approx(0.0)
        assert distance_modulus_to_distance(0.0) == pytest.approx(10.0)

    def test_100pc(self):
        assert distance_to_distance_modulus(100.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Extinction (fixed coefficients)
# ---------------------------------------------------------------------------

class TestExtinctionFixed:
    def test_zero_extinction(self):
        ext = Extinction(coefficients='fixed')
        A_G, A_BP, A_RP = ext.get_extinction(0.0)
        assert A_G == 0.0
        assert A_BP == 0.0
        assert A_RP == 0.0

    def test_positive_extinction(self):
        ext = Extinction(coefficients='fixed')
        A_G, A_BP, A_RP = ext.get_extinction(1.0)
        assert A_G > 0
        assert A_BP > 0
        assert A_RP > 0

    def test_ordering(self):
        """A_BP > A_G > A_RP for fixed coefficients."""
        ext = Extinction(coefficients='fixed')
        A_G, A_BP, A_RP = ext.get_extinction(1.0)
        assert A_BP > A_G > A_RP

    def test_linearity(self):
        ext = Extinction(coefficients='fixed')
        A_G_1, _, _ = ext.get_extinction(1.0)
        A_G_2, _, _ = ext.get_extinction(2.0)
        assert A_G_2 == pytest.approx(2 * A_G_1)

    def test_reddening(self):
        ext = Extinction(coefficients='fixed')
        E = ext.get_reddening(1.0)
        A_G, A_BP, A_RP = ext.get_extinction(1.0)
        assert E == pytest.approx(A_BP - A_RP)

    def test_AV_from_reddening_roundtrip(self):
        ext = Extinction(coefficients='fixed')
        AV = 0.75
        E = ext.get_reddening(AV)
        AV_back = ext.A_V_from_E_BP_RP(E)
        assert AV_back == pytest.approx(AV)


# ---------------------------------------------------------------------------
# apply_extinction_and_distance
# ---------------------------------------------------------------------------

class TestApplyExtinctionAndDistance:
    def test_output_shape(self):
        ext = Extinction()
        abs_mags = np.array([[1.0, 1.5, 0.8], [2.0, 2.5, 1.8]])
        app_mags = ext.apply_extinction_and_distance(abs_mags, distance_modulus=10.0, A_V=0.5)
        assert app_mags.shape == abs_mags.shape

    def test_zero_extinction_zero_dm(self):
        ext = Extinction()
        abs_mags = np.array([[1.0, 1.5, 0.8]])
        app_mags = ext.apply_extinction_and_distance(abs_mags, distance_modulus=0.0, A_V=0.0)
        np.testing.assert_allclose(app_mags, abs_mags)

    def test_apparent_brighter_not_possible(self):
        """Apparent mags should always be >= absolute mags for positive dm and AV."""
        ext = Extinction()
        abs_mags = np.array([[1.0, 1.5, 0.8]])
        app_mags = ext.apply_extinction_and_distance(abs_mags, distance_modulus=5.0, A_V=0.3)
        assert np.all(app_mags >= abs_mags)

    def test_does_not_modify_input(self):
        ext = Extinction()
        abs_mags = np.array([[1.0, 1.5, 0.8]])
        original = abs_mags.copy()
        ext.apply_extinction_and_distance(abs_mags, distance_modulus=10.0, A_V=1.0)
        np.testing.assert_array_equal(abs_mags, original)
