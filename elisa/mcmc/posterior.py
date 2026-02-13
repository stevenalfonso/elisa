"""
Likelihood Function for Cluster Parameter Inference

Implements the Bayesian likelihood from von Hippel et al. (2006), Equation 2.

The likelihood compares observed photometry with model predictions:
    L = product over all stars and bands of:
        (1 / sqrt(2*pi*sigma^2)) * exp(-(x_obs - x_pred)^2 / (2*sigma^2))

In log form:
    log(L) = sum of:
        -0.5 * log(2*pi*sigma^2) - (x_obs - x_pred)^2 / (2*sigma^2)

Note: This module expects apparent magnitudes as input. Users should apply
their preferred extinction correction before passing data to this module.
"""

import numpy as np
from typing import Tuple, Dict, Union
from .utils import log_prior_gaussian, log_prior_uniform, log_likelihood


class Posterior:
    """
    Log-posterior function for cluster parameter inference.

    Combines likelihood and priors for use with MCMC samplers.
    Samples only the 4 cluster parameters: logAge, MH, dm, AV.
    Stellar masses are estimated internally for each star based on
    the observed G magnitude and current cluster parameters.

    Parameters
    ----------
    observed_mags : np.ndarray
        Observed magnitudes, shape (n_stars, 3) for [G, BP, RP]
    observed_errors : np.ndarray
        Photometric errors, shape (n_stars, 3)
    interpolator : IsochroneInterpolator
        Isochrone interpolator object
    extinction_law : GaiaExtinction
        Extinction law object
    prior_logAge : tuple
        (mean, std) for Gaussian prior on log(age), or (low, high) for uniform
    prior_MH : tuple
        (mean, std) for Gaussian prior on [M/H]
    prior_dm : tuple
        (mean, std) for Gaussian prior on distance modulus
    prior_AV : tuple
        (mean, std) for Gaussian prior on A_V (truncated at 0)
    prior_type : str or dict
        Prior type for cluster parameters. Can be a single string
        ('gaussian' or 'uniform') applied to all parameters, or a dict
        with keys 'logAge', 'MH', 'dm', 'AV' mapping to 'gaussian' or
        'uniform' for per-parameter control. Example:
        {'logAge': 'uniform', 'MH': 'gaussian', 'dm': 'gaussian', 'AV': 'gaussian'}
    error_floor : float
        Minimum photometric error (default: 0.02 mag). Prevents unrealistically
        tight constraints from very small reported errors.
    intrinsic_scatter : float
        Additional scatter added in quadrature (default: 0.05 mag). Accounts for
        model imperfections, unresolved binaries, stellar rotation, activity, etc.
    """

    def __init__(
        self,
        observed_mags: np.ndarray,
        observed_errors: np.ndarray,
        interpolator,
        extinction_law,
        prior_logAge: Tuple[float, float] = (9.0, 0.5),
        prior_MH: Tuple[float, float] = (0.0, 0.3),
        prior_dm: Tuple[float, float] = (10.0, 1.0),
        prior_AV: Tuple[float, float] = (0.1, 0.2),
        prior_type: Union[str, Dict[str, str]] = 'gaussian',
        error_floor: float = 0.02,
        intrinsic_scatter: float = 0.05,
        correct_extinction: bool = False
    ):
        self.observed_mags = observed_mags
        self.observed_errors = observed_errors
        self.n_stars = observed_mags.shape[0]

        self.interpolator = interpolator
        self.extinction_law = extinction_law

        self.prior_logAge = prior_logAge
        self.prior_MH = prior_MH
        self.prior_dm = prior_dm
        self.prior_AV = prior_AV
        # Normalize prior_type to a per-parameter dict
        param_names = ['logAge', 'MH', 'dm', 'AV']
        if isinstance(prior_type, str):
            self.prior_type = {p: prior_type for p in param_names}
        else:
            self.prior_type = {p: prior_type.get(p, 'gaussian') for p in param_names}
        self.correct_extinction = correct_extinction

        self.error_floor = error_floor  # minimum photometric error
        self.intrinsic_scatter = intrinsic_scatter  # model scatter (binaries, rotation, etc.)

        # Bounds for parameters
        self.bounds = {
            'logAge': (interpolator.age_min, interpolator.age_max),
            'MH': (interpolator.MH_min, interpolator.MH_max),
            'dm': (0.1, 25.0),  # distance modulus
            'AV': (0.0, 5.0),   # extinction
            'mass': (interpolator.mass_min, interpolator.mass_max)
        }


    def log_prior(self, params: np.ndarray) -> float:
        """
        Compute log-prior for cluster parameters.

        Parameters
        ----------
        params : np.ndarray
            Cluster parameter vector: [logAge, MH, dm, AV]

        Returns
        -------
        float
            Log-prior probability
        """
        logAge, MH, dm, AV = params

        if not (self.bounds['logAge'][0] <= logAge <= self.bounds['logAge'][1]):
            return -np.inf
        if not (self.bounds['MH'][0] <= MH <= self.bounds['MH'][1]):
            return -np.inf
        if not (self.bounds['dm'][0] <= dm <= self.bounds['dm'][1]):
            return -np.inf
        if not (self.bounds['AV'][0] <= AV <= self.bounds['AV'][1]):
            return -np.inf

        log_p = 0.0

        # Priors on cluster parameters (per-parameter prior type)
        for value, prior_params, name in [(logAge, self.prior_logAge, 'logAge'), 
                                          (MH, self.prior_MH, 'MH'), 
                                          (dm, self.prior_dm, 'dm'), 
                                          (AV, self.prior_AV, 'AV'),]:
            
            if self.prior_type[name] == 'gaussian':
                log_p += log_prior_gaussian(value, *prior_params)
            else:  # uniform
                log_p += log_prior_uniform(value, *prior_params)

        return log_p


    def estimate_masses(self, logAge: float, MH: float, dm: float, AV: float) -> np.ndarray:
        """
        Estimate stellar masses from observed magnitudes given cluster parameters.

        For each star, finds the mass that minimizes the chi-squared across
        all photometric bands (G, BP, RP), not just G magnitude.

        Parameters
        ----------
        logAge : float
            Log10 of age in years
        MH : float
            Metallicity [M/H]
        dm : float
            Distance modulus
        AV : float
            V-band extinction

        Returns
        -------
        np.ndarray
            Estimated masses for each star
        """
        # Build a grid of masses and compute apparent magnitudes for each
        mass_grid = np.linspace(self.bounds['mass'][0], self.bounds['mass'][1], 300)

        # Get absolute magnitudes for the mass grid
        abs_mags_grid = self.interpolator.get_magnitudes_vectorized(mass_grid, logAge, MH)

        # Apply extinction and distance to get apparent magnitudes
        bp_rp_0 = abs_mags_grid[:, 1] - abs_mags_grid[:, 2]
        if self.correct_extinction:
            A_G, A_BP, A_RP = self.extinction_law.get_extinction(AV, bp_rp_0)
        else:
            A_G, A_BP, A_RP = 0.0, 0.0, 0.0

        app_mags_grid = abs_mags_grid.copy()
        app_mags_grid[:, 0] += dm + A_G
        app_mags_grid[:, 1] += dm + A_BP
        app_mags_grid[:, 2] += dm + A_RP

        # Find valid grid points (no NaN)
        valid = ~np.any(np.isnan(app_mags_grid), axis=1)
        mass_grid_valid = mass_grid[valid]
        app_mags_valid = app_mags_grid[valid]

        if len(mass_grid_valid) < 2:
            # Fallback to simple mass estimate based on G
            G_obs = self.observed_mags[:, 0]
            G_abs_approx = G_obs - dm - 0.8 * AV
            return np.clip(10 ** ((4.83 - G_abs_approx) / 7.5), self.bounds['mass'][0] + 0.01, self.bounds['mass'][1] - 0.01)

        # For each star, find the mass that minimizes chi-squared across all bands
        masses = np.zeros(self.n_stars)

        for i in range(self.n_stars):
            obs = self.observed_mags[i]  # shape (3,)
            err = self.observed_errors[i]  # shape (3,)

            # Use error floor for mass estimation
            err_eff = np.maximum(err, 0.02)

            # Compute chi-squared for each mass in the grid
            residuals = app_mags_valid - obs  # shape (n_valid, 3)
            chi2 = np.sum((residuals / err_eff) ** 2, axis=1)

            # Find mass with minimum chi-squared
            idx_best = np.argmin(chi2)
            masses[i] = mass_grid_valid[idx_best]

        # Clip to valid mass range
        masses = np.clip(
            masses,
            self.bounds['mass'][0] + 0.01,
            self.bounds['mass'][1] - 0.01
        )

        return masses

    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood for cluster parameters.

        Masses are estimated internally based on the observed G magnitudes
        and the current cluster parameters.

        Parameters
        ----------
        params : np.ndarray
            Cluster parameter vector: [logAge, MH, dm, AV]

        Returns
        -------
        float
            Log-likelihood
        """
        logAge, MH, dm, AV = params

        # Estimate masses given current cluster parameters
        masses = self.estimate_masses(logAge, MH, dm, AV)

        # Get predicted absolute magnitudes from isochrones
        abs_mags = self.interpolator.get_magnitudes_vectorized(masses, logAge, MH)

        if np.any(np.isnan(abs_mags)):
            return -np.inf

        # Apply extinction and distance
        bp_rp_0 = abs_mags[:, 1] - abs_mags[:, 2]
        if self.correct_extinction:
            A_G, A_BP, A_RP = self.extinction_law.get_extinction(AV, bp_rp_0)
        else:
            A_G, A_BP, A_RP = 0.0, 0.0, 0.0

        # Convert to apparent magnitudes
        predicted_mags = abs_mags.copy()
        predicted_mags[:, 0] += dm + A_G
        predicted_mags[:, 1] += dm + A_BP
        predicted_mags[:, 2] += dm + A_RP

        return log_likelihood(
            self.observed_mags,
            self.observed_errors,
            predicted_mags,
            error_floor=self.error_floor,
            intrinsic_scatter=self.intrinsic_scatter
        )


    def __call__(self, params: np.ndarray) -> float:
        """
        Compute log-posterior for cluster parameters.

        Parameters
        ----------
        params : np.ndarray
            Cluster parameter vector: [logAge, MH, dm, AV]

        Returns
        -------
        float
            Log-posterior probability
        """
        # Validate input
        if len(params) != 4:
            raise ValueError(f"Expected 4 parameters [logAge, MH, dm, AV], got {len(params)}")

        # Prior
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf

        # Likelihood
        ll = self.log_likelihood(params)
        if not np.isfinite(ll):
            return -np.inf

        return lp + ll


    def get_initial_params(
        self,
        logAge_init: float = None,
        MH_init: float = None,
        dm_init: float = None,
        AV_init: float = None
    ) -> np.ndarray:
        """
        Generate initial cluster parameter vector.

        Parameters
        ----------
        logAge_init, MH_init, dm_init, AV_init : float, optional
            Initial cluster parameters. If None, uses prior mean.

        Returns
        -------
        np.ndarray
            Initial parameter vector of shape (4,): [logAge, MH, dm, AV]
        """
        if logAge_init is None:
            logAge_init = self.prior_logAge[0]
        if MH_init is None:
            MH_init = self.prior_MH[0]
        if dm_init is None:
            dm_init = self.prior_dm[0]
        if AV_init is None:
            AV_init = max(0.01, self.prior_AV[0])

        return np.array([logAge_init, MH_init, dm_init, AV_init])
