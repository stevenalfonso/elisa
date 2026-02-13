"""
ELISA: Efficient Likelihood Inference for Stellar Ages

A Python package for Bayesian inference of stellar cluster parameters
(age, metallicity, distance, extinction) using isochrone fitting.

Main class:
    Elisa - The main interface for cluster parameter inference

Example usage:
    from elisa import Elisa

    elisa = Elisa()

    # Download or load isochrone grid
    grid = elisa.download_isochrones(...)
    # or
    grid = elisa.load_isochrone_grid('path/to/grid.parquet')

    # Setup components
    elisa.setup_interpolator(grid=grid)
    elisa.setup_extinction()

    # Setup posterior and run MCMC
    posterior = elisa.setup_logposterior(observed_mags, observed_errors, ...)
    init_params = posterior.get_initial_params()
    sampler, samples = elisa.run_mcmc(posterior, init_params, ...)

    # Analyze results
    results = elisa.get_results_summary(samples)
    elisa.plot_corner(samples)
"""

from .mcmc.elisa import ElisaClusterInference
#from .interpolator import IsochroneInterpolator
#from .extinction import Extinction, distance_modulus_to_distance, distance_to_distance_modulus
#from .likelihood import ElisaPosterior

__version__ = "0.1.0"
__author__ = "J. Alfonso"

__all__ = [
    "ElisaClusterInference",
    "IsochroneInterpolator",
    "Extinction",
    "ElisaPosterior",
    "distance_modulus_to_distance",
    "distance_to_distance_modulus",
]
