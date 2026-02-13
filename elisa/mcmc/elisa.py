import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import emcee
from multiprocessing import Pool
import matplotlib.pyplot as plt

from .interpolator import IsochroneInterpolator
from .extinction import Extinction, distance_modulus_to_distance
from .posterior import Posterior


class ElisaClusterInference:

    def __init__(self):
        self.interpolator = None
        self.extinction_law = None
        self.logposterior = None


    # def setup_interpolator(self, grid=None, **kwargs):
    #     self.interpolator = IsochroneInterpolator(grid=grid, **kwargs)
    #     return self.interpolator
    

    # def setup_extinction(self, **kwargs):
    #     self.extinction_law = Extinction(**kwargs)
    #     return self.extinction_law
    

    def setup_logposterior(self,
                           grid: pd.DataFrame,
                           observed_mags: np.ndarray, 
                           observed_errors: np.ndarray, 
                           **kwargs):
        """
        Set up the log-posterior function for MCMC sampling.

        Parameters
        ----------
        grid : pd.DataFrame
            Isochrone grid with columns including 'logAge', 'MH', 'mass', 'G_mag', 'BP_mag', 'RP_mag'
        observed_mags : np.ndarray
            Observed magnitudes, shape (n_stars, 3) for [G, BP, RP]
        observed_errors : np.ndarray
            Photometric errors, shape (n_stars, 3)
        **kwargs : dict
            Additional arguments passed to ElisaPosterior:

            - prior_logAge : tuple, (mean, std) or (low, high)
            - prior_MH : tuple, (mean, std) or (low, high)
            - prior_dm : tuple, (mean, std) or (low, high)
            - prior_AV : tuple, (mean, std) or (low, high)
            - prior_type : str or dict, 'gaussian'/'uniform' for all params,
              or dict like {'logAge': 'uniform', 'MH': 'gaussian', ...}

        Returns
        -------
        ElisaPosterior
            The log-posterior function object
        """

        # if self.interpolator is None:
        #     raise RuntimeError("Interpolator must be initialized first. Call setup_interpolator().")
        
        # if self.extinction_law is None:
        #     raise RuntimeError("Extinction law must be initialized first. Call setup_extinction().")
        
        if grid.empty:
            raise ValueError("Isochrone grid is empty. Provide a valid grid DataFrame.")
        
        # Only pass kwargs that IsochroneInterpolator accepts
        interp_keys = {'grid_path', 'photometry_bands'}
        interp_kwargs = {k: v for k, v in kwargs.items() if k in interp_keys}
        self.interpolator = IsochroneInterpolator(grid=grid, **interp_kwargs)

        # Only pass kwargs that Extinction accepts
        ext_keys = {'coefficients'}
        ext_kwargs = {k: v for k, v in kwargs.items() if k in ext_keys}
        self.extinction_law = Extinction(**ext_kwargs)

        self.logposterior = Posterior(
            observed_mags=observed_mags,
            observed_errors=observed_errors,
            interpolator=self.interpolator,
            extinction_law=self.extinction_law,
            **kwargs
        )

        return self.logposterior


    def download_isochrones(self,
                            logage_range: tuple = (6.0, 10.5, 0.05), # (min, max, step)
                            MH_range: tuple = (-2.5, 1.0, 0.1),      # (min, max, step)
                            photsys:str = 'gaiaEDR3',
                            output_dir:str = './isochrone_data',
                            output_name:str = 'parsec_gaia_grid',
                            save_file:str = False) -> pd.DataFrame:

        from .download import download_isochrone_grid

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate grid size
        n_ages = int((logage_range[1] - logage_range[0]) / logage_range[2]) + 1
        n_metals = int((MH_range[1] - MH_range[0]) / MH_range[2]) + 1
        print(f"Grid size: {n_ages} ages x {n_metals} metallicities = {n_ages * n_metals} isochrones\n")
        
        isochrones_slim = download_isochrone_grid(logage_range, MH_range, photsys)

        if isochrones_slim.empty:
            raise RuntimeError("Downloaded isochrone grid is empty. Try again, check internet or use another set of ranges!")
            
        if save_file:
            parquet_path = os.path.join(output_dir, f"{output_name}.parquet")
            isochrones_slim.to_parquet(parquet_path, index=False)
            parquet_size = os.path.getsize(parquet_path) / (1024**2)
            print(f"Saved to: {parquet_path} ({parquet_size:.1f} MB)")
            
            metadata = {
                'logage_min': logage_range[0],
                'logage_max': logage_range[1],
                'logage_step': logage_range[2],
                'MH_min': MH_range[0],
                'MH_max': MH_range[1],
                'MH_step': MH_range[2],
                'photsys': photsys,
                'n_isochrones': n_ages * n_metals,
                'n_rows': len(isochrones_slim),
                'columns': list(isochrones_slim.columns),
                'unique_logages': sorted(isochrones_slim['logAge'].unique().tolist()),
                'unique_MH': sorted(isochrones_slim['MH'].unique().tolist()),
            }
            
            metadata_path = os.path.join(output_dir, f"{output_name}_metadata.txt")
            with open(metadata_path, 'w') as f:
                f.write("PARSEC Isochrone Grid Metadata\n")
                f.write("="*40 + "\n\n")
                for key, value in metadata.items():
                    if isinstance(value, list) and len(value) > 10:
                        f.write(f"{key}: [{value[0]}, {value[1]}, ..., {value[-1]}] ({len(value)} values)\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            print(f"Saved to: {metadata_path}")
            print()

        return isochrones_slim
    

    def load_isochrone_grid(self, filepath: str) -> pd.DataFrame:
        """
        Load a previously saved isochrone grid.
        
        Parameters
        ----------
        filepath : str
            Path to the parquet or csv file
        
        Returns
        -------
        pd.DataFrame
            The isochrone grid
        """
        if filepath.endswith('.parquet'):
            return pd.read_parquet(filepath)
        elif filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            raise ValueError("File must be .parquet or .csv")
        
    
    def run_mcmc(self,
        log_posterior,
        init_params: np.ndarray,
        n_walkers: int = 32,
        n_steps: int = 1000,
        progress: bool = True,
        parallel: bool = False,
        n_cores: int = 4,
        moves: emcee.moves.Move = None,
        init_spread: float = 1e-4
    ) -> emcee.EnsembleSampler:
        """
        Run MCMC sampling using emcee for cluster parameters.

        Parameters
        ----------
        log_posterior : callable
            Log-posterior function that accepts [logAge, MH, dm, AV]
        init_params : np.ndarray
            Initial parameter vector of length 4: [logAge, MH, dm, AV]
            Typically from an optimization result (e.g., scipy.optimize)
        n_walkers : int
            Number of walkers. Default: 32. Must be >= 2 * ndim (8 for 4 params)
        n_steps : int
            Number of MCMC steps per walker
        progress : bool
            Show progress bar
        parallel : bool
            Use parallel processing
        n_cores : int
            Number of cores for parallel processing
        moves : emcee.moves.Move, optional
            Custom move proposal. Default: StretchMove
        init_spread : float
            Spread for initial walker positions around init_params

        Returns
        -------
        sampler : emcee.EnsembleSampler
            The sampler object with chain
        flat_samples : np.ndarray
            Flattened samples after burn-in, shape (n_samples, 4)
        """
        # Validate init_params has exactly 4 parameters
        init_params = np.asarray(init_params)
        if len(init_params) != 4:
            raise ValueError(f"init_params must have exactly 4 parameters [logAge, MH, dm, AV], got {len(init_params)}")

        ndim = 4  # logAge, MH, dm, AV

        # Ensure n_walkers is valid for emcee (must be >= 2 * ndim and even)
        n_walkers = max(n_walkers, 2 * ndim)
        if n_walkers % 2 != 0:
            n_walkers += 1

        print(f"MCMC Setup:")
        print(f"  Parameters: {ndim} (logAge, MH, dm, AV)")
        print(f"  Walkers: {n_walkers}")
        print(f"  Steps: {n_steps}")

        # Initialize walker positions around init_params with small random spread
        # Shape: (n_walkers, ndim)
        pos = init_params + init_spread * np.random.randn(n_walkers, ndim)

        # Ensure AV (extinction) is non-negative
        pos[:, 3] = np.maximum(pos[:, 3], 1e-4)

        # Check that initial positions have finite log-posterior
        print("Checking initial positions...")
        valid_count = 0
        for i in range(n_walkers):
            lp = log_posterior(pos[i])
            if np.isfinite(lp):
                valid_count += 1
            else:
                # Reset to init_params with smaller perturbation
                pos[i] = init_params + 1e-5 * np.random.randn(ndim)
                pos[i, 3] = max(1e-4, pos[i, 3])

        print(f"  Valid initial positions: {valid_count}/{n_walkers}")

        if valid_count == 0:
            raise ValueError("All initial positions have -inf log-posterior. Check that init_params is within prior bounds.")

        # Set up and run sampler
        if parallel:
            with Pool(n_cores) as pool:
                sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, pool=pool, moves=moves, a=0.1)
                print(f"Running MCMC with {n_cores} cores...")
                sampler.run_mcmc(pos, n_steps, progress=progress)
        else:
            sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, moves=moves, a=0.1)
            print("Running MCMC...")
            sampler.run_mcmc(pos, n_steps, progress=progress)

        return sampler
    

    def get_gelman_rubin(self, chain):
        """Compute the Gelman-Rubin convergence diagnostic (R-hat) for MCMC chains.

        Parameters
        ----------
        chain : np.ndarray
            Array of shape (M, N, P) where M is the number of chains,
            N is the number of samples per chain, and P is the number of parameters.

        Returns
        -------
        np.ndarray
            R-hat values for each parameter. Values close to 1.0 indicate convergence.
        """
        ssq = np.var(chain, axis=1, ddof=1)
        W = np.mean(ssq, axis=0)
        theta_mean = np.mean(chain, axis=1)
        theta_m = np.mean(theta_mean, axis=0)
        M = chain.shape[0]
        N = chain.shape[1]
        B = N / (M - 1) * np.sum((theta_m - theta_mean)**2, axis=0)
        V = (N - 1) / N * W + 1 / N * B
        R = np.sqrt(V / W)
        return R


    def get_results_summary(
        self,
        flat_samples: np.ndarray,
        percentiles: List[float] = [16, 50, 84]
    ) -> Dict:
        """
        Summarize MCMC results for cluster parameters.

        Parameters
        ----------
        flat_samples : np.ndarray
            Flattened samples from MCMC, shape (n_samples, 4)
        percentiles : list
            Percentiles to compute (default: 16, 50, 84 for median +/- 1 sigma)

        Returns
        -------
        dict
            Summary statistics for each parameter including:
            - median, err_low, err_high for each parameter
            - distance in parsecs (derived from dm)
        """
        param_names = ['logAge', 'MH', 'dm', 'AV']
        results = {}

        print("=" * 60)
        print("CLUSTER PARAMETER RESULTS")
        print("=" * 60)

        for i, name in enumerate(param_names):
            pct = np.percentile(flat_samples[:, i], percentiles)
            median = pct[1]
            err_low = median - pct[0]
            err_high = pct[2] - median

            results[name] = {
                'median': median,
                'err_low': err_low,
                'err_high': err_high,
                'percentiles': pct
            }

            print(f"  {name:8s}: {median:.4f} (+{err_high:.4f} / -{err_low:.4f})")

        # Derived quantities
        dm_median = results['dm']['median']
        distance_pc = distance_modulus_to_distance(dm_median)
        results['distance_pc'] = distance_pc

        age_median = results['logAge']['median']
        age_myr = 10 ** (age_median - 6)  # Convert to Myr
        results['age_myr'] = age_myr

        print()
        print("Derived quantities:")
        print(f"  Distance: {distance_pc:.1f} pc")
        print(f"  Age: {age_myr:.1f} Myr")
        print("=" * 60)

        return results


    def plot_isochrone_fit(
        self,
        flat_samples: np.ndarray,
        observed_mags: np.ndarray,
        n_draws: int = 100,
        figsize: Tuple[float, float] = (8, 10)
    ):
        """
        Plot observed CMD with fitted isochrones from posterior samples.

        Parameters
        ----------
        flat_samples : np.ndarray
            Flattened samples, shape (n_samples, 4)
        observed_mags : np.ndarray
            Observed magnitudes, shape (n_stars, 3) for [G, BP, RP]
        n_draws : int
            Number of posterior draws to plot
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        fig, ax : matplotlib figure and axis
        """

        if self.interpolator is None or self.extinction_law is None:
            raise RuntimeError("Interpolator and Extinction must be initialized first.")

        fig, ax = plt.subplots(figsize=figsize)

        bp_rp_obs = observed_mags[:, 1] - observed_mags[:, 2]
        ax.scatter(bp_rp_obs, observed_mags[:, 0], c='k', s=5, alpha=0.5, label='Observed', zorder=10)

        # Random draws from posterior
        n_samples = len(flat_samples)
        idx_draws = np.random.choice(n_samples, size=min(n_draws, n_samples), replace=False)

        masses_iso = np.linspace(0.1, 10.0, 500)

        for idx in idx_draws:
            logAge, MH, dm, AV = flat_samples[idx]

            # Get isochrone
            abs_mags = self.interpolator.get_magnitudes_vectorized(masses_iso, logAge, MH)

            # Apply extinction and distance
            app_mags = self.extinction_law.apply_extinction_and_distance(abs_mags, dm, AV)

            bp_rp_iso = app_mags[:, 1] - app_mags[:, 2]
            valid = ~np.isnan(bp_rp_iso)
            ax.plot(bp_rp_iso[valid], app_mags[valid, 0], 'r-', alpha=0.05, lw=1)

        median_params = np.median(flat_samples, axis=0)
        logAge, MH, dm, AV = median_params

        abs_mags = self.interpolator.get_magnitudes_vectorized(masses_iso, logAge, MH)
        app_mags = self.extinction_law.apply_extinction_and_distance(abs_mags, dm, AV)

        bp_rp_iso = app_mags[:, 1] - app_mags[:, 2]
        valid = ~np.isnan(bp_rp_iso)
        ax.plot(bp_rp_iso[valid], app_mags[valid, 0], 'b-', lw=2, label=f'Median: logAge={logAge:.2f}, [M/H]={MH:.2f}')
        ax.set_xlabel('BP - RP (mag)')
        ax.set_ylabel('G (mag)')
        ax.invert_yaxis()
        ax.legend()
        ax.set_title('Fitted Isochrones from Posterior')

        return fig, ax
