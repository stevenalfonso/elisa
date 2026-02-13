"""
Isochrone Interpolator
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Tuple, Optional


class IsochroneInterpolator:
    """
    Interpolator for PARSEC isochrones over (mass, age, metallicity).
    
    Parameters
    ----------
    grid_path : str
        Path to the isochrone grid file (.csv or .parquet)
    photometry_bands : list, optional
        List of photometry columns to interpolate.
        Default: ['Gmag', 'G_BPmag', 'G_RPmag']
    """
    
    def __init__(
        self,
        grid: pd.DataFrame = None,
        grid_path: str = None,
        photometry_bands: list = None
    ):
        
        
        if grid is not None:
            self.grid = grid
        elif grid_path is not None:
            print(f"Loading isochrone grid from {grid_path}...")
            if grid_path.endswith('.parquet'):
                self.grid = pd.read_parquet(grid_path)
            else:
                self.grid = pd.read_csv(grid_path)
        else:
            raise ValueError("Either 'grid' or 'grid_path' must be provided.")
        

        if photometry_bands is None:
            self.photometry_bands = ['Gmag', 'G_BPmag', 'G_RPmag']
        else:
            self.photometry_bands = photometry_bands
        

        for band in self.photometry_bands:
            if band not in self.grid.columns:
                raise ValueError(f"Band '{band}' not found in grid.")
        

        self.unique_ages = np.sort(self.grid['logAge'].unique())
        self.unique_MH = np.sort(self.grid['MH'].unique())
        
        print("Building lookup structure...")
        self._isochrone_cache = {}
        for (age, mh), group in self.grid.groupby(['logAge', 'MH']):
            group_sorted = group.sort_values('Mini').reset_index(drop=True)
            self._isochrone_cache[(age, mh)] = group_sorted
        
        self.age_min = self.unique_ages.min()
        self.age_max = self.unique_ages.max()
        self.MH_min = self.unique_MH.min()
        self.MH_max = self.unique_MH.max()
        self.mass_min = self.grid['Mini'].min()
        self.mass_max = self.grid['Mini'].max()
        
        print(f"Grid loaded!")
        print(f"  Age range: [{self.age_min:.2f}, {self.age_max:.2f}]")
        print(f"  [M/H] range: [{self.MH_min:.2f}, {self.MH_max:.2f}]")
        print(f"  Mass range: [{self.mass_min:.3f}, {self.mass_max:.2f}] M_sun")
    

    def _get_bracket_indices(
        self, 
        value: float, 
        grid_values: np.ndarray
    ) -> Tuple[int, int, float]:
        """Find bracketing indices and interpolation weight."""
        if value <= grid_values[0]:
            return 0, 0, 0.0
        if value >= grid_values[-1]:
            return len(grid_values)-1, len(grid_values)-1, 0.0
        
        idx_hi = np.searchsorted(grid_values, value)
        idx_lo = idx_hi - 1
        
        val_lo = grid_values[idx_lo]
        val_hi = grid_values[idx_hi]
        weight = (value - val_lo) / (val_hi - val_lo)
        
        return idx_lo, idx_hi, weight
    

    def _interpolate_single_isochrone(
        self,
        mass: float,
        logAge: float,
        MH: float
    ) -> Optional[np.ndarray]:
        """Interpolate photometry at exact grid point (logAge, MH)."""
        key = (logAge, MH)
        if key not in self._isochrone_cache:
            return None
        
        iso = self._isochrone_cache[key]
        masses = iso['Mini'].values
        
        if mass < masses.min() or mass > masses.max():
            return None
        
        mags = np.zeros(len(self.photometry_bands))
        for i, band in enumerate(self.photometry_bands):
            interp_func = interp1d(masses, iso[band].values, kind='linear', bounds_error=False, fill_value=np.nan)
            mags[i] = interp_func(mass)
        
        return mags
    

    def get_magnitudes(
        self,
        mass: float,
        logAge: float,
        MH: float
    ) -> Optional[np.ndarray]:
        """
        Get predicted absolute magnitudes for a star.
        
        Parameters
        ----------
        mass : float
            Initial stellar mass in solar masses
        logAge : float
            Log10 of cluster age in years
        MH : float
            Metallicity [M/H]
        
        Returns
        -------
        np.ndarray or None
            Array of absolute magnitudes [G, BP, RP]
        """
        if logAge < self.age_min or logAge > self.age_max:
            return None
        if MH < self.MH_min or MH > self.MH_max:
            return None
        
        age_lo_idx, age_hi_idx, age_weight = self._get_bracket_indices(
            logAge, self.unique_ages
        )
        MH_lo_idx, MH_hi_idx, MH_weight = self._get_bracket_indices(
            MH, self.unique_MH
        )
        
        age_lo = self.unique_ages[age_lo_idx]
        age_hi = self.unique_ages[age_hi_idx]
        MH_lo = self.unique_MH[MH_lo_idx]
        MH_hi = self.unique_MH[MH_hi_idx]
        
        mag_ll = self._interpolate_single_isochrone(mass, age_lo, MH_lo)
        mag_lh = self._interpolate_single_isochrone(mass, age_lo, MH_hi)
        mag_hl = self._interpolate_single_isochrone(mass, age_hi, MH_lo)
        mag_hh = self._interpolate_single_isochrone(mass, age_hi, MH_hi)
        
        if any(m is None for m in [mag_ll, mag_lh, mag_hl, mag_hh]):
            valid_mags = [m for m in [mag_ll, mag_lh, mag_hl, mag_hh] if m is not None]
            if len(valid_mags) == 0:
                return None
            return np.mean(valid_mags, axis=0)
        
        mag_l = (1 - MH_weight) * mag_ll + MH_weight * mag_lh
        mag_h = (1 - MH_weight) * mag_hl + MH_weight * mag_hh
        mags = (1 - age_weight) * mag_l + age_weight * mag_h
        
        return mags
    

    def get_magnitudes_vectorized(
        self,
        masses: np.ndarray,
        logAge: float,
        MH: float
    ) -> np.ndarray:
        """
        Get predicted magnitudes for multiple stars at the same (age, MH).
        
        Parameters
        ----------
        masses : np.ndarray
            Array of initial stellar masses
        logAge : float
            Log10 of cluster age
        MH : float
            Metallicity [M/H]
        
        Returns
        -------
        np.ndarray
            Shape (n_stars, n_bands) array of magnitudes.
        """
        n_stars = len(masses)
        n_bands = len(self.photometry_bands)
        result = np.full((n_stars, n_bands), np.nan)
        
        if logAge < self.age_min or logAge > self.age_max:
            return result
        if MH < self.MH_min or MH > self.MH_max:
            return result
        
        age_lo_idx, age_hi_idx, age_weight = self._get_bracket_indices(
            logAge, self.unique_ages
        )
        MH_lo_idx, MH_hi_idx, MH_weight = self._get_bracket_indices(
            MH, self.unique_MH
        )
        
        age_lo = self.unique_ages[age_lo_idx]
        age_hi = self.unique_ages[age_hi_idx]
        MH_lo = self.unique_MH[MH_lo_idx]
        MH_hi = self.unique_MH[MH_hi_idx]
        
        corners = [
            (age_lo, MH_lo),
            (age_lo, MH_hi),
            (age_hi, MH_lo),
            (age_hi, MH_hi)
        ]
        
        corner_mags = []
        for age, mh in corners:
            iso = self._isochrone_cache.get((age, mh))
            if iso is None:
                corner_mags.append(None)
                continue
            
            iso_masses = iso['Mini'].values
            mags = np.full((n_stars, n_bands), np.nan)
            
            for i, band in enumerate(self.photometry_bands):
                interp_func = interp1d(
                    iso_masses,
                    iso[band].values,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                mags[:, i] = interp_func(masses)
            
            corner_mags.append(mags)
        
        mag_ll, mag_lh, mag_hl, mag_hh = corner_mags
        
        if any(m is None for m in corner_mags):
            for j, mass in enumerate(masses):
                mag = self.get_magnitudes(mass, logAge, MH)
                if mag is not None:
                    result[j] = mag
            return result
        
        mag_l = (1 - MH_weight) * mag_ll + MH_weight * mag_lh
        mag_h = (1 - MH_weight) * mag_hl + MH_weight * mag_hh
        result = (1 - age_weight) * mag_l + age_weight * mag_h
        
        return result
    

    def get_max_mass_at_age(self, logAge: float, MH: float) -> float:
        """Get the maximum initial mass that exists at given age."""
        age_idx = np.argmin(np.abs(self.unique_ages - logAge))
        MH_idx = np.argmin(np.abs(self.unique_MH - MH))
        
        age = self.unique_ages[age_idx]
        mh = self.unique_MH[MH_idx]
        
        iso = self._isochrone_cache.get((age, mh))
        if iso is None:
            return self.mass_max
        
        return iso['Mini'].max()
    
    
    def get_min_mass(self) -> float:
        """Get minimum mass in the grid."""
        return self.mass_min