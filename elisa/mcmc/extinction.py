"""
Step 3: Extinction Law for Gaia Photometry

Converts A_V (visual extinction) to extinction in Gaia bands (A_G, A_BP, A_RP).

References:
- Cardelli, Clayton & Mathis (1989) - CCM extinction law
- Casagrande & VandenBerg (2018) - Gaia DR2 extinction coefficients
- Gaia Collaboration (2018) - Recommended coefficients
"""

import numpy as np
from typing import Tuple, Union


class Extinction:
    """
    Extinction law for Gaia G, BP, RP bands.
    
    Provides conversion from A_V to Gaia extinctions.
    
    Parameters
    ----------
    R_V : float, optional
        Total-to-selective extinction ratio. Default: 3.1
    coefficients : str, optional
        Which coefficient set to use: 'fixed' or 'color_dependent'
        Default: 'fixed'
    """
    
    def __init__(self, coefficients: str = 'fixed'):
        self.coefficients = coefficients
        
        # Fixed coefficients (average values, good approximation)
        # From Casagrande & VandenBerg (2018) and Gaia DR2 documentation
        # A_X / A_V ratios
        self.k_G = 0.83627   # A_G / A_V
        self.k_BP = 1.08337  # A_BP / A_V
        self.k_RP = 0.63439  # A_RP / A_V
        
        # For E(BP-RP) / E(B-V)
        # E(BP-RP) = A_BP - A_RP = (k_BP - k_RP) * A_V
        # E(B-V) = A_V / R_V
        # So E(BP-RP) / A_V = k_BP - k_RP = 0.44898
        self.k_E_BP_RP = self.k_BP - self.k_RP  # ~0.449
    

    def get_extinction(
        self, 
        A_V: float,
        bp_rp_0: Union[float, np.ndarray] = None
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert A_V to Gaia band extinctions.
        
        Parameters
        ----------
        A_V : float
            Visual extinction in magnitudes
        bp_rp_0 : float or array, optional
            Intrinsic BP-RP color (for color-dependent coefficients).
            If None, uses fixed coefficients.
        
        Returns
        -------
        A_G, A_BP, A_RP : floats or arrays
            Extinction in each Gaia band
        """
        if self.coefficients == 'fixed' or bp_rp_0 is None:
            A_G = self.k_G * A_V
            A_BP = self.k_BP * A_V
            A_RP = self.k_RP * A_V
        else:
            # Color-dependent coefficients
            # Polynomial fits from Gaia DR2 documentation
            A_G, A_BP, A_RP = self._color_dependent_extinction(A_V, bp_rp_0)
        
        return A_G, A_BP, A_RP
    

    def _color_dependent_extinction(
        self, 
        A_V: float, 
        bp_rp_0: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Compute color-dependent extinction coefficients.
        
        Based on polynomial fits from Gaia Collaboration.
        Valid for -0.5 < BP-RP < 2.75
        """
        # Ensure within valid range
        bp_rp = np.clip(bp_rp_0, -0.5, 2.75)
        
        # Polynomial coefficients for k_X = A_X / A_V
        # From Babusiaux et al. (2018) / Gaia DR2
        
        # k_G coefficients
        c_G = [0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099]
        k_G = (c_G[0] + c_G[1]*bp_rp + c_G[2]*bp_rp**2 + c_G[3]*bp_rp**3 +
               c_G[4]*A_V + c_G[5]*A_V**2 + c_G[6]*bp_rp*A_V)
        
        # k_BP coefficients  
        c_BP = [1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043]
        k_BP = (c_BP[0] + c_BP[1]*bp_rp + c_BP[2]*bp_rp**2 + c_BP[3]*bp_rp**3 +
                c_BP[4]*A_V + c_BP[5]*A_V**2 + c_BP[6]*bp_rp*A_V)
        
        # k_RP coefficients
        c_RP = [0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006]
        k_RP = (c_RP[0] + c_RP[1]*bp_rp + c_RP[2]*bp_rp**2 + c_RP[3]*bp_rp**3 +
                c_RP[4]*A_V + c_RP[5]*A_V**2 + c_RP[6]*bp_rp*A_V)
        
        A_G = k_G * A_V
        A_BP = k_BP * A_V
        A_RP = k_RP * A_V
        
        return A_G, A_BP, A_RP
    

    def get_reddening(self, A_V: float) -> float:
        """
        Get color excess E(BP-RP) from A_V.
        
        Parameters
        ----------
        A_V : float
            Visual extinction
        
        Returns
        -------
        E_BP_RP : float
            Color excess E(BP-RP) = A_BP - A_RP
        """
        A_G, A_BP, A_RP = self.get_extinction(A_V)
        return A_BP - A_RP
    

    def A_V_from_E_BP_RP(self, E_BP_RP: float) -> float:
        """
        Convert E(BP-RP) to A_V.
        
        Parameters
        ----------
        E_BP_RP : float
            Color excess E(BP-RP)
        
        Returns
        -------
        A_V : float
            Visual extinction
        """
        return E_BP_RP / self.k_E_BP_RP


    def apply_extinction_and_distance(self,
        abs_mags: np.ndarray,
        distance_modulus: float,
        A_V: float,
        bp_rp_0: np.ndarray = None
    ) -> np.ndarray:
        """
        Convert absolute magnitudes to apparent magnitudes.
        
        Applies distance modulus and extinction.
        
        Parameters
        ----------
        abs_mags : np.ndarray
            Absolute magnitudes, shape (n_stars, 3) for [G, BP, RP]
        distance_modulus : float
            Distance modulus (m - M) in magnitudes
        A_V : float
            Visual extinction in magnitudes
        bp_rp_0 : np.ndarray, optional
            Intrinsic colors for color-dependent extinction
        
        Returns
        -------
        app_mags : np.ndarray
            Apparent magnitudes, shape (n_stars, 3)
        """
        
        # Compute intrinsic color if not provided
        if bp_rp_0 is None:
            bp_rp_0 = abs_mags[:, 1] - abs_mags[:, 2]  # BP - RP
        
        # Get extinction in each band
        A_G, A_BP, A_RP = self.get_extinction(A_V, bp_rp_0)
        
        # Apply to magnitudes
        app_mags = abs_mags.copy()
        app_mags[:, 0] += distance_modulus + A_G   # G
        app_mags[:, 1] += distance_modulus + A_BP  # BP
        app_mags[:, 2] += distance_modulus + A_RP  # RP
        
        return app_mags


def distance_modulus_to_distance(dm: float) -> float:
    """Convert distance modulus to distance in parsecs."""
    return 10 ** (dm / 5 + 1)


def distance_to_distance_modulus(d_pc: float) -> float:
    """Convert distance in parsecs to distance modulus."""
    return 5 * np.log10(d_pc) - 5
