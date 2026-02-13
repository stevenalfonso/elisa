import torch
import numpy as np
import pandas as pd
import os
import pickle
from scipy import interpolate
from tqdm import tqdm
from isochrones import get_ichrone
from sbi.inference import SNPE as method
from typing import Tuple, List, Dict, Optional


# Load Gaia error model from your data
def build_gaia_error_splines(data_path: str):
    """
    Build spline interpolations for Gaia photometric errors from observed data.
    Uses the flux_over_error columns if available, otherwise estimates from magnitude.
    """

    df = pd.read_csv(data_path)

    # Gaia error model: sigma_mag ≈ 2.5/(ln(10) * flux_over_error)
    # For NGC 2547 data, we estimate errors based on typical Gaia DR3 performance
    # These are approximate relations from Gaia DR3 documentation

    splines = {}
    for band in ['G', 'BP', 'RP']:
        mag_col = f'{band}mag' if band != 'G' else 'Gmag'
        mags = df[mag_col].values

        # Estimate errors based on Gaia DR3 typical uncertainties
        # Bright end (G<13): ~0.003 mag; Faint end (G>17): ~0.01-0.1 mag
        if band == 'G':
            errs = 0.0003 * 10**((mags - 12) / 5)  # Exponential scaling
            errs = np.clip(errs, 0.001, 0.1)
        elif band == 'BP':
            errs = 0.001 * 10**((mags - 12) / 4)  # BP has larger errors
            errs = np.clip(errs, 0.002, 0.15)
        else:  # RP
            errs = 0.0004 * 10**((mags - 12) / 5)
            errs = np.clip(errs, 0.001, 0.1)

        # Sort by magnitude and create spline
        sort_idx = np.argsort(mags)
        spl = interpolate.UnivariateSpline(mags[sort_idx], errs[sort_idx], s=0.01, k=3)
        splines[band] = spl

    return splines


def get_gaia_errors(bp_mag, rp_mag, g_mag, parallax=None):
    """
    Generate realistic Gaia photometric errors using lognormal sampling.
    Following the approach from awallace142857/sbi_code.
    """
    global _gaia_error_splines

    if _gaia_error_splines is None:
        _gaia_error_splines = build_gaia_error_splines()

    errors = {}
    sigma = 0.5  # Lognormal shape parameter (same as reference code)

    for band, mag in [('G', g_mag), ('BP', bp_mag), ('RP', rp_mag)]:
        mode = float(_gaia_error_splines[band](mag))
        mode = max(mode, 0.001)  # Floor at 1 mmag
        # Lognormal: mean = ln(mode) + sigma^2
        mean = np.log(mode) + sigma**2
        errors[band] = np.random.lognormal(mean, sigma)

    # Parallax error: typical for NGC 2547 distance (~400 pc, plx ~2.5 mas)
    if parallax is not None:
        # Error scales roughly with magnitude (proxy for brightness)
        plx_mode = 0.02 + 0.01 * (g_mag - 10) / 8  # 0.02-0.03 mas typical
        plx_mode = max(plx_mode, 0.01)
        mean = np.log(plx_mode) + sigma**2
        errors['parallax'] = np.random.lognormal(mean, sigma)

    return errors


class ElisaBinary:

    def __init__(self, params: List[str] = ["M1", "q", "age", "feh", "distance"],
                 prior: classmethod = None):
        self.params = params
        self.prior = prior
        self.tracks_on = False


    def set_tracks(self):
        self.tracks = get_ichrone('mist', tracks=True)
        self.tracks_on = True

    
    def binary_color_mag_isochrones(self, 
                                    m1, 
                                    q, 
                                    age, 
                                    fe_h, 
                                    distance=None, 
                                    add_noise=True):
        """
        Generate synthetic photometry for a binary star system.

        Parameters
        ----------
        add_noise : bool
            If True, adds realistic Gaia photometric errors using lognormal sampling.
            This is CRITICAL for SBI to work properly - training on noise-free data
            but inferring from noisy observations causes severe biases (e.g., all
            stars appearing as high-q binaries).
        """

        m1 = float(m1)
        q = float(q)
        age = float(age)
        fe_h = float(fe_h)
        if distance is not None:
            distance = float(distance)


        if distance is not None:
            # Pass distance to generate_binary - it returns apparent magnitudes directly
            properties = self.tracks.generate_binary(m1, q * m1, np.log10(age) + 9, fe_h, distance=distance, bands=["G", "BP", "RP"])
            bp_mag = properties.BP_mag.values[0]
            g_mag = properties.G_mag.values[0]
            rp_mag = properties.RP_mag.values[0]

            # Parallax in milliarcseconds: plx = 1000 / d(pc)
            parallax = 1000.0 / distance

            if add_noise:
                # Get lognormal-sampled errors based on magnitude
                errs = get_gaia_errors(bp_mag, rp_mag, g_mag, parallax)
                bp_mag = np.random.normal(bp_mag, errs['BP'])
                rp_mag = np.random.normal(rp_mag, errs['RP'])
                g_mag = np.random.normal(g_mag, errs['G'])
                parallax = np.random.normal(parallax, errs['parallax'])

            # Return [BP, RP, G, parallax] to match observation order
            return np.array([bp_mag, rp_mag, g_mag, parallax])
        else:
            # No distance - get absolute magnitudes
            properties = self.tracks.generate_binary(m1, q * m1, np.log10(age) + 9, fe_h, bands=["G", "BP", "RP"])
            bp_abs = properties.BP_mag.values[0]
            g_abs = properties.G_mag.values[0]
            rp_abs = properties.RP_mag.values[0]

            if add_noise:
                # Estimate apparent mags assuming NGC 2547 distance for error calculation
                distance_modulus_approx = 8.0  # ~400 pc
                bp_app = bp_abs + distance_modulus_approx
                rp_app = rp_abs + distance_modulus_approx
                g_app = g_abs + distance_modulus_approx

                errs = get_gaia_errors(bp_app, rp_app, g_app)
                bp_abs = np.random.normal(bp_abs, errs['BP'])
                rp_abs = np.random.normal(rp_abs, errs['RP'])
                g_abs = np.random.normal(g_abs, errs['G'])

            # Return absolute magnitudes [BP, RP, G]
            return np.array([bp_abs, rp_abs, g_abs])
        
    
    def make_simulator(self, 
                       fixed_params, 
                       use_apparent_mags):
        """
        Factory function that creates a simulator with the given configuration.
        """
        def simulator(theta):
            """
            Simulator that maps theta (only the inferred parameters) to observables.
            Uses fixed_params for any parameters that were specified via command line.
            """
            # theta contains only the parameters we're inferring, in order: [M1, q, (age), (feh), (distance)]
            # We need to reconstruct the full parameter set for binary_color_mag_isochrones
            theta_list = theta.tolist() if hasattr(theta, 'tolist') else list(theta)

            idx = 0
            m1 = theta_list[idx]; idx += 1
            q = theta_list[idx]; idx += 1

            if 'age' in fixed_params:
                age = fixed_params['age']
            else:
                age = theta_list[idx]; idx += 1

            if 'feh' in fixed_params:
                fe_h = fixed_params['feh']
            else:
                fe_h = theta_list[idx]; idx += 1

            if use_apparent_mags:
                if 'distance' in fixed_params:
                    distance = fixed_params['distance']
                else:
                    distance = theta_list[idx]; idx += 1
            else:
                distance = None

            return torch.tensor(self.binary_color_mag_isochrones(m1, q, age, fe_h, distance))

        return simulator
    

    def train(self,
            simulator, 
            proposal, 
            num_simulations, 
            max_trials=np.inf):
        """
        simulate_for_sbi_strict
        """
        
        num_trials, num_simulated, theta, x = (0, 0, [], [])

        with tqdm(total=num_simulations, desc="Simulating") as pbar:
            while num_simulated < num_simulations:
                N = num_simulations - num_simulated
                _theta = proposal.sample((N,))
                _x = simulator(_theta)
                _x = _x.squeeze(1)

                keep = np.all(np.isfinite(_x.numpy()), axis=1)
                theta.extend(np.array(_theta[keep]))
                x.extend(np.array(_x[keep]))

                num_trials += 1
                num_valid = sum(keep)
                num_simulated += num_valid
                pbar.update(num_valid)  # Update progress bar by number of valid simulations

                if num_trials > max_trials:
                    print(f"Warning: exceeding max trials ({max_trials}) with {num_simulated} / {num_simulations} simulations")
                    break

        theta = torch.tensor(np.vstack(theta))
        x = torch.tensor(np.vstack(x))

        return (theta, x)
    

if __name__ == "__main__":
    
    # Example usage
    elisa = ElisaBinary()
    elisa.set_tracks()

    fixed_params = {'age': 30e6, 'feh': 0.0}  # Example fixed parameters
    use_apparent_mags = True

    simulator = elisa.make_simulator(fixed_params, use_apparent_mags)

    prior = method.priors.BoxUniform(low=torch.tensor([0.1, 0.1, 50.0]), high=torch.tensor([3.0, 1.0, 1000.0]))
    theta, x = elisa.train(simulator, prior, num_simulations=1000)

    print("Simulated parameters (theta):", theta)
    print("Simulated observables (x):", x)
