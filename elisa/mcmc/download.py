"""
Step 1: Download and Save PARSEC Isochrone Grid for Cluster Parameter Inference

This script downloads isochrones from the PARSEC database using ezpadova
and saves them locally for fast access during MCMC sampling.

Requirements:
    pip install ezpadova pandas pyarrow --break-system-packages
"""

import ezpadova


def download_isochrone_grid(
    logage_range=(6.0, 10.5, 0.05), # (min, max, step), 0, 12
    MH_range=(-2.0, 1.0, 0.1),      # (min, max, step)
    photsys='gaiaEDR3'
):
    """
    Download PARSEC isochrones and save locally.
    
    Parameters
    ----------
    logage_range : tuple
        (min, max, step) for log(age/yr)
    MH_range : tuple
        (min, max, step) for [M/H]
    photsys : str
        Photometric system (default: gaiaEDR3)
    output_dir : str
        Directory to save the isochrone grid
    output_name : str
        Base name for output files
    save_file : bool
        Whether to save the grid to file (default: False)
    
    Returns
    -------
    pd.DataFrame
        The downloaded isochrone grid
    """

    
    print("Downloading from PARSEC server (this may take a few minutes)...")
    
    try:
        isochrones = ezpadova.get_isochrones(logage=logage_range, MH=MH_range, photsys_file=photsys)
        print(f"Download complete! Shape: {isochrones.shape}")
        
    except Exception as e:
        print(f"Error downloading isochrones: {e}")
        raise
    
    print()
    print("Grid summary:")
    print(f"  Total rows: {len(isochrones):,}")
    print(f"  Unique log(age) values: {isochrones['logAge'].nunique()}")
    print(f"  Unique [M/H] values: {isochrones['MH'].nunique()}")
    print(f"  Mass range: {isochrones['Mini'].min():.3f} - {isochrones['Mini'].max():.2f} M_sun")
    print()
    
    # Select relevant columns for cluster fitting
    # Keep only what we need to reduce file size
    essential_columns = [
        # Grid parameters
        'MH',           # Metallicity [M/H]
        'logAge',       # Log age
        'Mini',         # Initial mass (ZAMS mass)
        'Mass',         # Current mass
        'label',        # Evolutionary phase label
        # Physical parameters (useful for diagnostics)
        'logTe',        # Log effective temperature
        'logg',         # Log surface gravity
        'logL',         # Log luminosity
        # Gaia photometry
        'Gmag',         # Gaia G absolute magnitude
        'G_BPmag',      # Gaia BP absolute magnitude
        'G_RPmag',      # Gaia RP absolute magnitude
    ]
    
    # Check which columns exist
    available_columns = [c for c in essential_columns if c in isochrones.columns]
    missing_columns = [c for c in essential_columns if c not in isochrones.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    isochrones_slim = isochrones[available_columns].copy()
    
    return isochrones_slim

