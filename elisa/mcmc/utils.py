import numpy as np

# =============================================================================
# Likelihood Function
# =============================================================================

def log_likelihood(
    observed_mags: np.ndarray,
    observed_errors: np.ndarray,
    predicted_mags: np.ndarray,
    error_floor: float = 0.02,
    intrinsic_scatter: float = 0.05
) -> float:
    """
    Compute log-likelihood comparing observed and predicted magnitudes.

    Implements Equation 2 from von Hippel et al. (2006), with added
    error floor and intrinsic scatter to handle model imperfections.

    Parameters
    ----------
    observed_mags : np.ndarray
        Observed magnitudes, shape (n_stars, n_bands)
    observed_errors : np.ndarray
        Photometric errors, shape (n_stars, n_bands)
    predicted_mags : np.ndarray
        Predicted magnitudes from model, shape (n_stars, n_bands)
    error_floor : float
        Minimum error to prevent overly tight constraints (default: 0.02 mag)
    intrinsic_scatter : float
        Additional scatter added in quadrature to account for model
        imperfections, binaries, rotation, etc. (default: 0.05 mag)

    Returns
    -------
    float
        Log-likelihood value. Returns -inf if any prediction is NaN.
    """

    if np.any(np.isnan(predicted_mags)):
        return -np.inf

    # Apply error floor and add intrinsic scatter in quadrature
    # This accounts for: model imperfections, unresolved binaries, stellar rotation, activity, etc.
    total_errors = np.sqrt(np.maximum(observed_errors, error_floor) ** 2 + intrinsic_scatter ** 2)

    # Gaussian log-likelihood
    # log(L) = -0.5 * sum[ log(2*pi*sigma^2) + (x - mu)^2 / sigma^2 ]

    variance = total_errors ** 2
    residuals = observed_mags - predicted_mags

    log_like = -0.5 * np.sum(np.log(2 * np.pi * variance) + (residuals ** 2) / variance)

    return log_like


# =============================================================================
# Prior Functions
# =============================================================================

def log_prior_imf_chabrier(mass: float) -> float:
    """
    Chabrier (2003) IMF log-prior for stellar mass.
    
    Parameters
    ----------
    mass : float
        Stellar mass in solar masses
    
    Returns
    -------
    float
        Log-prior probability
    """

    if mass <= 0:
        return -np.inf
    
    if mass <= 1.0:
        # Log-normal for M <= 1 Msun
        m_c = 0.079  # characteristic mass
        sigma = 0.69
        log_prior = -((np.log10(mass) - np.log10(m_c)) ** 2) / (2 * sigma ** 2)
    else:
        # Power law for M > 1 Msun
        alpha = 2.3  # Salpeter-like slope
        log_prior = -alpha * np.log(mass)
    
    return log_prior


def log_prior_imf_chabrier_vectorized(masses: np.ndarray) -> float:
    """
    Vectorized Chabrier IMF log-prior for array of masses.
    
    Parameters
    ----------
    masses : np.ndarray
        Array of stellar masses
    
    Returns
    -------
    float
        Sum of log-prior probabilities
    """
    if np.any(masses <= 0):
        return -np.inf
    
    log_priors = np.zeros_like(masses)
    
    # Low mass: log-normal
    low_mass = masses <= 1.0
    m_c = 0.079
    sigma = 0.69
    log_priors[low_mass] = -((np.log10(masses[low_mass]) - np.log10(m_c)) ** 2) / (2 * sigma ** 2)
    
    # High mass: power law
    high_mass = masses > 1.0
    alpha = 2.3
    log_priors[high_mass] = -alpha * np.log(masses[high_mass])
    
    return np.sum(log_priors)


def log_prior_gaussian(value: float, mean: float, std: float) -> float:
    """
    Gaussian log-prior.
    
    Parameters
    ----------
    value : float
        Parameter value
    mean : float
        Prior mean
    std : float
        Prior standard deviation
    
    Returns
    -------
    float
        Log-prior probability
    """
    return -0.5 * ((value - mean) / std) ** 2


def log_prior_uniform(value: float, low: float, high: float) -> float:
    """
    Uniform log-prior.
    
    Parameters
    ----------
    value : float
        Parameter value
    low : float
        Lower bound
    high : float
        Upper bound
    
    Returns
    -------
    float
        Log-prior probability (0 if in bounds, -inf otherwise)
    """
    if low <= value <= high:
        return 0.0
    return -np.inf