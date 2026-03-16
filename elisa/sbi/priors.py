"""
Built-in prior distributions for binary-star SBI.

``StellarPrior`` is one convenience option.  You are free to use any
prior compatible with ``sbi`` — for example ``sbi.utils.BoxUniform`` for
a simple uniform prior, or any ``torch.distributions`` object with
``.sample()`` and ``.log_prob()`` methods.

Examples
--------
Uniform prior (two parameters: M1, q)::

    from sbi.utils import BoxUniform
    import torch
    prior = BoxUniform(
        low=torch.tensor([0.1, 0.0]),
        high=torch.tensor([5.0, 1.0]),
    )

Astrophysically-motivated prior::

    from elisa.sbi.priors import StellarPrior
    prior = StellarPrior(
        M1_bounds=(0.1, 5.0),
        q_bounds=(0.0, 1.0),
        fixed_params={"age": 0.035, "feh": 0.0, "distance": 400.0},
    )
"""

import torch
from torch.distributions import Beta, Independent
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from typing import Dict, List, Optional, Tuple


class StellarPrior:
    """
    Joint prior over binary-star parameters using Beta distributions.

    Parameters ``M1`` and ``[Fe/H]`` use Beta priors that encode
    astrophysical knowledge (IMF-like shape for mass; metallicity
    distribution peaked near solar).  Mass ratio ``q`` uses
    ``Beta(q_alpha, q_beta)`` — uniform by default, or set
    ``q_beta > 1`` to favour single stars (low *q*).  ``age`` and
    ``distance`` use uniform priors when not fixed.

    Parameters
    ----------
    M1_bounds : tuple
        ``(low, high)`` primary-mass range in solar masses.
    q_bounds : tuple
        ``(low, high)`` mass-ratio range, typically ``(0.0, 1.0)``.
    tau_bounds : tuple or None
        Age range in Gyr.  Required when ``'age'`` is not in
        ``fixed_params``.
    fe_h_bounds : tuple or None
        ``[Fe/H]`` range.  Required when ``'feh'`` is not in
        ``fixed_params``.
    distance_bounds : tuple
        Distance range in pc, used when ``with_distance=True`` and
        ``'distance'`` is not in ``fixed_params``.
    M1_alpha, M1_beta : float
        Beta-distribution shape parameters for *M1*.  ``Beta(1, 5)``
        gives an IMF-like shape skewed towards low masses.
    q_alpha, q_beta : float
        Beta-distribution shape parameters for *q*.  ``Beta(1, 1)`` is
        uniform; ``Beta(1, 3)`` favours single stars (mean *q* ≈ 0.25).
    fe_h_alpha, fe_h_beta, fe_h_scale : float
        Shape parameters for the ``[Fe/H]`` Beta prior.
    with_distance : bool
        Include distance as an inferred parameter.
    fixed_params : dict, optional
        Parameters that are held fixed (not sampled).  Keys:
        ``'age'``, ``'feh'``, ``'distance'``.
    """

    def __init__(
        self,
        M1_bounds: Tuple[float, float] = (0.3, 5.0),
        q_bounds: Tuple[float, float] = (0.0, 1.0),
        tau_bounds: Optional[Tuple[float, float]] = None,
        fe_h_bounds: Optional[Tuple[float, float]] = None,
        distance_bounds: Tuple[float, float] = (100.0, 1000.0),
        M1_alpha: float = 1.0,
        M1_beta: float = 5.0,
        q_alpha: float = 1.0,
        q_beta: float = 1.0,
        fe_h_alpha: float = 10.0,
        fe_h_beta: float = 2.0,
        fe_h_scale: float = 3.0,
        with_distance: bool = False,
        fixed_params: Optional[Dict] = None,
    ):
        fixed_params = fixed_params or {}

        lower_list: List[float] = []
        upper_list: List[float] = []
        alphas_list: List[float] = []
        betas_list: List[float] = []
        loc_list: List[float] = []
        scale_list: List[float] = []

        # M1 — always inferred
        lower_list.append(M1_bounds[0])
        upper_list.append(M1_bounds[1])
        alphas_list.append(M1_alpha)
        betas_list.append(M1_beta)
        loc_list.append(M1_bounds[0])
        scale_list.append(M1_bounds[1] - M1_bounds[0])

        # q — always inferred
        lower_list.append(q_bounds[0])
        upper_list.append(q_bounds[1])
        alphas_list.append(q_alpha)
        betas_list.append(q_beta)
        loc_list.append(q_bounds[0])
        scale_list.append(q_bounds[1] - q_bounds[0])

        # age — only when not fixed
        if "age" not in fixed_params:
            if tau_bounds is None:
                raise ValueError("tau_bounds must be provided when 'age' is not fixed.")
            lower_list.append(tau_bounds[0])
            upper_list.append(tau_bounds[1])
            alphas_list.append(1.0)
            betas_list.append(1.0)
            loc_list.append(tau_bounds[0])
            scale_list.append(tau_bounds[1] - tau_bounds[0])

        # [Fe/H] — only when not fixed
        if "feh" not in fixed_params:
            if fe_h_bounds is None:
                raise ValueError("fe_h_bounds must be provided when 'feh' is not fixed.")
            fe_h_mode = (fe_h_alpha - 1) / (fe_h_alpha + fe_h_beta - 2)
            lower_list.append(fe_h_bounds[0])
            upper_list.append(fe_h_bounds[1])
            alphas_list.append(fe_h_alpha)
            betas_list.append(fe_h_beta)
            loc_list.append(-fe_h_mode * fe_h_scale)
            scale_list.append(fe_h_scale)

        # distance — only when with_distance and not fixed
        if with_distance and "distance" not in fixed_params:
            lower_list.append(distance_bounds[0])
            upper_list.append(distance_bounds[1])
            alphas_list.append(1.0)
            betas_list.append(1.0)
            loc_list.append(distance_bounds[0])
            scale_list.append(distance_bounds[1] - distance_bounds[0])

        lower_bound = torch.tensor(lower_list, dtype=torch.float32)
        upper_bound = torch.tensor(upper_list, dtype=torch.float32)
        alphas = torch.tensor(alphas_list, dtype=torch.float32)
        betas = torch.tensor(betas_list, dtype=torch.float32)
        loc = torch.tensor(loc_list, dtype=torch.float32)
        scale = torch.tensor(scale_list, dtype=torch.float32)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bounds = dict(lower_bound=lower_bound, upper_bound=upper_bound)
        self.dist = Independent(
            TransformedDistribution(
                Beta(alphas, betas, validate_args=False),
                AffineTransform(loc=loc, scale=scale),
            ),
            1,
        )
        # Pre-compute mean and variance so sbi doesn't warn
        _samples = self.dist.sample((10_000,))
        self.mean = _samples.mean(dim=0)
        self.variance = _samples.var(dim=0)

    def sample(self, sample_shape=torch.Size([])):
        return self.dist.sample(sample_shape)

    def log_prob(self, values):
        return self.dist.log_prob(values)
