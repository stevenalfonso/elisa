import os
import pickle

import numpy as np
import pandas as pd
import torch
from isochrones import get_ichrone
from scipy import interpolate
from sbi.inference import SNPE
from sbi.utils.user_input_checks import prepare_for_sbi
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


def build_error_splines(
    data_path: str,
    s: float = 0.01,
    k: int = 3,
    bands: List[str] = None,
) -> Dict[str, interpolate.UnivariateSpline]:
    """
    Build spline interpolations for Gaia photometric errors from observed data.

    The CSV must contain columns ``{band}mag`` and ``e_{band}mag`` for each
    band (e.g. ``Gmag``, ``e_Gmag``, ``BPmag``, ``e_BPmag``, ``RPmag``, ``e_RPmag``).

    Parameters
    ----------
    data_path : str
        Path to the CSV file with observed photometry.
    s : float
        Smoothing factor for the spline fit.
    k : int
        Degree of the spline.
    bands : list of str, optional
        Photometric bands to process.  Defaults to ``['G', 'BP', 'RP']``.

    Returns
    -------
    dict
        Mapping from band name to a fitted ``UnivariateSpline``.
    """
    if bands is None:
        bands = ["G", "BP", "RP"]

    df = pd.read_csv(data_path)
    print(f"Building error splines from {data_path} ({len(df)} stars)")

    splines: Dict[str, interpolate.UnivariateSpline] = {}
    for band in bands:
        mag_col = f"{band}mag"
        err_col = f"e_{band}mag"
        mag = df[mag_col].values
        err = df[err_col].values
        sort_idx = np.argsort(mag)
        splines[band] = interpolate.UnivariateSpline(
            mag[sort_idx], err[sort_idx], s=s, k=k
        )

    return splines


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ElisaBinary:
    """
    Simulation-Based Inference (SBI) for binary-star parameters using
    Gaia photometry of star-cluster members.

    Typical workflow::

        elisa = ElisaBinary()
        elisa.set_tracks()                        # load MIST tracks
        elisa.set_error_splines("cluster.csv")    # build noise model

        prior = StellarPrior(
            M1_bounds=(0.1, 5.0),
            q_bounds=(0.0, 1.0),
            fixed_params={"age": 0.035, "feh": 0.0, "distance": 400.0},
        )

        posterior, theta, x = elisa.build_posterior(
            prior,
            fixed_params={"age": 0.035, "feh": 0.0, "distance": 400.0},
            num_simulations=100_000,
        )
        elisa.save(posterior, theta, x, "cluster_posterior.pkl")

        results_df = elisa.infer(posterior, df_observed)

    Parameters
    ----------
    params : list of str, optional
        Names of the physical parameters (used for bookkeeping).
        Default: ``["M1", "q", "age", "feh", "distance"]``.
    prior : prior object, optional
        Prior to store on the instance.  Can also be passed directly
        to :meth:`build_posterior`.
    """

    # Reasonable hard bounds for clipping posterior samples
    _PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
        "M1":       (0.1,  5.0),
        "q":        (0.0,  1.0),
        "age":      (0.001, 5.0),
        "[Fe/H]":   (-2.5, 0.5),
        "distance": (10.0, 5000.0),
    }

    def __init__(
        self,
        params: List[str] = None,
        prior=None,
    ):
        self.params = params or ["M1", "q", "age", "feh", "distance"]
        self.prior = prior
        self.tracks_on = False
        self._error_splines: Optional[Dict] = None
        self._last_labels: Optional[Tuple[str, ...]] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_tracks(self):
        """Load MIST stellar evolution tracks (required before simulating)."""
        self.tracks = get_ichrone("mist", tracks=True)
        self.tracks_on = True

    def set_error_splines(
        self,
        data_path: str,
        s: float = 0.01,
        k: int = 3,
        bands: List[str] = None,
    ):
        """
        Build photometric-error splines from observed cluster data.

        The CSV must have columns ``{band}mag`` and ``e_{band}mag``
        (e.g. ``Gmag``, ``e_Gmag``).

        Parameters
        ----------
        data_path : str
            Path to the CSV file.
        s : float
            Smoothing factor for ``UnivariateSpline``.
        k : int
            Spline degree.
        bands : list of str, optional
            Bands to process.  Default: ``['G', 'BP', 'RP']``.
        """
        self._error_splines = build_error_splines(data_path, s=s, k=k, bands=bands)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_errors(
        self,
        bp_mag: float,
        rp_mag: float,
        g_mag: float,
        parallax: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Sample realistic Gaia photometric errors using lognormal sampling.
        Requires :meth:`set_error_splines` to have been called.
        """
        if self._error_splines is None:
            raise RuntimeError(
                "Error splines not set.  Call set_error_splines(data_path) first."
            )

        sigma = 0.5  # lognormal shape parameter
        errors: Dict[str, float] = {}

        for band, mag in [("G", g_mag), ("BP", bp_mag), ("RP", rp_mag)]:
            mode = float(self._error_splines[band](mag))
            mode = max(mode, 0.001)  # floor at 1 mmag
            mean = np.log(mode) + sigma ** 2
            errors[band] = float(np.random.lognormal(mean, sigma))

        if parallax is not None:
            plx_mode = 0.02 + 0.01 * (g_mag - 10.0) / 8.0
            plx_mode = max(plx_mode, 0.01)
            mean = np.log(plx_mode) + sigma ** 2
            errors["parallax"] = float(np.random.lognormal(mean, sigma))

        return errors

    @staticmethod
    def _infer_labels(
        fixed_params: Dict,
        use_apparent_mags: bool,
    ) -> Tuple[str, ...]:
        """Derive the ordered list of inferred parameter names."""
        labels = ["M1", "q"]
        if "age" not in fixed_params:
            labels.append("age")
        if "feh" not in fixed_params:
            labels.append("[Fe/H]")
        if use_apparent_mags and "distance" not in fixed_params:
            labels.append("distance")
        return tuple(labels)

    # ------------------------------------------------------------------
    # Simulator
    # ------------------------------------------------------------------

    def binary_color_mag_isochrones(
        self,
        m1: float,
        q: float,
        age: float,
        fe_h: float,
        distance: Optional[float] = None,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Generate synthetic Gaia photometry for a binary-star system.

        Parameters
        ----------
        m1 : float
            Primary mass in solar masses.
        q : float
            Mass ratio *m2/m1*, in ``[0, 1]``.
        age : float
            Cluster age in Gyr.
        fe_h : float
            Metallicity ``[Fe/H]``.
        distance : float or None
            Distance in pc.  When given, returns apparent magnitudes
            plus parallax.  When ``None``, returns absolute magnitudes.
        add_noise : bool
            If ``True``, adds realistic Gaia photometric noise sampled
            from the error splines.  **Critical for unbiased SBI
            training** — training on noise-free data but inferring from
            noisy observations leads to severe biases.

        Returns
        -------
        np.ndarray
            ``[BP, RP, G, parallax]`` when *distance* is given, else
            ``[BP, RP, G]`` (absolute magnitudes).
        """
        if not self.tracks_on:
            raise RuntimeError("Tracks not loaded.  Call set_tracks() first.")

        m1 = float(m1)
        q = float(q)
        age = float(age)
        fe_h = float(fe_h)

        if distance is not None:
            distance = float(distance)
            props = self.tracks.generate_binary(
                m1, q * m1, np.log10(age) + 9, fe_h,
                distance=distance, bands=["G", "BP", "RP"],
            )
            bp = props.BP_mag.values[0]
            g = props.G_mag.values[0]
            rp = props.RP_mag.values[0]
            plx = 1000.0 / distance

            if add_noise:
                errs = self._get_errors(bp, rp, g, plx)
                bp = np.random.normal(bp, errs["BP"])
                rp = np.random.normal(rp, errs["RP"])
                g = np.random.normal(g, errs["G"])
                plx = np.random.normal(plx, errs["parallax"])

            return np.array([bp, rp, g, plx])

        else:
            props = self.tracks.generate_binary(
                m1, q * m1, np.log10(age) + 9, fe_h, bands=["G", "BP", "RP"],
            )
            bp = props.BP_mag.values[0]
            g = props.G_mag.values[0]
            rp = props.RP_mag.values[0]

            if add_noise:
                # Approximate apparent mags for error estimation (~400 pc)
                dm = 8.0
                errs = self._get_errors(bp + dm, rp + dm, g + dm)
                bp = np.random.normal(bp, errs["BP"])
                rp = np.random.normal(rp, errs["RP"])
                g = np.random.normal(g, errs["G"])

            return np.array([bp, rp, g])

    def make_simulator(
        self,
        fixed_params: Dict,
        use_apparent_mags: bool,
    ):
        """
        Return a single-sample simulator compatible with ``sbi``.

        The returned callable accepts a 1-D ``theta`` tensor whose
        dimensions correspond to the *inferred* parameters in order:
        ``M1``, ``q``, then any un-fixed subset of ``age``, ``feh``,
        ``distance``.

        Parameters
        ----------
        fixed_params : dict
            Parameters held fixed.  Keys: ``'age'``, ``'feh'``,
            ``'distance'``.
        use_apparent_mags : bool
            When ``True`` the simulator returns apparent magnitudes +
            parallax; otherwise absolute magnitudes.

        Returns
        -------
        callable
            ``simulator(theta: 1-D tensor) -> 1-D tensor``
        """
        def simulator(theta):
            theta_list = theta.tolist() if hasattr(theta, "tolist") else list(theta)
            idx = 0
            m1 = theta_list[idx]; idx += 1
            q = theta_list[idx]; idx += 1

            if "age" in fixed_params:
                age = fixed_params["age"]
            else:
                age = theta_list[idx]; idx += 1

            if "feh" in fixed_params:
                fe_h = fixed_params["feh"]
            else:
                fe_h = theta_list[idx]; idx += 1

            if use_apparent_mags:
                if "distance" in fixed_params:
                    distance = fixed_params["distance"]
                else:
                    distance = theta_list[idx]; idx += 1
            else:
                distance = None

            return torch.tensor(
                self.binary_color_mag_isochrones(m1, q, age, fe_h, distance),
                dtype=torch.float32,
            )

        return simulator

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _simulate(
        self,
        sbi_simulator,
        sbi_prior,
        num_simulations: int,
        max_trials: float = np.inf,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate simulations, discarding runs that produce NaN outputs.

        Returns
        -------
        theta : torch.Tensor, shape ``(num_simulations, n_params)``
        x : torch.Tensor, shape ``(num_simulations, n_obs)``
        """
        num_trials, num_simulated = 0, 0
        theta_list: List[np.ndarray] = []
        x_list: List[np.ndarray] = []

        with tqdm(total=num_simulations, desc="Simulating") as pbar:
            while num_simulated < num_simulations:
                N = num_simulations - num_simulated
                _theta = sbi_prior.sample((N,))
                _x = sbi_simulator(_theta)
                if _x.dim() == 3:
                    _x = _x.squeeze(1)

                keep = np.all(np.isfinite(_x.numpy()), axis=1)
                theta_list.extend(np.array(_theta[keep]))
                x_list.extend(np.array(_x[keep]))

                num_trials += 1
                num_valid = int(keep.sum())
                num_simulated += num_valid
                pbar.update(num_valid)

                if num_trials > max_trials:
                    print(
                        f"Warning: exceeded max_trials ({max_trials}) with "
                        f"{num_simulated}/{num_simulations} valid simulations."
                    )
                    break

        return (
            torch.tensor(np.vstack(theta_list), dtype=torch.float32),
            torch.tensor(np.vstack(x_list), dtype=torch.float32),
        )

    def build_posterior(
        self,
        prior,
        fixed_params: Dict = None,
        use_apparent_mags: bool = True,
        num_simulations: int = 100_000,
        sample_with: str = "mcmc",
        max_trials: float = np.inf,
    ):
        """
        Run the full SBI training pipeline and return a trained posterior.

        1. Wraps the single-sample simulator with ``prepare_for_sbi``.
        2. Generates *num_simulations* valid (finite) simulations.
        3. Trains a Sequential Neural Posterior Estimator (SNPE).
        4. Builds and returns the posterior.

        The inferred parameter labels are stored on the instance as
        ``self._last_labels`` for use by :meth:`infer`.

        Parameters
        ----------
        prior : StellarPrior or compatible
            Prior distribution over the inferred parameters.
        fixed_params : dict, optional
            Parameters to hold fixed.  Keys: ``'age'``, ``'feh'``,
            ``'distance'``.
        use_apparent_mags : bool
            When ``True``, the simulator returns apparent magnitudes +
            parallax (requires *distance* to be either fixed or
            inferred).  When ``False``, absolute magnitudes.
        num_simulations : int
            Number of simulations used for training.
        sample_with : str
            Posterior sampling algorithm: ``'mcmc'`` (default) or
            ``'rejection'``.
        max_trials : float
            Maximum number of simulation rounds before raising a
            warning and stopping early.

        Returns
        -------
        posterior : sbi posterior object
        theta : torch.Tensor
            Simulated parameter vectors used for training.
        x : torch.Tensor
            Corresponding simulated observables used for training.
        """
        if fixed_params is None:
            fixed_params = {}

        self._last_labels = self._infer_labels(fixed_params, use_apparent_mags)

        simulator = self.make_simulator(fixed_params, use_apparent_mags)
        sbi_simulator, sbi_prior = prepare_for_sbi(simulator, prior)

        inference = SNPE(sbi_prior)
        theta, x = self._simulate(sbi_simulator, sbi_prior, num_simulations, max_trials)

        density_estimator = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(density_estimator, sample_with=sample_with)

        return posterior, theta, x

    # ------------------------------------------------------------------
    # Inference on observations
    # ------------------------------------------------------------------

    def infer(
        self,
        posterior,
        data,
        labels: Tuple[str, ...] = None,
        fixed_params: Dict = None,
        use_apparent_mags: bool = True,
        num_samples: int = 10_000,
        source_col: str = None,
    ) -> pd.DataFrame:
        """
        Run posterior inference on observed stellar photometry.

        Parameters
        ----------
        posterior : sbi posterior
            Trained posterior returned by :meth:`build_posterior`.
        data : pd.DataFrame or np.ndarray
            Observed photometry.  When a ``DataFrame`` is passed,
            column names are auto-detected.  Accepted column names:

            * ``BPmag`` / ``phot_bp_mean_mag``
            * ``RPmag`` / ``phot_rp_mean_mag``
            * ``Gmag`` / ``phot_g_mean_mag``
            * ``PlxCorr`` / ``parallax`` (required when
              *use_apparent_mags* is ``True``)

            Observation order must match the simulator output:
            ``[BP, RP, G]`` or ``[BP, RP, G, parallax]``.
        labels : tuple of str, optional
            Inferred parameter names in prior order, e.g.
            ``("M1", "q")`` or ``("M1", "q", "age", "[Fe/H]")``.
            Defaults to ``self._last_labels`` set by the last call to
            :meth:`build_posterior`.
        fixed_params : dict, optional
            Fixed parameters recorded in the output for reference.
        use_apparent_mags : bool
            Whether the observations are apparent magnitudes + parallax.
            When ``False``, apparent magnitudes are converted to
            absolute using the ``PlxCorr`` column.
        num_samples : int
            Posterior samples drawn per star.
        source_col : str, optional
            DataFrame column to use as the star identifier in the
            output.  Falls back to ``'Source'`` if present, otherwise
            uses a running integer index.

        Returns
        -------
        pd.DataFrame
            One row per star with columns
            ``{param}_median``, ``{param}_err_lower``,
            ``{param}_err_upper`` for each inferred parameter.
        """
        if fixed_params is None:
            fixed_params = {}

        if labels is None:
            if self._last_labels is None:
                raise ValueError(
                    "labels not provided and build_posterior() has not been "
                    "called on this instance yet."
                )
            labels = self._last_labels

        # --- normalise column names ---
        _col_map = {
            "phot_g_mean_mag":  "Gmag",
            "phot_bp_mean_mag": "BPmag",
            "phot_rp_mean_mag": "RPmag",
            "parallax":         "PlxCorr",
            "source_id":        "Source",
        }

        if isinstance(data, pd.DataFrame):
            df = data.rename(
                columns={k: v for k, v in _col_map.items() if k in data.columns}
            )
            if use_apparent_mags:
                obs = df[["BPmag", "RPmag", "Gmag", "PlxCorr"]].values
            else:
                plx = df["PlxCorr"].values
                dm = 5.0 * np.log10(plx) - 10.0
                obs = np.column_stack([
                    df["BPmag"].values + dm,
                    df["RPmag"].values + dm,
                    df["Gmag"].values + dm,
                ])

            if source_col and source_col in df.columns:
                source_ids = df[source_col].values
            elif "Source" in df.columns:
                source_ids = df["Source"].values
            else:
                source_ids = np.arange(len(df))
        else:
            obs = np.asarray(data)
            source_ids = np.arange(len(obs))

        # --- bounds for clipping MCMC samples ---
        lower_bounds = np.array([self._PARAM_BOUNDS[l][0] for l in labels])
        upper_bounds = np.array([self._PARAM_BOUNDS[l][1] for l in labels])

        _label_to_col = {
            "M1":       "M1",
            "q":        "q",
            "age":      "age",
            "[Fe/H]":   "FeH",
            "distance": "distance",
        }

        results = []
        num_failed = 0

        for obs_row, src_id in tqdm(
            zip(obs, source_ids), total=len(obs), desc="Inferring"
        ):
            obs_tensor = torch.tensor(obs_row, dtype=torch.float32)
            result: Dict = {"Source": src_id}

            try:
                samples = posterior.sample((num_samples,), x=obs_tensor)
                samples_np = np.clip(samples.numpy(), lower_bounds, upper_bounds)
                medians = np.median(samples_np, axis=0)
                p16 = np.percentile(samples_np, 16, axis=0)
                p84 = np.percentile(samples_np, 84, axis=0)

                for j, label in enumerate(labels):
                    col = _label_to_col[label]
                    result[f"{col}_median"]    = medians[j]
                    result[f"{col}_err_lower"] = medians[j] - p16[j]
                    result[f"{col}_err_upper"] = p84[j] - medians[j]

            except RuntimeError as exc:
                num_failed += 1
                print(f"\nWarning: sampling failed for Source={src_id}: {exc}")
                for label in labels:
                    col = _label_to_col[label]
                    result[f"{col}_median"]    = np.nan
                    result[f"{col}_err_lower"] = np.nan
                    result[f"{col}_err_upper"] = np.nan

            for key in ("age", "feh", "distance"):
                if key in fixed_params:
                    result[f"{key}_fixed"] = fixed_params[key]

            results.append(result)

        if num_failed:
            print(f"\n{num_failed}/{len(obs)} stars failed during sampling.")

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, posterior, theta: torch.Tensor, x: torch.Tensor, path: str):
        """
        Save the trained posterior and training data to disk.

        Parameters
        ----------
        posterior : sbi posterior
        theta : torch.Tensor
        x : torch.Tensor
        path : str
            Output file path (e.g. ``'outputs/cluster_posterior.pkl'``).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fp:
            pickle.dump((posterior, (theta, x)), fp)
        print(f"Saved posterior to {path}")

    @staticmethod
    def load(path: str):
        """
        Load a posterior previously saved with :meth:`save`.

        Returns
        -------
        posterior : sbi posterior
        (theta, x) : tuple of torch.Tensor
        """
        with open(path, "rb") as fp:
            return pickle.load(fp)


# ---------------------------------------------------------------------------
# Quick-start example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sbi.utils import BoxUniform

    elisa = ElisaBinary()
    elisa.set_tracks()

    data_path = "cluster_photometry.csv"   # must have Gmag, e_Gmag, BPmag, ...
    elisa.set_error_splines(data_path)

    fixed_params = {"age": 0.035, "feh": 0.0, "distance": 400.0}

    # Bring your own prior — here a simple uniform over (M1, q)
    prior = BoxUniform(
        low=torch.tensor([0.1, 0.0]),
        high=torch.tensor([5.0, 1.0]),
    )

    posterior, theta, x = elisa.build_posterior(
        prior=prior,
        fixed_params=fixed_params,
        use_apparent_mags=True,
        num_simulations=50_000,
    )
    elisa.save(posterior, theta, x, "cluster_posterior.pkl")

    df_obs = pd.read_csv(data_path)
    results = elisa.infer(posterior, df_obs, fixed_params=fixed_params)
    results.to_csv("cluster_results.csv", index=False)
    print(results.head())
