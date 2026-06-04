"""
Photometric redshift estimation using red sequence colors.

Bayesian photo-z with Schechter luminosity function priors.
Based on Vakili et al. (2019) Section 3.4-3.7.

See README.md for detailed documentation and usage examples.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, Callable, List, Any
from scipy.linalg import det, inv
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize, differential_evolution
from iminuit import Minuit
from astropy.cosmology import FlatLambdaCDM
from joblib import Parallel, delayed

from red_galaxy_pipeline.utils import create_color_arrays_and_covariance


# Default cosmology
DEFAULT_COSMO = FlatLambdaCDM(Om0=0.30, H0=100)


def build_offset_interp(z_nodes: np.ndarray, values: np.ndarray,
                        interp: str = 'cubic'):
    """Build the Δz(z) offset function with the requested interpolation family.

    'cubic' (default, CubicSpline) or 'linear' (piecewise-linear). Both
    extrapolate linearly/flat-ish at the edges.
    """
    if interp == 'linear':
        return interp1d(z_nodes, values, kind='linear', fill_value='extrapolate')
    return CubicSpline(z_nodes, values, extrapolate=True)


def fit_redshift_offset(z_photo: np.ndarray, z_spec: np.ndarray,
                       z_nodes: np.ndarray, interp: str = 'cubic') -> CubicSpline:
    """
    Fit spline offset to minimize photo-z bias (L1 norm).

    Parameters
    ----------
    z_photo : ndarray
        Photometric redshifts
    z_spec : ndarray
        Spectroscopic redshifts
    z_nodes : ndarray
        Redshift nodes for the offset spline
    interp : {'cubic', 'linear'}
        Interpolation family for the offset Δz(z) (default: cubic).

    Returns
    -------
    offset_func : callable
        Offset function Δz(z) such that z_corrected = z_photo + Δz(z_photo)
    """
    def objective(delta_z_vals):
        delta_z_spline = build_offset_interp(z_nodes, delta_z_vals, interp)
        corrected_z = z_photo + delta_z_spline(z_photo)
        return np.sum(np.abs(z_spec - corrected_z))  # L1 norm

    # Initial guess: zero offset
    x0 = np.zeros_like(z_nodes)
    res = minimize(objective, x0, method='L-BFGS-B')

    return build_offset_interp(z_nodes, res.x, interp)


def bias_and_scatter(z_bins: np.ndarray, z_spec: np.ndarray,
                    z_photo: np.ndarray,
                    l_ratio: Optional[np.ndarray] = None,
                    l_threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute photo-z bias and NMAD scatter in redshift bins.

    Parameters
    ----------
    z_bins : ndarray
        Redshift bin edges
    z_spec : ndarray
        Spectroscopic redshifts
    z_photo : ndarray
        Photometric redshifts
    l_ratio : ndarray, optional
        Luminosity ratio L/L* for each galaxy
    l_threshold : float, optional
        Luminosity threshold for selecting galaxies

    Returns
    -------
    bin_centers : ndarray
        Redshift bin centers
    bias_arr : ndarray
        Median bias in each bin
    scatter_arr : ndarray
        NMAD scatter in each bin
    n_total : int
        Total number of galaxies used
    """
    # Apply luminosity cut if requested
    if l_ratio is not None and l_threshold is not None:
        lum_mask = l_ratio > l_threshold
        z_spec = z_spec[lum_mask]
        z_photo = z_photo[lum_mask]
        n_total = len(z_spec)
    else:
        n_total = len(z_spec)

    n_bins = len(z_bins) - 1
    bias_arr = np.zeros(n_bins)
    scatter_arr = np.zeros(n_bins)

    for i in range(n_bins):
        # Select galaxies in this photo-z bin
        mask = (z_photo > z_bins[i]) & (z_photo < z_bins[i+1])

        if np.sum(mask) < 5:  # Need minimum galaxies for statistics
            bias_arr[i] = np.nan
            scatter_arr[i] = np.nan
            continue

        z_true = z_spec[mask]
        z_obs = z_photo[mask]

        # Bias: median offset
        delta_z = z_obs - z_true
        bias_arr[i] = np.median(delta_z)

        # NMAD scatter: normalized median absolute deviation
        delta_z_norm = delta_z / (1.0 + z_true)
        med_delta = np.median(delta_z_norm)
        scatter_arr[i] = 1.48 * np.median(np.abs(delta_z_norm - med_delta))

    # Bin centers
    bin_centers = (z_bins[:-1] + z_bins[1:]) / 2.0

    return bin_centers, bias_arr, scatter_arr, n_total


def compute_luminosity_diagnostics(z_photo: np.ndarray, z_spec: np.ndarray,
                                   m_obs: np.ndarray, schechter: 'SchechterFunction',
                                   z_bins: np.ndarray,
                                   l_thresholds: List[float] = [0.5, 1.0],
                                   use_spec_for_lum: bool = False) -> Dict:
    """
    Compute photo-z diagnostics stratified by luminosity ratio L/L*.

    Parameters
    ----------
    z_photo : ndarray
        Photometric redshifts
    z_spec : ndarray
        Spectroscopic redshifts
    m_obs : ndarray
        Observed magnitudes
    schechter : SchechterFunction
        Schechter function for computing L/L*
    z_bins : ndarray
        Redshift bin edges
    l_thresholds : list of float, optional
        Luminosity thresholds to evaluate (default: [0.5, 1.0])
    use_spec_for_lum : bool, optional
        If True, use z_spec for L/L* computation (default: False)

    Returns
    -------
    diagnostics : dict
        Dictionary with 'all' and 'l_XX' keys containing bias/scatter arrays
    """
    # Compute L/L* for all galaxies
    # Use z_spec for training (true luminosity) or z_photo for validation/application
    z_for_lum = z_spec if use_spec_for_lum else z_photo
    l_ratio = schechter.compute_luminosity_ratio(m_obs, z_for_lum)

    diagnostics = {
        'l_ratio': l_ratio
    }

    # Diagnostics for all galaxies
    z_centers, bias_all, scatter_all, n_all = bias_and_scatter(
        z_bins, z_spec, z_photo
    )
    diagnostics['all'] = {
        'bias': bias_all,
        'scatter': scatter_all,
        'z_bins': z_centers,
        'n_galaxies': n_all
    }

    # Diagnostics for each luminosity threshold
    for l_thresh in l_thresholds:
        key = f'l_{l_thresh}'.replace('.', '')  # e.g., 'l_05' or 'l_10'
        z_centers, bias, scatter, n_gal = bias_and_scatter(
            z_bins, z_spec, z_photo,
            l_ratio=l_ratio, l_threshold=l_thresh
        )
        diagnostics[key] = {
            'bias': bias,
            'scatter': scatter,
            'z_bins': z_centers,
            'n_galaxies': n_gal,
            'threshold': l_thresh
        }

    return diagnostics


def save_offset_function(offset_func: CubicSpline, z_nodes: np.ndarray,
                        train_metrics: Dict, val_metrics: Dict,
                        filepath: str,
                        luminosity_diagnostics: Optional[Dict] = None,
                        interp: str = 'cubic'):
    """
    Save offset function and calibration metrics to .npz file.

    Parameters
    ----------
    offset_func : CubicSpline
        Fitted offset function
    z_nodes : ndarray
        Redshift nodes for the offset spline
    train_metrics : dict
        Training set metrics (bias, scatter, z_bins)
    val_metrics : dict
        Validation set metrics (bias, scatter, z_bins)
    filepath : str
        Output file path
    luminosity_diagnostics : dict, optional
        Luminosity-stratified diagnostics
    """
    offset_values = offset_func(z_nodes)

    # Base metrics (always saved)
    save_dict = {
        'z_nodes': z_nodes,
        'offset_values': offset_values,
        'train_bias': train_metrics.get('bias', np.array([])),
        'train_scatter': train_metrics.get('scatter', np.array([])),
        'train_z_bins': train_metrics.get('z_bins', np.array([])),
        'val_bias': val_metrics.get('bias', np.array([])),
        'val_scatter': val_metrics.get('scatter', np.array([])),
        'val_z_bins': val_metrics.get('z_bins', np.array([])),
        'offset_interp': np.array(interp),
    }

    # Add luminosity diagnostics if provided
    if luminosity_diagnostics:
        # Train set luminosity metrics
        train_lum = luminosity_diagnostics.get('train', {})
        for key, value in train_lum.items():
            if key != 'all' and key != 'l_ratio':  # Skip 'all' (already saved above)
                save_dict[f'train_{key}_bias'] = value.get('bias', np.array([]))
                save_dict[f'train_{key}_scatter'] = value.get('scatter', np.array([]))
                save_dict[f'train_{key}_n'] = value.get('n_galaxies', 0)

        # Validation set luminosity metrics
        val_lum = luminosity_diagnostics.get('val', {})
        for key, value in val_lum.items():
            if key != 'all' and key != 'l_ratio':
                save_dict[f'val_{key}_bias'] = value.get('bias', np.array([]))
                save_dict[f'val_{key}_scatter'] = value.get('scatter', np.array([]))
                save_dict[f'val_{key}_n'] = value.get('n_galaxies', 0)

    np.savez(filepath, **save_dict)


def load_offset_function(filepath: str) -> Tuple[CubicSpline, Dict]:
    """
    Load offset function and metrics from .npz file.

    Parameters
    ----------
    filepath : str
        Input file path

    Returns
    -------
    offset_func : CubicSpline
        Offset function Δz(z)
    metrics : dict
        Dictionary with train/validation bias and scatter arrays
    """
    data = np.load(filepath)

    # Reconstruct spline with the persisted interpolation family (legacy files
    # default to cubic).
    z_nodes = data['z_nodes']
    offset_values = data['offset_values']
    interp = str(data['offset_interp']) if 'offset_interp' in data.files else 'cubic'
    offset_func = build_offset_interp(z_nodes, offset_values, interp)

    # Extract base metrics (always present)
    metrics = {
        'train_bias': data['train_bias'],
        'train_scatter': data['train_scatter'],
        'train_z_bins': data['train_z_bins'],
        'val_bias': data['val_bias'],
        'val_scatter': data['val_scatter'],
        'val_z_bins': data['val_z_bins']
    }

    # Extract luminosity metrics if present (backward compatible)
    for key in data.files:
        if key.startswith('train_l') or key.startswith('val_l'):
            metrics[key] = data[key]

    return offset_func, metrics


class SchechterFunction:
    """
    Schechter luminosity function with redshift-dependent m*(z), α(z), φ(z).

    Parameters
    ----------
    z_nodes : ndarray
        Redshift nodes for interpolation
    m_star : ndarray
        Characteristic magnitude m*(z) at each node
    alpha : ndarray, optional
        Faint-end slope α(z) at each node. If None, uses constant -1.0.
    phi : ndarray, optional
        Normalization φ(z) at each node. If None, uses constant 1.0.
    """

    def __init__(self, z_nodes: np.ndarray, m_star: np.ndarray,
                 alpha: Optional[np.ndarray] = None,
                 phi: Optional[np.ndarray] = None):
        if len(z_nodes) != len(m_star):
            raise ValueError("z_nodes and m_star must have same length")

        # Store m_star interpolation
        self.m_star = CubicSpline(z_nodes, m_star)

        # Alpha: use array or constant
        if alpha is not None:
            if len(alpha) != len(z_nodes):
                raise ValueError("alpha must have same length as z_nodes")
            self.alpha = CubicSpline(z_nodes, alpha)
        else:
            self.alpha = lambda z: -1.0

        # Phi: use array or constant
        if phi is not None:
            if len(phi) != len(z_nodes):
                raise ValueError("phi must have same length as z_nodes")
            self.phi = CubicSpline(z_nodes, phi)
        else:
            self.phi = lambda z: 1.0

    def log_pdf(self, m: float, z: float) -> float:
        """
        Compute log of Schechter function (unnormalized).

        Parameters
        ----------
        m : float
            Observed magnitude
        z : float
            Redshift

        Returns
        -------
        log_prob : float
            Log probability density
        """
        alpha_z = self.alpha(z)
        m_star_z = self.m_star(z)

        delta_m = m - m_star_z
        # L/L* = 10^(-0.4 * delta_m)
        l_ratio = np.exp(-0.4 * delta_m * np.log(10))

        # log(φ(L)) = log(L/L*) * (alpha + 1) - (L/L*)
        # In magnitudes: log(φ(m)) = -0.4 * (alpha + 1) * delta_m * ln(10) - l_ratio
        log_prob = -0.4 * (alpha_z + 1) * delta_m * np.log(10) - l_ratio

        return log_prob

    def log_normalization(self, z: float) -> float:
        """
        Compute log(φ(z)) normalization.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        log_phi : float
            Log of normalization factor
        """
        return np.log(self.phi(z))

    def compute_luminosity_ratio(self, m: Union[float, np.ndarray],
                                 z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute L/L* = 10^(-0.4 * (m - m*(z))).

        Parameters
        ----------
        m : float or ndarray
            Observed magnitude(s)
        z : float or ndarray
            Redshift(s)

        Returns
        -------
        l_ratio : float or ndarray
            Luminosity ratio L/L*
        """
        return 10 ** (-0.4 * (m - self.m_star(z)))


class PhotoZEstimator:
    """
    Bayesian photo-z estimator with Schechter and volume priors.

    Parameters
    ----------
    spline_functions : dict
        Nested dictionary of color splines: spline_functions[color]['a'/'b'/'c']
    m_ref_func : callable
        Reference magnitude function m_ref(z)
    colour_names : list of str
        Color names
    schechter : SchechterFunction, optional
        Schechter luminosity function (default: constant m* = 20)
    cosmology : FlatLambdaCDM, optional
        Cosmology for volume prior (default: Om0=0.3, H0=100)
    z_min : float, optional
        Minimum redshift (default: 0.05)
    z_max : float, optional
        Maximum redshift (default: 1.0)
    offset_func : callable, optional
        Offset function Δz(z) for bias correction
    apply_offset : bool, optional
        If True, apply offset correction (default: False)
    """

    def __init__(self, spline_functions: Dict[str, Dict[str, CubicSpline]],
                 m_ref_func: Callable,
                 colour_names: List[str],
                 schechter: Optional[SchechterFunction] = None,
                 cosmology: Optional[FlatLambdaCDM] = None,
                 z_min: float = 0.05,
                 z_max: float = 1.0,
                 offset_func: Optional[Callable] = None,
                 apply_offset: bool = False,
                 r_splines: Optional[Dict[Tuple[int, int], CubicSpline]] = None):
        self.splines = spline_functions
        self.m_ref_func = m_ref_func
        self.colour_names = colour_names
        self.schechter = schechter or SchechterFunction()
        self.cosmo = cosmology or DEFAULT_COSMO
        self.z_min = z_min
        self.z_max = z_max
        self.offset_func = offset_func
        self.apply_offset = apply_offset
        # Optional cross-covariance r(z) splines, keyed by colour-index pair.
        self.r_splines = r_splines or {}

        if self.apply_offset and self.offset_func is None:
            raise ValueError("apply_offset=True requires offset_func to be provided")

    def color_model(self, z: float, m: float) -> np.ndarray:
        """
        Predict colors at given redshift and magnitude.

        Parameters
        ----------
        z : float
            Redshift
        m : float
            Observed magnitude

        Returns
        -------
        colors : ndarray
            Predicted color values
        """
        delta_m = m - self.m_ref_func(z)
        return np.array([
            self.splines[col]['a'](z) + self.splines[col]['b'](z) * delta_m
            for col in self.colour_names
        ])

    def intrinsic_covariance(self, z: float) -> np.ndarray:
        """
        Compute intrinsic color scatter covariance matrix.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        C_int : ndarray, shape (n_colors, n_colors)
            Diagonal covariance matrix of intrinsic scatter
        """
        c_z = np.array([self.splines[col]['c'](z) for col in self.colour_names])
        C_int = np.diag(c_z ** 2)
        # Off-diagonals from cross-covariance r(z) splines, if present.
        for (i, j), r_spl in self.r_splines.items():
            cov_ij = float(r_spl(z)) * c_z[i] * c_z[j]
            C_int[i, j] = cov_ij
            C_int[j, i] = cov_ij
        return C_int

    def chi_squared(self, c_obs: np.ndarray, c_model: np.ndarray,
                   C_tot: np.ndarray) -> float:
        """
        Compute Mahalanobis distance (chi-squared).

        Parameters
        ----------
        c_obs : ndarray
            Observed colors
        c_model : ndarray
            Model colors
        C_tot : ndarray
            Total covariance matrix

        Returns
        -------
        chi2 : float
            Chi-squared value
        """
        delta = c_obs - c_model
        return delta @ inv(C_tot) @ delta

    def comoving_volume_log_prior(self, z: float) -> float:
        """
        Compute log(dV/dz) comoving volume prior.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        log_dV_dz : float
            Log of differential comoving volume
        """
        return np.log(self.cosmo.differential_comoving_volume(z).value)

    def objective_function(self, z: float, c_obs: np.ndarray, m_obs: float,
                          C_obs: np.ndarray) -> float:
        """
        Compute -2 ln p(z|c,m) for photo-z estimation.

        Parameters
        ----------
        z : float
            Redshift
        c_obs : ndarray
            Observed colors
        m_obs : float
            Observed magnitude
        C_obs : ndarray
            Observational covariance matrix

        Returns
        -------
        objective : float
            Negative log-posterior (to be minimized)
        """
        z = float(z)

        # Total covariance
        C_int = self.intrinsic_covariance(z)
        C_tot = C_obs + C_int

        # Color chi-squared
        c_model = self.color_model(z, m_obs)
        chi2 = self.chi_squared(c_obs, c_model, C_tot)

        # Log determinant term
        log_det_C = np.log(det(C_tot))

        # Comoving volume prior
        log_dV_dz = self.comoving_volume_log_prior(z)

        # Schechter function
        log_p_m_z = self.schechter.log_pdf(m_obs, z)
        log_phi_z = self.schechter.log_normalization(z)

        # Total objective (Vakili+19 Eq. 26: −2 ln|dV/dz| with bars inside the log;
        # dV/dz > 0 so the bars are vacuous and the term is just −2 ln(dV/dz)).
        objective = (chi2 + log_det_C
                    - 2 * log_dV_dz
                    - 2 * log_p_m_z
                    - 2 * log_phi_z)

        return objective

    def estimate(self, c_obs: np.ndarray, m_obs: float, C_obs: np.ndarray,
                z_init: float = 0.5, method: str = 'iminuit',
                return_chi2: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Estimate photo-z for a single galaxy.

        Parameters
        ----------
        c_obs : ndarray
            Observed colors
        m_obs : float
            Observed magnitude
        C_obs : ndarray
            Observational covariance matrix
        z_init : float, optional
            Initial redshift guess (default: 0.5)
        method : str, optional
            Optimization method: 'iminuit' or 'differential_evolution' (default: 'iminuit')
        return_chi2 : bool, optional
            If True, also return chi-squared value (default: False)

        Returns
        -------
        z_best : float
            Best-fit photometric redshift
        chi2 : float (only if return_chi2=True)
            Chi-squared at best-fit redshift
        """
        if method == 'iminuit':
            z_best = self._estimate_iminuit(c_obs, m_obs, C_obs, z_init)
        elif method == 'differential_evolution':
            z_best = self._estimate_differential_evolution(c_obs, m_obs, C_obs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iminuit' or 'differential_evolution'")

        # Apply offset correction if enabled
        if self.apply_offset and not np.isnan(z_best):
            z_best = z_best + self.offset_func(z_best)

        # Compute chi2 if requested
        if return_chi2:
            if np.isnan(z_best):
                chi2 = np.nan
            else:
                C_int = self.intrinsic_covariance(z_best)
                C_tot = C_obs + C_int
                c_model = self.color_model(z_best, m_obs)
                chi2 = self.chi_squared(c_obs, c_model, C_tot)
            return z_best, chi2
        else:
            return z_best

    def _estimate_iminuit(self, c_obs: np.ndarray, m_obs: float,
                         C_obs: np.ndarray, z_init: float) -> float:
        """
        Estimate photo-z using iminuit.

        Parameters
        ----------
        c_obs : ndarray
            Observed colors
        m_obs : float
            Observed magnitude
        C_obs : ndarray
            Observational covariance matrix
        z_init : float
            Initial redshift guess

        Returns
        -------
        z_best : float
            Best-fit redshift (or NaN if fit failed)
        """
        def wrapped_nll(z):
            return self.objective_function(z, c_obs, m_obs, C_obs)

        m = Minuit(wrapped_nll, z=z_init)
        m.limits['z'] = (self.z_min, self.z_max)
        m.errordef = Minuit.LIKELIHOOD
        m.migrad()

        if not m.valid:
            return np.nan

        return m.values['z']

    def _estimate_differential_evolution(self, c_obs: np.ndarray, m_obs: float,
                                         C_obs: np.ndarray) -> float:
        """
        Estimate photo-z using scipy's differential_evolution (global optimizer).

        Parameters
        ----------
        c_obs : ndarray
            Observed colors
        m_obs : float
            Observed magnitude
        C_obs : ndarray
            Observational covariance matrix

        Returns
        -------
        z_best : float
            Best-fit redshift (or NaN if fit failed)
        """
        def wrapped_nll(z):
            return self.objective_function(z[0], c_obs, m_obs, C_obs)

        bounds = [(self.z_min, self.z_max)]
        result = differential_evolution(wrapped_nll, bounds=bounds)

        if not result.success:
            return np.nan

        return result.x[0]

    def estimate_batch(self, df: pd.DataFrame,
                      color_definitions: List[Tuple[str, str]],
                      magnitude_col: str = 'mag_v2',
                      z_init: float = 0.5,
                      method: str = 'iminuit',
                      n_jobs: int = 1,
                      return_chi2: bool = False,
                      verbose: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate photo-z for a batch of galaxies (supports parallel processing).

        Parameters
        ----------
        df : DataFrame
            Galaxy catalog
        color_definitions : list of tuples
            Color definitions as (band1, band2) pairs
        magnitude_col : str, optional
            Magnitude column name (default: 'mag_v2')
        z_init : float, optional
            Initial redshift guess (default: 0.5)
        method : str, optional
            Optimization method (default: 'iminuit')
        n_jobs : int, optional
            Number of parallel jobs (default: 1)
        return_chi2 : bool, optional
            If True, also return chi-squared values (default: False)
        verbose : bool, optional
            Print progress (default: True)

        Returns
        -------
        z_photo : ndarray
            Photometric redshifts
        chi2_array : ndarray (only if return_chi2=True)
            Chi-squared values
        """
        # Prepare data
        color_array, color_covariance = create_color_arrays_and_covariance(
            df, color_definitions, magnitude_col
        )

        n_galaxies = len(df)
        m_obs_array = df[magnitude_col].values

        # Prepare data as list of tuples for parallel processing
        galaxy_data = list(zip(color_array, m_obs_array, color_covariance))

        if n_jobs == 1:
            # Sequential processing with progress output
            z_photo = np.zeros(n_galaxies)
            chi2_array = np.zeros(n_galaxies) if return_chi2 else None

            for i, (c_obs, m_obs, C_obs) in enumerate(galaxy_data):
                result = self.estimate(c_obs, m_obs, C_obs, z_init, method=method,
                                      return_chi2=return_chi2)

                if return_chi2:
                    z_photo[i], chi2_array[i] = result
                else:
                    z_photo[i] = result

                if verbose and (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{n_galaxies} galaxies")

        else:
            # Parallel processing using joblib
            if verbose:
                print(f"Running photo-z estimation in parallel with {n_jobs} jobs...")

            def process_galaxy(c_obs, m_obs, C_obs):
                return self.estimate(c_obs, m_obs, C_obs, z_init, method=method,
                                   return_chi2=return_chi2)

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_galaxy)(c, m, C) for c, m, C in galaxy_data
            )

            if return_chi2:
                z_photo, chi2_array = zip(*results)
                z_photo = np.array(z_photo)
                chi2_array = np.array(chi2_array)
            else:
                z_photo = np.array(results)

        if return_chi2:
            return z_photo, chi2_array
        else:
            return z_photo


def create_schechter_from_params(params: Dict[str, np.ndarray]) -> SchechterFunction:
    """
    Create SchechterFunction from dict with z_nodes, m_star, and optional alpha/phi.

    Parameters
    ----------
    params : dict
        Dictionary with keys:
        - 'z_nodes': ndarray of redshift nodes
        - 'm_star': ndarray of m*(z) values
        - 'alpha': ndarray of α(z) values (optional)
        - 'phi': ndarray of φ(z) values (optional)

    Returns
    -------
    schechter : SchechterFunction
        Configured Schechter function
    """
    if 'z_nodes' not in params:
        raise ValueError("params must contain 'z_nodes'")
    if 'm_star' not in params:
        raise ValueError("params must contain 'm_star'")

    z_nodes = params['z_nodes']
    m_star = params['m_star']
    alpha = params.get('alpha', None)
    phi = params.get('phi', None)

    return SchechterFunction(z_nodes=z_nodes, m_star=m_star, alpha=alpha, phi=phi)


class PhotoZFitter:
    """
    High-level interface wrapping PhotoZEstimator for batch processing.

    Parameters
    ----------
    spline_functions : dict
        Nested dictionary of color splines
    m_ref_func : callable
        Reference magnitude function m_ref(z)
    colour_names : list of str
        Color names
    schechter_params : dict
        Schechter function parameters
    cosmology : FlatLambdaCDM, optional
        Cosmology for volume prior
    z_min : float, optional
        Minimum redshift (default: 0.05)
    z_max : float, optional
        Maximum redshift (default: 1.0)
    offset_func : callable, optional
        Offset function for bias correction
    apply_offset : bool, optional
        If True, apply offset correction (default: False)
    """

    def __init__(self, spline_functions: Dict[str, Dict[str, CubicSpline]],
                 m_ref_func: Callable,
                 colour_names: List[str],
                 schechter_params: Dict,
                 cosmology: Optional[FlatLambdaCDM] = None,
                 z_min: float = 0.05,
                 z_max: float = 1.0,
                 offset_func: Optional[Callable] = None,
                 apply_offset: bool = False,
                 r_splines: Optional[Dict[Tuple[int, int], CubicSpline]] = None):
        # Create Schechter function
        if schechter_params is None:
            raise ValueError(
                "schechter_params is required for photo-z estimation. "
                "Must provide a dict with 'z_nodes' and 'm_star' arrays. "
                "Example: {'z_nodes': z_array, 'm_star': mstar_array}"
            )
        schechter = create_schechter_from_params(schechter_params)

        # Create estimator
        self.estimator = PhotoZEstimator(
            spline_functions, m_ref_func, colour_names,
            schechter=schechter, cosmology=cosmology,
            z_min=z_min, z_max=z_max,
            offset_func=offset_func, apply_offset=apply_offset,
            r_splines=r_splines,
        )

        self.colour_names = colour_names

    def estimate_photoz(self, df: pd.DataFrame,
                       color_definitions: List[Tuple[str, str]],
                       magnitude_col: str = 'mag_v2',
                       z_init: float = 0.5,
                       method: str = 'iminuit',
                       n_jobs: int = 1,
                       return_chi2: bool = False,
                       verbose: bool = True) -> pd.DataFrame:
        """
        Estimate photo-z for catalog, returns df with 'z_phot' column.

        Parameters
        ----------
        df : DataFrame
            Galaxy catalog
        color_definitions : list of tuples
            Color definitions as (band1, band2) pairs
        magnitude_col : str, optional
            Magnitude column name (default: 'mag_v2')
        z_init : float, optional
            Initial redshift guess (default: 0.5)
        method : str, optional
            Optimization method (default: 'iminuit')
        n_jobs : int, optional
            Number of parallel jobs (default: 1)
        return_chi2 : bool, optional
            If True, add 'chi2' column (default: False)
        verbose : bool, optional
            Print progress (default: True)

        Returns
        -------
        df_out : DataFrame
            Input catalog with added 'z_phot' column (and 'chi2' if requested)
        """
        result = self.estimator.estimate_batch(
            df, color_definitions, magnitude_col, z_init, method,
            n_jobs, return_chi2, verbose
        )

        df_out = df.copy()

        if return_chi2:
            z_photo, chi2_array = result
            df_out['z_phot'] = z_photo
            df_out['chi2'] = chi2_array
        else:
            df_out['z_phot'] = result

        return df_out
