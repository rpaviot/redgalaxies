"""
Utility functions for the red galaxy pipeline.

Includes color parsing, covariance matrix construction, and data processing utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from scipy.interpolate import CubicSpline, splrep, splev


def parse_colors(colors: Union[List[str], List[Tuple[str, str]]]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Parse color definitions into standardized formats.

    Parameters
    ----------
    colors : list
        Either list of color strings like ['gi', 'rz', 'iy']
        or list of band tuples like [('g','i'), ('r','z'), ('i','y')]

    Returns
    -------
    color_names : list of str
        Standardized color names (e.g., ['gi', 'rz', 'iy'])
    color_definitions : list of tuples
        Band pairs (e.g., [('g','i'), ('r','z'), ('i','y')])

    Examples
    --------
    >>> parse_colors(['gi', 'rz'])
    (['gi', 'rz'], [('g', 'i'), ('r', 'z')])

    >>> parse_colors([('g','i'), ('r','z')])
    (['gi', 'rz'], [('g', 'i'), ('r', 'z')])
    """
    if not colors:
        raise ValueError("colors list cannot be empty")

    # Check if first element is a string or tuple
    if isinstance(colors[0], str):
        # Format: ['gi', 'rz', ...]
        color_names = colors
        color_definitions = []
        for color in colors:
            if len(color) != 2:
                raise ValueError(f"Color string '{color}' must be exactly 2 characters (e.g., 'gi', 'rz')")
            color_definitions.append((color[0], color[1]))

    elif isinstance(colors[0], (tuple, list)):
        # Format: [('g','i'), ('r','z'), ...]
        color_definitions = [tuple(c) for c in colors]
        color_names = []
        for band1, band2 in color_definitions:
            if not (isinstance(band1, str) and isinstance(band2, str)):
                raise ValueError(f"Band names must be strings, got {type(band1)}, {type(band2)}")
            color_names.append(f"{band1}{band2}")

    else:
        raise ValueError("colors must be a list of strings or list of tuples")

    return color_names, color_definitions


def create_color_arrays_and_covariance(df: pd.DataFrame,
                                        color_definitions: List[Tuple[str, str]],
                                        magnitude_col: str = 'mag_v2') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create color arrays and covariance matrices from dataframe.

    Accounts for correlated errors when colors share common bands.

    Parameters
    ----------
    df : DataFrame
        Galaxy catalog with magnitude columns 'mag_{band}' and error columns 'mag_{band}_err'
    color_definitions : list of tuples
        Color definitions as band pairs, e.g., [('g','i'), ('r','z'), ('i','y')]
    magnitude_col : str
        Reference magnitude column name

    Returns
    -------
    color_array : ndarray, shape (n_galaxies, n_colors)
        Color values for each galaxy
    color_covariance : ndarray, shape (n_galaxies, n_colors, n_colors)
        Color covariance matrices including off-diagonal correlations

    Notes
    -----
    Off-diagonal covariance terms arise when two colors share a common band.
    For example, if color1 = g-i and color2 = i-y, they share band 'i',
    so Cov(color1, color2) = -sigma_i^2 (negative because of anticorrelation).
    """
    n_galaxies = len(df)
    n_colors = len(color_definitions)

    color_array = np.zeros((n_galaxies, n_colors))
    color_covariance = np.zeros((n_galaxies, n_colors, n_colors))

    # Compute colors and diagonal covariance elements
    for i, (band1, band2) in enumerate(color_definitions):
        mag1_col = f'mag_{band1}'
        mag2_col = f'mag_{band2}'
        mag1_err_col = f'mag_{band1}_err'
        mag2_err_col = f'mag_{band2}_err'

        # Check if columns exist
        for col in [mag1_col, mag2_col, mag1_err_col, mag2_err_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")

        # Color = mag1 - mag2
        color_array[:, i] = df[mag1_col] - df[mag2_col]

        # Diagonal: Var(color_i) = sigma1^2 + sigma2^2
        color_covariance[:, i, i] = df[mag1_err_col]**2 + df[mag2_err_col]**2

    # Compute off-diagonal covariance elements
    for i in range(n_colors):
        for j in range(i + 1, n_colors):
            band1_i, band2_i = color_definitions[i]
            band1_j, band2_j = color_definitions[j]

            # Find common bands between the two colors
            common_bands = set([band1_i, band2_i]) & set([band1_j, band2_j])

            if common_bands:
                # There's a common band, so covariance is non-zero
                for common_band in common_bands:
                    common_err_col = f'mag_{common_band}_err'

                    # Determine the sign of the covariance
                    # If common band appears in same position (both first or both second), cov is positive
                    # If in different positions, cov is negative
                    sign_i = 1 if band1_i == common_band else -1
                    sign_j = 1 if band1_j == common_band else -1
                    sign = sign_i * sign_j

                    # Cov(color_i, color_j) = sign * sigma_common^2
                    covariance_value = sign * df[common_err_col]**2

                    color_covariance[:, i, j] += covariance_value
                    color_covariance[:, j, i] += covariance_value  # Symmetric

    return color_array, color_covariance


def asinh_magnitudes(flux: np.ndarray, flux_err: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute asinh magnitudes and errors from fluxes.

    Parameters
    ----------
    flux : ndarray
        Flux values
    flux_err : ndarray
        Flux uncertainties

    Returns
    -------
    mag : ndarray
        Asinh magnitudes
    mag_err : ndarray
        Magnitude errors (clipped to [0, 10])

    Notes
    -----
    Uses the softening parameter b = median(flux_err) to handle low S/N sources.
    Formula: mag = -a*(arcsinh(flux/(2*b)) + log(b) - m0)
    where a = 2.5*log10(e) and m0 = log(3.631e9) for AB magnitudes.
    """
    a = 2.5 * np.log10(np.e)
    b = np.median(flux_err[~np.isnan(flux_err)])
    m0 = np.log(3.631e9)

    mag = -a * (np.arcsinh(flux / (2 * b)) + np.log(b) - m0)
    mag_err = np.clip(a * flux_err / np.sqrt(flux**2 + 4 * b**2), a_min=0, a_max=10)

    return mag, mag_err


def create_magnitude_reference(df: pd.DataFrame,
                                z_col: str = 'z_spec',
                                mag_col: str = 'mag_v2',
                                z_bins: np.ndarray = None,
                                smooth_mref: bool = False,
                                smooth_s: float = 0.15,
                                return_data: bool = False):
    """
    Create reference magnitude function m_ref(z) from median magnitudes in redshift bins.

    Parameters
    ----------
    df : DataFrame
        Galaxy catalog with redshift and magnitude columns
    z_col : str
        Column name for redshift
    mag_col : str
        Column name for reference magnitude
    z_bins : ndarray, optional
        Redshift bin edges. If None, uses 0.03 spacing from min to max redshift.
    smooth_mref : bool
        If True, use splrep smoothing instead of CubicSpline (default: False)
    smooth_s : float
        Smoothing parameter for splrep (default: 0.15)
    return_data : bool
        If True, also return the underlying (z, m_ref) arrays (default: False)

    Returns
    -------
    m_ref_func : callable
        Function that takes redshift(s) and returns reference magnitude(s)
    m_ref_z : ndarray (only if return_data=True)
        Redshift nodes for reference magnitude
    m_ref_values : ndarray (only if return_data=True)
        Reference magnitude values at each node

    Notes
    -----
    Uses cubic spline interpolation of the median magnitude in each redshift bin.
    If smooth_mref=True, uses scipy.interpolate.splrep with smoothing parameter s.
    """
    if z_bins is None:
        z_min = df[z_col].min()
        z_max = df[z_col].max()
        z_bins = np.arange(z_min, z_max + 0.03, 0.03)

    z_mid = (z_bins[1:] + z_bins[:-1]) / 2.0
    m_ref_values = np.zeros(len(z_mid))

    for i in range(len(z_mid)):
        mask = (df[z_col] >= z_bins[i]) & (df[z_col] < z_bins[i+1])
        if mask.sum() > 0:
            m_ref_values[i] = df.loc[mask, mag_col].median()
        else:
            # If no galaxies in bin, interpolate from neighbors
            m_ref_values[i] = np.nan

    # Fill NaN values with linear interpolation
    valid_mask = ~np.isnan(m_ref_values)
    if valid_mask.sum() < 2:
        raise ValueError("Not enough valid redshift bins to create magnitude reference")

    m_ref_values = np.interp(z_mid, z_mid[valid_mask], m_ref_values[valid_mask])

    # Create spline function
    if smooth_mref:
        # Use splrep with smoothing
        tck = splrep(z_mid, m_ref_values, s=smooth_s)
        m_ref_func = lambda z: splev(z, tck)
    else:
        # Use cubic spline interpolation (no smoothing)
        spline = CubicSpline(z_mid, m_ref_values)
        m_ref_func = spline

    if return_data:
        return m_ref_func, z_mid, m_ref_values
    return m_ref_func
