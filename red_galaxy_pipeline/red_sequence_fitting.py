"""
Red sequence fitting module.

Fits red sequence ridge line parameters a(z), b(z), c(z) using cubic spline
interpolation over redshift nodes. Based on Vakili et al. (2019) Section 3.3.

Color model: color(z, m) = a(z) + b(z) * (m - m_ref(z)) + scatter(z)

Where:
- a(z): Zero-point color evolution
- b(z): Color-magnitude slope evolution
- c(z): Intrinsic scatter evolution
- m_ref(z): Reference magnitude spline
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Callable
from scipy.interpolate import interp1d, CubicSpline
from iminuit import Minuit

from red_galaxy_pipeline.utils import create_color_arrays_and_covariance, parse_colors


def setup_spline_nodes(z_min: float, z_max: float,
                       delta_a: float, delta_b: float, delta_c: float) -> Dict[str, np.ndarray]:
    """
    Setup spline node positions for a(z), b(z), c(z).

    Parameters
    ----------
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    delta_a : float
        Node spacing for a(z) spline
    delta_b : float
        Node spacing for b(z) spline
    delta_c : float
        Node spacing for c(z) spline

    Returns
    -------
    nodes : dict
        Dictionary with keys 'a', 'b', 'c' containing node arrays
    """
    npt_a = int((z_max - z_min + 0.001) / delta_a) + 1
    npt_b = int((z_max - z_min + 0.001) / delta_b) + 1
    npt_c = int((z_max - z_min + 0.001) / delta_c) + 1

    return {
        'a': np.linspace(z_min, z_max, npt_a),
        'b': np.linspace(z_min, z_max, npt_b),
        'c': np.linspace(z_min, z_max, npt_c)
    }


def interpolate_parameters(z_nodes: np.ndarray, parameter_values: np.ndarray,
                          z_eval: np.ndarray) -> np.ndarray:
    """
    Interpolate parameters using cubic spline.

    Parameters
    ----------
    z_nodes : ndarray
        Redshift nodes for interpolation
    parameter_values : ndarray
        Parameter values at each node
    z_eval : ndarray
        Redshifts at which to evaluate the interpolation

    Returns
    -------
    values : ndarray
        Interpolated parameter values at z_eval
    """
    return interp1d(z_nodes, parameter_values, kind='cubic', fill_value='extrapolate')(z_eval)


class RedSequenceModel:
    """
    Objective function for joint red sequence fitting across multiple colors.

    Parameters
    ----------
    galaxy_data : tuple of (colors, mi_j, z_j)
        colors : ndarray, shape (n_colors, n_galaxies)
            Observed colors
        mi_j : ndarray, shape (n_galaxies,)
            Observed magnitudes
        z_j : ndarray, shape (n_galaxies,)
            Spectroscopic redshifts
    C_obs : ndarray, shape (n_colors, n_galaxies)
        Observational variance for each color
    mi_ref : ndarray, shape (n_galaxies,)
        Reference magnitudes
    nodes : dict
        Spline node positions with keys 'a', 'b', 'c'
    n_colors : int
        Number of colors being fitted
    """

    def __init__(self, galaxy_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 C_obs: np.ndarray, mi_ref: np.ndarray,
                 nodes: Dict[str, np.ndarray], n_colors: int):
        self.colors, self.mi_j, self.z_j = galaxy_data
        self.C_obs = C_obs
        self.mi_ref = mi_ref
        self.nodes = nodes
        self.n_colors = n_colors

        # Build parameter names list
        self.param_names = []
        for i in range(n_colors):
            self.param_names += [f'a_{i}_{j}' for j in range(len(nodes['a']))]
            self.param_names += [f'b_{i}_{j}' for j in range(len(nodes['b']))]
            self.param_names += [f'c_{i}_{j}' for j in range(len(nodes['c']))]

    def __call__(self, *params):
        """
        Evaluate objective: sum of chi^2 + 2*log(sigma) over colors.

        Parameters
        ----------
        *params : float
            Flattened array of all spline node values

        Returns
        -------
        loss : float
            Total loss (chi-squared + log-likelihood terms)
        """
        p = dict(zip(self.param_names, params))

        total_loss = 0.0
        for i in range(self.n_colors):
            # Extract parameters for this color
            a = np.array([p[f'a_{i}_{j}'] for j in range(len(self.nodes['a']))])
            b = np.array([p[f'b_{i}_{j}'] for j in range(len(self.nodes['b']))])
            c = np.array([p[f'c_{i}_{j}'] for j in range(len(self.nodes['c']))])

            # Interpolate to galaxy redshifts
            a_z = interpolate_parameters(self.nodes['a'], a, self.z_j)
            b_z = interpolate_parameters(self.nodes['b'], b, self.z_j)
            c_z = interpolate_parameters(self.nodes['c'], c, self.z_j)

            # Total variance = intrinsic^2 + observational
            sigma = np.sqrt(c_z**2 + self.C_obs[i])

            # Model prediction
            model = a_z + b_z * (self.mi_j - self.mi_ref)
            residual = self.colors[i] - model

            # Chi-squared + log-likelihood
            chi2 = np.sum(residual**2 / sigma**2) + 2 * np.sum(np.log(sigma))
            total_loss += chi2

        return total_loss


class RedSequenceModelRegularized(RedSequenceModel):
    """
    Red sequence model with optional regularization penalties.

    Parameters
    ----------
    galaxy_data : tuple of (colors, mi_j, z_j)
        Galaxy data arrays
    C_obs : ndarray
        Observational variance
    mi_ref : ndarray
        Reference magnitudes
    nodes : dict
        Spline node positions
    n_colors : int
        Number of colors
    regularization_config : dict, optional
        Regularization configuration with keys:
        - 'type': 'smoothness', 'amplitude', or 'difference'
        - 'strength': dict with keys 'a', 'b', 'c' and lambda values
        - 'apply_to': list of parameters to regularize
    """

    def __init__(self, galaxy_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 C_obs: np.ndarray, mi_ref: np.ndarray,
                 nodes: Dict[str, np.ndarray], n_colors: int,
                 regularization_config: Optional[Dict] = None):
        super().__init__(galaxy_data, C_obs, mi_ref, nodes, n_colors)
        self.reg_config = regularization_config or {}

    def __call__(self, *params):
        """
        Evaluate objective with regularization.

        Parameters
        ----------
        *params : float
            Flattened array of all spline node values

        Returns
        -------
        loss : float
            Total loss including regularization penalty
        """
        loss = super().__call__(*params)

        if not self.reg_config:
            return loss

        p = dict(zip(self.param_names, params))
        reg_type = self.reg_config.get('type', 'smoothness')
        strength = self.reg_config.get('strength', {})
        apply_to = self.reg_config.get('apply_to', ['a', 'b', 'c'])

        reg_penalty = 0.0

        for kind in apply_to:
            if kind not in strength:
                continue

            lambda_reg = strength[kind]
            n_nodes = len(self.nodes[kind])

            for i in range(self.n_colors):
                values = np.array([p[f'{kind}_{i}_{j}'] for j in range(n_nodes)])

                if reg_type == 'smoothness':
                    # Penalize second derivatives
                    if n_nodes >= 3:
                        second_deriv = np.diff(values, n=2)
                        reg_penalty += lambda_reg * np.sum(second_deriv**2)

                elif reg_type == 'amplitude':
                    # Penalize large parameter values
                    reg_penalty += lambda_reg * np.sum(values**2)

                elif reg_type == 'difference':
                    # Penalize differences between consecutive nodes
                    first_deriv = np.diff(values)
                    reg_penalty += lambda_reg * np.sum(first_deriv**2)

                else:
                    raise ValueError(f"Unknown regularization type: {reg_type}")

        # Store for later retrieval
        self._last_loss = loss
        self._last_reg_penalty = reg_penalty

        return loss + reg_penalty


def extract_named_params(model: RedSequenceModel, params: Dict[str, float],
                        colour_names: List[str], errors: Dict[str, float]) -> Dict:
    """
    Extract fit parameters organized by color name.

    Parameters
    ----------
    model : RedSequenceModel
        Fitted model instance
    params : dict
        Parameter values from minimizer
    colour_names : list of str
        Color names for organizing output
    errors : dict
        Parameter errors from minimizer

    Returns
    -------
    results : dict
        Nested dictionary with structure results[color][param]['values'/'errors'/'nodes']
    """
    results = {'colors': colour_names}

    for i, cname in enumerate(colour_names):
        results[cname] = {}

        for kind in ['a', 'b', 'c']:
            param_list = []
            err_list = []
            j = 0
            while True:
                pname = f'{kind}_{i}_{j}'
                if pname in params:
                    param_list.append(params[pname])
                else:
                    break
                if pname in errors:
                    err_list.append(errors[pname])
                j += 1

            results[cname][kind] = {
                'values': np.array(param_list),
                'errors': np.array(err_list),
                'nodes': model.nodes[kind]
            }

    return results


def fit_red_sequence(galaxy_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    C_obs: np.ndarray, mi_ref: np.ndarray,
                    z_min: float, z_max: float,
                    delta_a: float, delta_b: float, delta_c: float,
                    colour_names: Optional[List[str]] = None,
                    regularization_config: Optional[Dict] = None) -> Dict:
    """
    Fit red sequence parameters jointly across all colors.

    Parameters
    ----------
    galaxy_data : tuple of (colors, mi_j, z_j)
        Galaxy data arrays
    C_obs : ndarray
        Observational variance for each color
    mi_ref : ndarray
        Reference magnitudes
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    delta_a, delta_b, delta_c : float
        Node spacings for a(z), b(z), c(z)
    colour_names : list of str, optional
        Names for each color
    regularization_config : dict, optional
        Regularization configuration

    Returns
    -------
    results : dict
        Fitted parameters organized by color
    """
    colors, mi_j, z_j = galaxy_data
    n_colors = len(colors)
    nodes = setup_spline_nodes(z_min, z_max, delta_a, delta_b, delta_c)

    if colour_names is None:
        colour_names = [str(i) for i in range(n_colors)]

    # Choose model class
    if regularization_config:
        model = RedSequenceModelRegularized(galaxy_data, C_obs, mi_ref, nodes,
                                           n_colors, regularization_config)
    else:
        model = RedSequenceModel(galaxy_data, C_obs, mi_ref, nodes, n_colors)

    # Initial values and limits
    init = []
    limits = []
    for _ in range(n_colors):
        init += [1.0] * len(nodes['a'])
        limits += [(0.2, 3.5)] * len(nodes['a'])

        init += [0.0] * len(nodes['b'])
        limits += [(-0.5, 0.5)] * len(nodes['b'])

        init += [0.1] * len(nodes['c'])
        limits += [(0.001, 0.4)] * len(nodes['c'])

    # Fit
    m = Minuit(model, *init)
    m.errordef = 1.0
    m.limits = limits
    m.migrad()

    params = dict(zip(model.param_names, m.values))
    errors = dict(zip(model.param_names, m.errors))

    # Print final loss breakdown if regularization with verbose
    if regularization_config and regularization_config.get('verbose', False):
        _ = model(*m.values)  # Update stored values
        chi2 = model._last_loss
        reg = model._last_reg_penalty
        total = chi2 + reg
        ratio = 0 if chi2 == 0 else reg / chi2
        print(f"[Joint fit] chi2={chi2:.2f}, reg_penalty={reg:.2f}, total={total:.2f}, ratio={ratio:.6f}")

    return extract_named_params(model, params, colour_names, errors)


def fit_red_sequence_single_colour(galaxy_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                   C_obs: np.ndarray, mi_ref: np.ndarray,
                                   z_min: float, z_max: float,
                                   delta_a: float, delta_b: float, delta_c: float,
                                   colour_names: Optional[List[str]] = None,
                                   regularization_config: Optional[Dict] = None) -> Dict:
    """
    Fit red sequence parameters independently for each color.

    Parameters
    ----------
    galaxy_data : tuple of (colors, mi_j, z_j)
        Galaxy data arrays
    C_obs : ndarray
        Observational variance for each color
    mi_ref : ndarray
        Reference magnitudes
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    delta_a, delta_b, delta_c : float
        Node spacings for a(z), b(z), c(z)
    colour_names : list of str, optional
        Names for each color
    regularization_config : dict, optional
        Regularization configuration

    Returns
    -------
    results : dict
        Fitted parameters organized by color
    """
    colors, mi_j, z_j = galaxy_data
    n_colors = len(colors)
    nodes = setup_spline_nodes(z_min, z_max, delta_a, delta_b, delta_c)

    if colour_names is None:
        colour_names = [str(i) for i in range(n_colors)]

    results = {'colors': colour_names}

    # Fit each color independently
    for i in range(n_colors):
        data_i = ([colors[i]], mi_j, z_j)

        if regularization_config:
            model = RedSequenceModelRegularized(data_i, [C_obs[i]], mi_ref,
                                               nodes, 1, regularization_config)
        else:
            model = RedSequenceModel(data_i, [C_obs[i]], mi_ref, nodes, 1)

        init = ([1.0] * len(nodes['a']) +
                [0.0] * len(nodes['b']) +
                [0.1] * len(nodes['c']))
        limits = ([(0.2, 3.5)] * len(nodes['a']) +
                 [(-0.5, 0.5)] * len(nodes['b']) +
                 [(0.001, 0.4)] * len(nodes['c']))

        m = Minuit(model, *init)
        m.errordef = 1.0
        m.limits = limits
        m.migrad()

        params = dict(zip(model.param_names, m.values))
        errors = dict(zip(model.param_names, m.errors))

        # Print final loss breakdown if regularization with verbose
        if regularization_config and regularization_config.get('verbose', False):
            _ = model(*m.values)  # Update stored values
            chi2 = model._last_loss
            reg = model._last_reg_penalty
            total = chi2 + reg
            ratio = reg / chi2 if chi2 > 0 else 0
            print(f"[Color {colour_names[i]}] chi2={chi2:.2f}, reg_penalty={reg:.2f}, total={total:.2f}, ratio={ratio:.6f}")

        # Extract parameters for this color using new structure
        results[colour_names[i]] = {}
        for kind in ['a', 'b', 'c']:
            param_list = [params[f'{kind}_0_{j}'] for j in range(len(nodes[kind]))]
            err_list = [errors[f'{kind}_0_{j}'] for j in range(len(nodes[kind]))]
            results[colour_names[i]][kind] = {
                'values': np.array(param_list),
                'errors': np.array(err_list),
                'nodes': nodes[kind]
            }

    return results


def save_params(results: Dict, filepath: str,
                m_ref_z: Optional[np.ndarray] = None,
                m_ref_values: Optional[np.ndarray] = None) -> None:
    """
    Save fitted parameters to .npz file.

    Parameters
    ----------
    results : dict
        Fitted parameters from fit_red_sequence or fit_red_sequence_single_colour
    filepath : str
        Full path to output file (e.g., 'folder/ridgeline.npz')
    m_ref_z : ndarray, optional
        Redshift nodes for reference magnitude
    m_ref_values : ndarray, optional
        Reference magnitude values at each node
    """
    flat_dict = {'colors': results['colors']}

    # Save nodes ONCE (same for all colors)
    first_color = results['colors'][0]
    flat_dict['nodes_a'] = results[first_color]['a']['nodes']
    flat_dict['nodes_b'] = results[first_color]['b']['nodes']
    flat_dict['nodes_c'] = results[first_color]['c']['nodes']

    # Save values/errors per color
    for color in results['colors']:
        for kind in ['a', 'b', 'c']:
            flat_dict[f'{color}_{kind}_values'] = results[color][kind]['values']
            flat_dict[f'{color}_{kind}_errors'] = results[color][kind]['errors']

    if m_ref_z is not None and m_ref_values is not None:
        flat_dict['m_ref_z'] = m_ref_z
        flat_dict['m_ref_values'] = m_ref_values

    np.savez(filepath, **flat_dict)


def load_params(filepath: str) -> Tuple[Dict, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Load fitted parameters from .npz file.

    Parameters
    ----------
    filepath : str
        Full path to input file (e.g., 'folder/ridgeline.npz')

    Returns
    -------
    results : dict
        Fitted parameters organized by color
    m_ref_data : tuple or None
        Tuple of (m_ref_z, m_ref_values) if available, else None
    """
    data = np.load(filepath, allow_pickle=True)

    colors = data['colors'].tolist() if hasattr(data['colors'], 'tolist') else list(data['colors'])
    results = {'colors': colors}

    # Load shared nodes
    nodes = {
        'a': data['nodes_a'],
        'b': data['nodes_b'],
        'c': data['nodes_c']
    }

    # Reconstruct per-color structure
    for color in colors:
        results[color] = {}
        for kind in ['a', 'b', 'c']:
            results[color][kind] = {
                'values': data[f'{color}_{kind}_values'],
                'errors': data[f'{color}_{kind}_errors'],
                'nodes': nodes[kind]
            }

    # Load reference magnitude if available
    m_ref_data = None
    if 'm_ref_z' in data and 'm_ref_values' in data:
        m_ref_data = (data['m_ref_z'], data['m_ref_values'])

    return results, m_ref_data


def make_spline_functions(results: Dict, colour_names: Optional[List[str]] = None) -> Dict[str, Dict[str, CubicSpline]]:
    """
    Create CubicSpline functions from fit results.

    Parameters
    ----------
    results : dict
        Fitted parameters from fit_red_sequence or load_params
    colour_names : list of str, optional
        Color names to create splines for. If None, uses all colors in results.

    Returns
    -------
    spline_dict : dict
        Nested dictionary: spline_dict[color]['a'/'b'/'c'] = CubicSpline
    """
    if colour_names is None:
        colour_names = results['colors']

    spline_dict = {}

    for col in colour_names:
        spline_dict[col] = {}
        for kind in ['a', 'b', 'c']:
            z_nodes = results[col][kind]['nodes']
            values = results[col][kind]['values']
            spline_dict[col][kind] = CubicSpline(z_nodes, values)

    return spline_dict


class RedSequenceFitter:
    """
    High-level interface for red sequence fitting.

    Parameters
    ----------
    z_min : float, optional
        Minimum redshift (default: 0.05)
    z_max : float, optional
        Maximum redshift (default: 0.95)
    delta_a : float, optional
        Node spacing for a(z) spline (default: 0.05)
    delta_b : float, optional
        Node spacing for b(z) spline (default: 0.1)
    delta_c : float, optional
        Node spacing for c(z) spline (default: 0.15)
    regularization_config : dict, optional
        Regularization configuration for fitting
    """

    def __init__(self, z_min: float = 0.05, z_max: float = 0.95,
                 delta_a: float = 0.05, delta_b: float = 0.1, delta_c: float = 0.15,
                 regularization_config: Optional[Dict] = None):
        self.z_min = z_min
        self.z_max = z_max
        self.delta_a = delta_a
        self.delta_b = delta_b
        self.delta_c = delta_c
        self.regularization_config = regularization_config

        self.results = None
        self.splines = None

    def setup_galaxy_data(self, df: pd.DataFrame,
                         color_definitions: List[Tuple[str, str]],
                         magnitude_col: str = 'mag_v2',
                         z_spec_col: str = 'z_spec',
                         m_ref_func: Optional[Callable] = None) -> Tuple:
        """
        Prepare galaxy data for fitting.

        Parameters
        ----------
        df : DataFrame
            Galaxy catalog
        color_definitions : list of tuples
            Color definitions as (band1, band2) pairs
        magnitude_col : str, optional
            Reference magnitude column name (default: 'mag_v2')
        z_spec_col : str, optional
            Spectroscopic redshift column name (default: 'z_spec')
        m_ref_func : callable, optional
            Reference magnitude function m_ref(z). If None, uses observed magnitudes.

        Returns
        -------
        galaxy_data : tuple
            (color_array, mi_j, z_j) arrays
        C_obs : ndarray
            Observational variance
        mi_ref : ndarray
            Reference magnitudes
        """
        # Build color arrays with covariance
        color_array, _ = create_color_arrays_and_covariance(df, color_definitions, magnitude_col)

        # Compute diagonal observational variance for each color
        C_obs = []
        for band1, band2 in color_definitions:
            err1_col = f'mag_{band1}_err'
            err2_col = f'mag_{band2}_err'
            variance = df[err1_col].values**2 + df[err2_col].values**2
            C_obs.append(variance)
        C_obs = np.array(C_obs)

        # Extract magnitudes and redshifts
        mi_j = df[magnitude_col].values
        z_j = df[z_spec_col].values

        # Reference magnitude: use function if provided, else use observed magnitudes
        if m_ref_func is not None:
            mi_ref = m_ref_func(z_j)
        else:
            # Fallback: use observed magnitudes (makes b=0 in fit)
            mi_ref = mi_j.copy()

        galaxy_data = (color_array.T, mi_j, z_j)  # Transpose to (n_colors, n_galaxies)

        return galaxy_data, C_obs, mi_ref

    def fit(self, galaxy_data: Tuple, C_obs: np.ndarray, mi_ref: np.ndarray,
           colour_names: List[str], method: str = 'single') -> Dict:
        """
        Fit red sequence model.

        Parameters
        ----------
        galaxy_data : tuple
            (color_array, mi_j, z_j) from setup_galaxy_data
        C_obs : ndarray
            Observational variance
        mi_ref : ndarray
            Reference magnitudes
        colour_names : list of str
            Color names
        method : str, optional
            'single' for independent fits (default), 'joint' for simultaneous fit

        Returns
        -------
        results : dict
            Fitted parameters organized by color
        """
        if method == 'single':
            self.results = fit_red_sequence_single_colour(
                galaxy_data, C_obs, mi_ref,
                self.z_min, self.z_max,
                self.delta_a, self.delta_b, self.delta_c,
                colour_names, self.regularization_config
            )
        elif method == 'joint':
            self.results = fit_red_sequence(
                galaxy_data, C_obs, mi_ref,
                self.z_min, self.z_max,
                self.delta_a, self.delta_b, self.delta_c,
                colour_names, self.regularization_config
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Build spline functions
        self.splines = make_spline_functions(self.results, colour_names)

        return self.results

    def save(self, path_prefix: str) -> None:
        """
        Save fitted results.

        Parameters
        ----------
        path_prefix : str
            Path prefix for output file
        """
        if self.results is None:
            raise ValueError("No results to save. Run fit() first.")
        save_params(self.results, path_prefix)

    def load(self, path_prefix: str, colour_names: List[str]) -> None:
        """
        Load fitted results.

        Parameters
        ----------
        path_prefix : str
            Path prefix for input file
        colour_names : list of str
            Color names to create splines for
        """
        self.results, _ = load_params(path_prefix)
        self.splines = make_spline_functions(self.results, colour_names)
