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
from scipy.interpolate import interp1d, CubicSpline, splrep, splev
from iminuit import Minuit

from red_galaxy_pipeline.utils import create_color_arrays_and_covariance, parse_colors


def setup_spline_nodes(z_min: float, z_max: float,
                       delta_a: float, delta_b: float, delta_c: float,
                       cap_last_node: bool = False) -> Dict[str, np.ndarray]:
    """
    Setup spline node positions for a(z), b(z), c(z).

    Nodes are placed at exact spacing ``delta`` starting from ``z_min``. The
    upper edge is extended past ``z_max`` if needed so the last node is
    guaranteed to be ``>= z_max`` — the data range is always covered without
    relying on extrapolation. This mirrors the convention used at the GMM
    selection stage (step 1), where ``z_max=1.05`` becomes 1.07 when
    ``delta=0.03``.

    The caller controls where the nodes land by passing the clipping range
    (``df[z_spec].min()/.max()``, or an explicit ``node_z_min/node_z_max``
    science window) as ``z_min``/``z_max`` — node placement is decided there,
    not by edge-anchoring inside this grid.

    Parameters
    ----------
    cap_last_node : bool, optional
        If True, clamp any last node that overshoots ``z_max`` back to
        ``z_max``. With the default spacings only c(z) (Δ=0.15) overshoots
        (last node 1.10 for a [0.05, 1.05] window); capping anchors that node
        at 1.05 where galaxies still live, instead of letting it float into the
        empty high-z region. a(z)/b(z) already land on z_max so are unaffected.

    Returns
    -------
    nodes : dict
        Dictionary with keys 'a', 'b', 'c' containing node arrays.
    """
    def _grid(delta: float) -> np.ndarray:
        # ceil with a tiny tolerance so an exact multiple doesn't add a node.
        n_steps = int(np.ceil((z_max - z_min) / delta - 1e-9))
        grid = z_min + np.arange(n_steps + 1) * delta
        if cap_last_node and grid[-1] > z_max:
            grid[-1] = z_max
        return grid

    return {'a': _grid(delta_a), 'b': _grid(delta_b), 'c': _grid(delta_c)}


def setup_r_nodes(z_min: float, z_max: float, delta_r: float) -> np.ndarray:
    """Spline node grid for the cross-covariance correlation coefficient r(z).

    Same convention as ``setup_spline_nodes``: nodes at exact spacing
    ``delta_r`` from ``z_min``, last node extended past ``z_max`` if needed so
    the data range is covered without extrapolation.
    """
    n_steps = int(np.ceil((z_max - z_min) / delta_r - 1e-9))
    return z_min + np.arange(n_steps + 1) * delta_r


def find_correlated_pairs(color_definitions: List[Tuple[str, str]]) -> List[Tuple[int, int]]:
    """Return color-index pairs (i, j) whose colors share at least one band.

    These are the only pairs that get a free r(z) cross-correlation term;
    non-adjacent pairs (no shared band) are held at r=0 per Rykoff+14 eq. 40.
    """
    pairs = []
    for i in range(len(color_definitions)):
        for j in range(i + 1, len(color_definitions)):
            shared = set(color_definitions[i]) & set(color_definitions[j])
            if shared:
                pairs.append((i, j))
    return pairs


# Default interpolation method for a(z)/b(z)/c(z) between spline nodes.
# 'cubic'  : scipy interp1d cubic with linear-ish extrapolation
#            (fill_value='extrapolate'). DEFAULT.
# 'linear' : piecewise-linear, no overshoot, mild linear edge extrapolation.
# Post-hoc smoothing (smooth_s > 0) applies on top of either.
INTERP_METHOD = 'cubic'


def build_param_interp(z_nodes: np.ndarray, parameter_values: np.ndarray,
                       method: Optional[str] = None,
                       smooth_s: float = 0.0) -> Callable:
    """Build a callable f(z) interpolating node values over redshift.

    Parameters
    ----------
    z_nodes, parameter_values : ndarray
        Spline node positions and their fitted values.
    method : {'cubic', 'linear'}, optional
        Interpolation family (default: module-level ``INTERP_METHOD``).
    smooth_s : float, optional
        If > 0, fit a smoothing B-spline (``splrep`` with ``s=smooth_s``)
        through the nodes instead of interpolating them exactly. Used for the
        post-hoc smoothing of a(z)/b(z)/c(z). Evaluation is clamped to the node
        range to avoid extrapolation blow-ups.
    """
    method = method or INTERP_METHOD
    lo, hi = float(z_nodes[0]), float(z_nodes[-1])

    if smooth_s and smooth_s > 0 and len(z_nodes) >= 4:
        k = min(3, len(z_nodes) - 1)
        # smooth_s is a *fraction* of the node-value sum-of-squares about the
        # mean, so one value works across a(~1), b(~0.01), c(~0.05). splrep's s
        # is the absolute residual-sum bound, hence the rescale. A tiny floor
        # keeps near-constant parameters from forcing s=0 (exact interpolation).
        ss_ref = float(np.sum((parameter_values - parameter_values.mean()) ** 2))
        ss = float(smooth_s) * max(ss_ref, 1e-12)
        tck = splrep(z_nodes, parameter_values, s=ss, k=k)

        def _f(z):
            z = np.clip(np.asarray(z, dtype=float), lo, hi)
            return splev(z, tck)
        return _f

    if method == 'linear':
        # Piecewise-linear interpolation; no overshoot, mild linear
        # extrapolation at the edges.
        return interp1d(z_nodes, parameter_values, kind='linear',
                        fill_value='extrapolate')

    # Legacy cubic with extrapolation.
    return interp1d(z_nodes, parameter_values, kind='cubic',
                    fill_value='extrapolate')


def interpolate_parameters(z_nodes: np.ndarray, parameter_values: np.ndarray,
                          z_eval: np.ndarray,
                          method: Optional[str] = None) -> np.ndarray:
    """
    Interpolate parameters between redshift nodes.

    Uses the module-level ``INTERP_METHOD`` (cubic by default) unless
    ``method`` is given explicitly. See :func:`build_param_interp`.

    Parameters
    ----------
    z_nodes : ndarray
        Redshift nodes for interpolation
    parameter_values : ndarray
        Parameter values at each node
    z_eval : ndarray
        Redshifts at which to evaluate the interpolation
    method : {'cubic', 'linear'}, optional
        Interpolation family (default: module-level ``INTERP_METHOD``).

    Returns
    -------
    values : ndarray
        Interpolated parameter values at z_eval
    """
    return build_param_interp(z_nodes, parameter_values, method=method)(z_eval)


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
                 nodes: Dict[str, np.ndarray], n_colors: int,
                 loss: str = 'l2'):
        self.colors, self.mi_j, self.z_j = galaxy_data
        self.C_obs = C_obs
        self.mi_ref = mi_ref
        self.nodes = nodes
        self.n_colors = n_colors
        # 'l2' = Gaussian NLL (chi^2 + 2 log sigma).
        # 'l1' = Laplace NLL (robust to outliers); c(z) parameterizes the
        #        Gaussian-equivalent sigma (Var=2b^2), so no extra 1.4826 factor
        #        is needed downstream.
        self.loss = loss

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

            if self.loss == 'l1':
                # Laplace -2 log L. With scale b = sigma/sqrt(2) (so the Laplace
                # variance equals sigma^2), -2logL = 2*sqrt(2)*|r|/sigma +
                # 2*log(sigma) + const. Robust to the outlier tails that inflate
                # the Gaussian c(z); c(z) stays a Gaussian-equivalent sigma.
                nll = (2.0 * np.sqrt(2.0) * np.sum(np.abs(residual) / sigma)
                       + 2.0 * np.sum(np.log(sigma)))
            else:
                # Gaussian -2 log L (chi^2 + 2 log sigma).
                nll = np.sum(residual**2 / sigma**2) + 2 * np.sum(np.log(sigma))
            total_loss += nll

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
                 regularization_config: Optional[Dict] = None,
                 loss: str = 'l2'):
        super().__init__(galaxy_data, C_obs, mi_ref, nodes, n_colors, loss=loss)
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
                    regularization_config: Optional[Dict] = None,
                    loss: str = 'l2',
                    cap_last_node: bool = False) -> Dict:
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
    nodes = setup_spline_nodes(z_min, z_max, delta_a, delta_b, delta_c,
                               cap_last_node=cap_last_node)

    if colour_names is None:
        colour_names = [str(i) for i in range(n_colors)]

    # Choose model class
    if regularization_config:
        model = RedSequenceModelRegularized(galaxy_data, C_obs, mi_ref, nodes,
                                           n_colors, regularization_config,
                                           loss=loss)
    else:
        model = RedSequenceModel(galaxy_data, C_obs, mi_ref, nodes, n_colors,
                                 loss=loss)

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
                                   regularization_config: Optional[Dict] = None,
                                   loss: str = 'l2',
                                   cap_last_node: bool = False) -> Dict:
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
    nodes = setup_spline_nodes(z_min, z_max, delta_a, delta_b, delta_c,
                               cap_last_node=cap_last_node)

    if colour_names is None:
        colour_names = [str(i) for i in range(n_colors)]

    results = {'colors': colour_names}

    # Fit each color independently
    for i in range(n_colors):
        data_i = ([colors[i]], mi_j, z_j)

        if regularization_config:
            model = RedSequenceModelRegularized(data_i, [C_obs[i]], mi_ref,
                                               nodes, 1, regularization_config,
                                               loss=loss)
        else:
            model = RedSequenceModel(data_i, [C_obs[i]], mi_ref, nodes, 1,
                                     loss=loss)

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


class CrossCovarianceModel:
    """Multivariate chi^2 with frozen a/b/c splines and free r(z) cross-correlation.

    Given residuals  res_i = c_i - (a_i(z) + b_i(z)·(m - m_ref))  with c_i(z)
    already fit per color, the per-galaxy log-likelihood is

        chi2 = res.T @ C^-1 @ res  +  log|det C|

    where C = C_int(z) + C_err  and  C_int_ii = c_i(z)^2,
    C_int_ij = r_ij(z) · c_i(z) · c_j(z) for adjacent pairs (i, j).
    Non-adjacent pairs (no shared band) are held at r=0.

    A Gaussian prior on each r-node value (mean 0, width ``prior_width``)
    regularises the fit (Rykoff+14 eq. 41).
    """

    def __init__(self, residuals: np.ndarray, c_z: np.ndarray,
                 C_err: np.ndarray, z_j: np.ndarray,
                 r_nodes: np.ndarray, pairs: List[Tuple[int, int]],
                 n_colors: int, prior_width: float = 0.45):
        self.residuals = residuals  # (n_colors, n_galaxies)
        self.c_z = c_z              # (n_colors, n_galaxies)
        self.C_err = C_err          # (n_galaxies, n_colors, n_colors)
        self.z_j = z_j
        self.r_nodes = r_nodes
        self.pairs = pairs
        self.n_colors = n_colors
        self.n_galaxies = residuals.shape[1]
        self.prior_width = prior_width

        self.param_names = []
        for (i, j) in pairs:
            self.param_names += [f'r_{i}_{j}_{k}' for k in range(len(r_nodes))]

    def __call__(self, *params):
        n_g, n_c = self.n_galaxies, self.n_colors
        n_r = len(self.r_nodes)

        # Build C_int per galaxy
        C_int = np.zeros((n_g, n_c, n_c))
        for i in range(n_c):
            C_int[:, i, i] = self.c_z[i] ** 2

        prior_penalty = 0.0
        for pair_idx, (i, j) in enumerate(self.pairs):
            r_vals = np.asarray(params[pair_idx * n_r:(pair_idx + 1) * n_r])
            r_z = interpolate_parameters(self.r_nodes, r_vals, self.z_j)
            cov_ij = r_z * self.c_z[i] * self.c_z[j]
            C_int[:, i, j] = cov_ij
            C_int[:, j, i] = cov_ij
            prior_penalty += np.sum((r_vals / self.prior_width) ** 2)

        C = C_int + self.C_err
        res = self.residuals.T  # (n_g, n_c)

        try:
            Cinv_res = np.linalg.solve(C, res[..., None])[..., 0]
        except np.linalg.LinAlgError:
            return 1e30
        sign, logdet = np.linalg.slogdet(C)
        if np.any(sign <= 0) or not np.all(np.isfinite(logdet)):
            return 1e30

        chi2 = float(np.sum(res * Cinv_res) + np.sum(logdet))
        return chi2 + prior_penalty


def fit_cross_covariance(galaxy_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         C_err_full: np.ndarray, mi_ref: np.ndarray,
                         fit_results: Dict,
                         colour_names: List[str],
                         color_definitions: List[Tuple[str, str]],
                         delta_r: float,
                         prior_width: float = 0.45,
                         r_bound: float = 0.95,
                         verbose: bool = True) -> Dict:
    """Stage B: with a/b/c frozen, fit r(z) splines for adjacent color pairs.

    Parameters
    ----------
    galaxy_data : tuple
        (color_array, mi_j, z_j) as built by setup_galaxy_data.
    C_err_full : ndarray, shape (n_galaxies, n_colors, n_colors)
        Full photometric covariance with off-diagonals (eqs. 39-40 of Rykoff+14).
    mi_ref : ndarray
        Reference magnitudes at galaxy redshifts.
    fit_results : dict
        Stage-A results containing a/b/c splines per color.
    colour_names : list of str
    color_definitions : list of (band1, band2) tuples
    delta_r : float
        Node spacing for r(z) splines.
    prior_width : float
        Gaussian prior width on each r-node value (default 0.45 per Rykoff+14).
    r_bound : float
        Hard bound on |r| to keep C positive-definite (default 0.95).

    Returns
    -------
    dict
        ``fit_results`` augmented with an ``'r_cross'`` block.
    """
    colors, mi_j, z_j = galaxy_data
    n_colors = len(colors)
    pairs = find_correlated_pairs(color_definitions)

    if not pairs:
        if verbose:
            print("No correlated color pairs (no shared bands); skipping cross-cov fit.")
        return fit_results

    # Node range: span the data via the existing a-spline nodes (already
    # anchored to the realised z-range).
    a_nodes_any = fit_results[colour_names[0]]['a']['nodes']
    z_min, z_max = float(a_nodes_any[0]), float(a_nodes_any[-1])
    r_nodes = setup_r_nodes(z_min, z_max, delta_r)

    # Evaluate stage-A splines at galaxy redshifts
    residuals = np.zeros((n_colors, len(z_j)))
    c_z = np.zeros((n_colors, len(z_j)))
    for i, cn in enumerate(colour_names):
        a_spl = build_param_interp(fit_results[cn]['a']['nodes'], fit_results[cn]['a']['values'])
        b_spl = build_param_interp(fit_results[cn]['b']['nodes'], fit_results[cn]['b']['values'])
        c_spl = build_param_interp(fit_results[cn]['c']['nodes'], fit_results[cn]['c']['values'])
        a_z = a_spl(z_j)
        b_z = b_spl(z_j)
        c_z[i] = c_spl(z_j)
        residuals[i] = colors[i] - (a_z + b_z * (mi_j - mi_ref))

    model = CrossCovarianceModel(residuals, c_z, C_err_full, z_j,
                                 r_nodes, pairs, n_colors, prior_width)

    init = [0.0] * len(model.param_names)
    limits = [(-r_bound, r_bound)] * len(model.param_names)

    m = Minuit(model, *init)
    m.errordef = 1.0
    m.limits = limits
    m.migrad()

    p = dict(zip(model.param_names, m.values))
    e = dict(zip(model.param_names, m.errors))

    r_block = {
        'nodes': r_nodes,
        'pairs': [(colour_names[i], colour_names[j]) for (i, j) in pairs],
        'pair_indices': pairs,
        'prior_width': prior_width,
        'r_bound': r_bound,
    }
    n_r = len(r_nodes)
    for (i, j) in pairs:
        key = f'{colour_names[i]}__{colour_names[j]}'
        vals = np.array([p[f'r_{i}_{j}_{k}'] for k in range(n_r)])
        errs = np.array([e[f'r_{i}_{j}_{k}'] for k in range(n_r)])
        r_block[key] = {'values': vals, 'errors': errs}

    results = dict(fit_results)
    results['r_cross'] = r_block

    if verbose:
        print(f"Cross-covariance fit: {len(pairs)} pair(s), {n_r} node(s) each, "
              f"prior_width={prior_width}")
        for (i, j) in pairs:
            key = f'{colour_names[i]}__{colour_names[j]}'
            print(f"  r({colour_names[i]}, {colour_names[j]}): "
                  f"{r_block[key]['values'].round(3)}")

    return results


# Alias to allow calling from RedSequenceFitter.fit() without shadowing the
# same-named keyword argument.
fit_cross_covariance_fn = fit_cross_covariance


class RedSequenceModelMultivariate:
    """Joint multivariate chi^2 over all colors, with r(z) splines FROZEN.

    Free parameters: a/b/c per color at the spline nodes (same as the existing
    joint fit). Loss: per-galaxy `res.T @ C^-1 @ res + log|det C|` with
    `C = C_int(z) + C_err`, where C_int_ii = c_i(z)^2 and the off-diagonals
    come from the frozen r-splines × c_i(z) × c_j(z). Used by the Stage A'
    refit in the iteration loop.
    """

    def __init__(self, galaxy_data, C_err_full, mi_ref,
                 nodes, n_colors, r_splines_frozen, pair_indices):
        self.colors, self.mi_j, self.z_j = galaxy_data
        self.C_err = C_err_full
        self.mi_ref = mi_ref
        self.nodes = nodes
        self.n_colors = n_colors
        self.r_frozen = r_splines_frozen  # dict (i, j) -> callable r(z)
        self.pairs = pair_indices

        self.param_names = []
        for i in range(n_colors):
            self.param_names += [f'a_{i}_{j}' for j in range(len(nodes['a']))]
            self.param_names += [f'b_{i}_{j}' for j in range(len(nodes['b']))]
            self.param_names += [f'c_{i}_{j}' for j in range(len(nodes['c']))]

    def __call__(self, *params):
        p = dict(zip(self.param_names, params))
        n_g = len(self.z_j)
        n_c = self.n_colors

        residuals = np.zeros((n_g, n_c))
        c_z_arr = np.zeros((n_c, n_g))
        for i in range(n_c):
            a = np.array([p[f'a_{i}_{j}'] for j in range(len(self.nodes['a']))])
            b = np.array([p[f'b_{i}_{j}'] for j in range(len(self.nodes['b']))])
            c = np.array([p[f'c_{i}_{j}'] for j in range(len(self.nodes['c']))])
            a_z = interpolate_parameters(self.nodes['a'], a, self.z_j)
            b_z = interpolate_parameters(self.nodes['b'], b, self.z_j)
            c_z_arr[i] = interpolate_parameters(self.nodes['c'], c, self.z_j)
            residuals[:, i] = self.colors[i] - (a_z + b_z * (self.mi_j - self.mi_ref))

        C_int = np.zeros((n_g, n_c, n_c))
        for i in range(n_c):
            C_int[:, i, i] = c_z_arr[i] ** 2
        for (i, j) in self.pairs:
            r_z = self.r_frozen[(i, j)](self.z_j)
            cov_ij = r_z * c_z_arr[i] * c_z_arr[j]
            C_int[:, i, j] = cov_ij
            C_int[:, j, i] = cov_ij

        C = C_int + self.C_err
        try:
            Cinv_res = np.linalg.solve(C, residuals[..., None])[..., 0]
        except np.linalg.LinAlgError:
            return 1e30
        sign, logdet = np.linalg.slogdet(C)
        if np.any(sign <= 0) or not np.all(np.isfinite(logdet)):
            return 1e30
        return float(np.sum(residuals * Cinv_res) + np.sum(logdet))


def fit_red_sequence_with_r_fixed(
    galaxy_data, C_err_full, mi_ref,
    fit_results, colour_names, color_definitions,
    z_min, z_max, delta_a, delta_b, delta_c,
    init_results=None, verbose=True,
):
    """Stage A': refit a/b/c jointly with the r(z) splines frozen."""
    colors, mi_j, z_j = galaxy_data
    n_colors = len(colors)
    pairs = find_correlated_pairs(color_definitions)

    # Build frozen r-splines from results['r_cross']
    rc = fit_results['r_cross']
    r_frozen = {}
    for (i, j) in pairs:
        key = f'{colour_names[i]}__{colour_names[j]}'
        r_frozen[(i, j)] = CubicSpline(rc['nodes'], rc[key]['values'])

    nodes = setup_spline_nodes(z_min, z_max, delta_a, delta_b, delta_c)
    model = RedSequenceModelMultivariate(
        galaxy_data, C_err_full, mi_ref, nodes, n_colors, r_frozen, pairs,
    )

    # Warm-start from previous fit_results
    init = []
    limits = []
    src = init_results if init_results is not None else fit_results
    for i, cn in enumerate(colour_names):
        init += list(src[cn]['a']['values'])
        limits += [(0.2, 3.5)] * len(nodes['a'])
        init += list(src[cn]['b']['values'])
        limits += [(-0.5, 0.5)] * len(nodes['b'])
        init += list(src[cn]['c']['values'])
        limits += [(0.001, 0.4)] * len(nodes['c'])

    m = Minuit(model, *init)
    m.errordef = 1.0
    m.limits = limits
    m.migrad()

    params = dict(zip(model.param_names, m.values))
    errors = dict(zip(model.param_names, m.errors))
    new_results = extract_named_params(model, params, colour_names, errors)
    # Preserve the r_cross block so the next Stage B can iterate
    new_results['r_cross'] = fit_results['r_cross']
    if verbose:
        for cn in colour_names:
            c_new = new_results[cn]['c']['values']
            c_old = fit_results[cn]['c']['values']
            dmax = float(np.max(np.abs(c_new - c_old)))
            print(f"  Stage A' [{cn}]: max |Δc(z)| = {dmax:.4f}, "
                  f"new c̄ = {c_new.mean():.3f} (was {c_old.mean():.3f})")
    return new_results


def iterate_cross_covariance(
    galaxy_data, C_err_full, mi_ref,
    initial_results, colour_names, color_definitions,
    z_min, z_max, delta_a, delta_b, delta_c,
    delta_r, prior_width=0.45, r_bound=0.95,
    max_iterations=3, tol=1e-3, verbose=True,
):
    """Alternate Stage B (fit r | a,b,c) and Stage A' (fit a,b,c | r).

    ``initial_results`` is the Stage-A output (a/b/c with r=0).
    The first Stage B uses those frozen a/b/c. Subsequent iterations refit
    a/b/c jointly (multivariate likelihood) with r frozen, then refit r.
    Converges when max |Δc(z)| across all colors falls below ``tol``.
    """
    # Iteration 1: Stage B from the supplied Stage-A results.
    if verbose:
        print(f"[iter 1/{max_iterations}] Stage B (r | a,b,c=Stage A)")
    results = fit_cross_covariance_fn(
        galaxy_data, C_err_full, mi_ref, initial_results,
        colour_names, color_definitions, delta_r,
        prior_width=prior_width, r_bound=r_bound, verbose=verbose,
    )
    if 'r_cross' not in results:
        return results  # no correlated pairs

    prev_results = results
    for it in range(2, max_iterations + 1):
        if verbose:
            print(f"[iter {it}/{max_iterations}] Stage A' (a,b,c | r fixed)")
        refit = fit_red_sequence_with_r_fixed(
            galaxy_data, C_err_full, mi_ref,
            prev_results, colour_names, color_definitions,
            z_min, z_max, delta_a, delta_b, delta_c,
            init_results=prev_results, verbose=verbose,
        )
        # Convergence: largest c-shift across colors
        dmax = max(
            float(np.max(np.abs(refit[cn]['c']['values'] - prev_results[cn]['c']['values'])))
            for cn in colour_names
        )
        if verbose:
            print(f"[iter {it}/{max_iterations}] max |Δc| over colors = {dmax:.4f}")
        if dmax < tol:
            if verbose:
                print(f"[converged] max |Δc| < tol ({tol})")
            return refit

        if verbose:
            print(f"[iter {it}/{max_iterations}] Stage B' (r | a,b,c=Stage A')")
        results = fit_cross_covariance_fn(
            galaxy_data, C_err_full, mi_ref, refit,
            colour_names, color_definitions, delta_r,
            prior_width=prior_width, r_bound=r_bound, verbose=verbose,
        )
        prev_results = results

    if verbose:
        print(f"[done] max_iterations ({max_iterations}) reached")
    return prev_results


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

    # Interpolation metadata (so the downstream photo-z reproduces the same
    # a/b/c(z) functions the fit used).
    flat_dict['interp_method'] = np.array(results.get('interp_method', 'cubic'))
    flat_dict['smooth_abc_s'] = np.array(float(results.get('smooth_abc_s', 0.0)))

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

    # Optional cross-covariance r(z) block
    if 'r_cross' in results:
        rc = results['r_cross']
        flat_dict['r_cross_nodes'] = rc['nodes']
        flat_dict['r_cross_pair_names'] = np.array(
            [f'{a}__{b}' for (a, b) in rc['pairs']]
        )
        flat_dict['r_cross_prior_width'] = np.array(rc['prior_width'])
        for (ca, cb) in rc['pairs']:
            key = f'{ca}__{cb}'
            flat_dict[f'r_cross_{key}_values'] = rc[key]['values']
            flat_dict[f'r_cross_{key}_errors'] = rc[key]['errors']

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

    # Interpolation metadata (default to legacy cubic for old .npz files).
    if 'interp_method' in data.files:
        results['interp_method'] = str(data['interp_method'])
    if 'smooth_abc_s' in data.files:
        results['smooth_abc_s'] = float(data['smooth_abc_s'])

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

    # Optional cross-covariance r(z) block
    if 'r_cross_nodes' in data.files:
        pair_names = [s for s in data['r_cross_pair_names']]
        pairs = [tuple(p.split('__')) for p in pair_names]
        r_block = {
            'nodes': data['r_cross_nodes'],
            'pairs': pairs,
            'prior_width': float(data['r_cross_prior_width']),
        }
        for key in pair_names:
            r_block[key] = {
                'values': data[f'r_cross_{key}_values'],
                'errors': data[f'r_cross_{key}_errors'],
            }
        results['r_cross'] = r_block

    return results, m_ref_data


def make_r_cross_splines(results: Dict, colour_names: List[str]) -> Dict[Tuple[int, int], CubicSpline]:
    """Build per-pair CubicSpline objects for the r(z) cross-covariance.

    Returns empty dict if ``results`` has no ``r_cross`` block. Keys are
    (i, j) colour indices into ``colour_names``.
    """
    if 'r_cross' not in results:
        return {}
    rc = results['r_cross']
    name_to_idx = {n: i for i, n in enumerate(colour_names)}
    out = {}
    for (ca, cb) in rc['pairs']:
        if ca not in name_to_idx or cb not in name_to_idx:
            continue
        key = f'{ca}__{cb}'
        i, j = name_to_idx[ca], name_to_idx[cb]
        out[(i, j)] = CubicSpline(rc['nodes'], rc[key]['values'])
    return out


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

    # Interpolation family and optional post-hoc smoothing are persisted in the
    # results dict so the downstream photo-z step uses the exact same a/b/c(z)
    # functions that the fit saw.
    method = results.get('interp_method', INTERP_METHOD)
    smooth_s = float(results.get('smooth_abc_s', 0.0) or 0.0)

    spline_dict = {}

    for col in colour_names:
        spline_dict[col] = {}
        for kind in ['a', 'b', 'c']:
            z_nodes = np.asarray(results[col][kind]['nodes'])
            values = np.asarray(results[col][kind]['values'])
            spline_dict[col][kind] = build_param_interp(
                z_nodes, values, method=method, smooth_s=smooth_s)

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
                 regularization_config: Optional[Dict] = None,
                 interp_method: Optional[str] = None,
                 smooth_abc_s: float = 0.0,
                 loss: str = 'l2',
                 cap_last_node: bool = False):
        self.z_min = z_min
        self.z_max = z_max
        self.delta_a = delta_a
        self.delta_b = delta_b
        self.delta_c = delta_c
        self.regularization_config = regularization_config
        # Interpolation family for a/b/c(z) (default: module-level INTERP_METHOD,
        # i.e. cubic) and optional post-hoc smoothing strength.
        self.interp_method = interp_method or INTERP_METHOD
        self.smooth_abc_s = float(smooth_abc_s or 0.0)
        self.loss = loss
        self.cap_last_node = bool(cap_last_node)

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
        # Build color arrays with full covariance (off-diagonals come from
        # bands shared between colors; needed for the cross-covariance fit).
        color_array, color_covariance = create_color_arrays_and_covariance(
            df, color_definitions, magnitude_col
        )

        # Diagonal observational variance per color (stage-A chi^2)
        C_obs = np.array([color_covariance[:, i, i] for i in range(len(color_definitions))])

        # Stash full C_err for the optional cross-covariance fit.
        self._C_err_full = color_covariance

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
           colour_names: List[str], method: str = 'single',
           fit_cross_covariance: bool = False,
           color_definitions: Optional[List[Tuple[str, str]]] = None,
           delta_r: Optional[float] = None,
           r_prior_width: float = 0.45,
           cross_cov_iterations: int = 1,
           cross_cov_tol: float = 1e-3,
           verbose: bool = True) -> Dict:
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
        # The fit objective interpolates a/b/c(z) via the module-level
        # INTERP_METHOD. Each fit runs in its own process, so setting it here is
        # safe. Both supported families (cubic, linear) are smooth in the node
        # values, so they drive MIGRAD directly.
        global INTERP_METHOD
        INTERP_METHOD = self.interp_method

        if method == 'single':
            self.results = fit_red_sequence_single_colour(
                galaxy_data, C_obs, mi_ref,
                self.z_min, self.z_max,
                self.delta_a, self.delta_b, self.delta_c,
                colour_names, self.regularization_config,
                loss=self.loss, cap_last_node=self.cap_last_node,
            )
        elif method == 'joint':
            self.results = fit_red_sequence(
                galaxy_data, C_obs, mi_ref,
                self.z_min, self.z_max,
                self.delta_a, self.delta_b, self.delta_c,
                colour_names, self.regularization_config,
                loss=self.loss, cap_last_node=self.cap_last_node,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Persist interpolation choice + post-hoc smoothing so save/load and the
        # downstream photo-z reproduce these exact a/b/c(z) functions.
        self.results['interp_method'] = self.interp_method
        self.results['smooth_abc_s'] = self.smooth_abc_s

        # Build spline functions
        self.splines = make_spline_functions(self.results, colour_names)

        # Optional Stage B: cross-covariance r(z) fit with a/b/c frozen.
        if fit_cross_covariance:
            if color_definitions is None:
                raise ValueError(
                    "fit_cross_covariance=True requires color_definitions"
                )
            C_err_full = getattr(self, '_C_err_full', None)
            if C_err_full is None:
                raise ValueError(
                    "Full C_err not available; call setup_galaxy_data() first."
                )
            if cross_cov_iterations <= 1:
                self.results = fit_cross_covariance_fn(
                    galaxy_data, C_err_full, mi_ref, self.results,
                    colour_names, color_definitions,
                    delta_r if delta_r is not None else self.delta_c,
                    prior_width=r_prior_width, verbose=verbose,
                )
            else:
                self.results = iterate_cross_covariance(
                    galaxy_data, C_err_full, mi_ref, self.results,
                    colour_names, color_definitions,
                    self.z_min, self.z_max,
                    self.delta_a, self.delta_b, self.delta_c,
                    delta_r if delta_r is not None else self.delta_c,
                    prior_width=r_prior_width,
                    max_iterations=cross_cov_iterations,
                    tol=cross_cov_tol,
                    verbose=verbose,
                )
            # Rebuild spline functions after possible A' refit
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
