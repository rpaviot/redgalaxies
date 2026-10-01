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
from scipy.special import erf, ndtr
from iminuit import Minuit

from red_galaxy_pipeline.utils import create_color_arrays_and_covariance, parse_colors


LOG_2PI = float(np.log(2.0 * np.pi))


def c_bounds() -> Tuple[float, float]:
    """Optimiser bounds on the intrinsic-scatter nodes c(z).

    The upper bound is a numerical guard, not a physical statement, so it is
    settable via ``RIDGELINE_C_MAX`` (default 0.4, the historical value).
    Raise it whenever a fit rails: a node parked exactly on the bound is a
    CENSORED result, and ``success=True`` will not tell you so -- L-BFGS-B
    reports convergence at a bound just as happily as at an interior minimum.
    That failure mode is the mirror image of the c -> 0.001 collapse the
    truncation correction was added to cure.
    """
    import os as _os
    return (0.001, float(_os.environ.get("RIDGELINE_C_MAX", "0.4")))


def p_bounds() -> Tuple[float, float]:
    """Optimiser bounds on the red-membership nodes p(z) of the Eq.31 mixture.

    The lower bound is a GUARD, not a physical statement, settable via
    ``RIDGELINE_P_MIN`` (default 0.5). It exists because the mixture has a
    runaway: the background component ``b(c, m)`` is broad, so an unbounded
    ``p`` can explain the whole sample as contamination and let ``c`` drift
    wherever it likes. We are fitting an already red-SELECTED sample, so the
    membership fraction should be high; if a node rails at ``p_min`` that is a
    statement about the selection, not a converged purity, and it is reported
    the same way a railed ``c`` node is. See :func:`c_bounds`.
    """
    import os as _os
    return (float(_os.environ.get("RIDGELINE_P_MIN", "0.5")), 1.0)


def setup_spline_nodes(z_min: float, z_max: float,
                       delta_a: float, delta_b: float, delta_c: float,
                       cap_last_node: bool = False,
                       delta_p: Optional[float] = None) -> Dict[str, np.ndarray]:
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

    delta_p : float, optional
        Node spacing for the Eq.31 membership fraction p(z). When None (the
        default) no ``'p'`` entry is produced and the model is the pure
        red-sequence likelihood, unchanged.

    Returns
    -------
    nodes : dict
        Dictionary with keys 'a', 'b', 'c' (and 'p' when ``delta_p`` is given)
        containing node arrays.
    """
    def _grid(delta: float) -> np.ndarray:
        # ceil with a tiny tolerance so an exact multiple doesn't add a node.
        n_steps = int(np.ceil((z_max - z_min) / delta - 1e-9))
        grid = z_min + np.arange(n_steps + 1) * delta
        if cap_last_node and grid[-1] > z_max:
            grid[-1] = z_max
        return grid

    out = {'a': _grid(delta_a), 'b': _grid(delta_b), 'c': _grid(delta_c)}
    if delta_p is not None:
        # Membership p(z) for the Eq.31 mixture. Defaults to the c(z) spacing at
        # the call sites: the contaminated fraction, like the scatter, is noisy
        # to estimate, so it gets the same coarse nodes.
        out['p'] = _grid(delta_p)
    return out


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

    # Legacy cubic with extrapolation. A short node range (u-g fitted only on
    # 0.10-0.40: c nodes 0.10/0.25/0.40) has too few nodes for a cubic; drop to
    # the highest order the nodes support (quadratic for 3, linear for 2).
    kind = {2: 'linear', 3: 'quadratic'}.get(len(z_nodes), 'cubic')
    return interp1d(z_nodes, parameter_values, kind=kind,
                    fill_value='extrapolate')


def load_truncation(filepath: str) -> Dict[str, Tuple[Callable, Callable]]:
    """Load a step-1 selection-edge model written by ``measure_step1_edge.py``.

    The step-1 GMM selection keeps a galaxy only if its colour residual satisfies
    ``|delta| < t``, so the retained sample is a *truncated* draw from the red
    sequence.  Fitting an untruncated Gaussian to it biases the intrinsic scatter
    c(z) low -- badly, once the photometric term dominates.  redMaPPer handles
    this by normalising the likelihood by the surviving probability mass
    (Rykoff et al. 2014, arXiv:1303.3562, Eq. 29); this loads the edge model that
    supplies ``t`` per galaxy.

    The edge is parameterised as ``t^2 = kappa(z) * C_phot + const(z)``, which is
    the shape the step-1 cut imprints (it cuts on
    ``delta^2 / (S_mod^2 + C_phot)``).  ``kappa`` and ``const`` are *not*
    separately meaningful -- only the line they define is -- so always fit and
    use them as a pair.

    Returns
    -------
    dict
        ``{colour_name: (kappa_func, const_func)}``.
    """
    d = np.load(filepath, allow_pickle=True)
    import os as _os
    centre = _os.environ.get("RIDGELINE_TRUNC_CENTRE", "model").lower()
    if 'format' in d.files and str(d['format']) == 'twostage':
        # Exact two-stage window from a one-seed disjoint step 1
        # (src/build_twostage_truncation.py). Per colour and per z bin:
        #   mu2, V2   stage-2 XD mean / marginal variance of this colour
        #   chi2_4    stage-2 Mahalanobis threshold (chi^2, 4 dof)
        #   is_ref    1 where this colour was the stage-1 reference colour
        #   mu1, V1, chi2_1   stage-1 window (only meaningful where is_ref)
        # Consumed by RedSequenceModel with RIDGELINE_TRUNC_CENTRE=twostage.
        if centre != 'twostage':
            raise ValueError(f"{filepath} is a two-stage window model; set "
                             f"RIDGELINE_TRUNC_CENTRE=twostage (got {centre!r})")
        edges = np.asarray(d['z_edges'], dtype=float)
        out = {'__format__': 'twostage', '__z_edges__': edges}
        for col in [str(c) for c in d['colors']]:
            out[col] = {k: np.asarray(d[f'{col}_{k}'], dtype=float)
                        for k in ('mu2', 'V2', 'chi2_4', 'is_ref', 'mu1', 'V1', 'chi2_1')}
            # ndim = colours in the bin's stage-2 cut (3 where u-g was dropped);
            # windows written before 2026-09-24 are all 4-colour
            out[col]['ndim'] = (np.asarray(d[f'{col}_ndim'], dtype=float)
                                if f'{col}_ndim' in d.files
                                else np.full(len(edges) - 1, 4.0))
        if 'V2full' in d.files:
            # full stage-2 matrices (colour order = d['colors']): needed by the
            # exact box-leakage term (build_xcov_leak)
            out['__colors__'] = [str(c) for c in d['colors']]
            out['__V2full__'] = np.asarray(d['V2full'], float)
            out['__mu2full__'] = np.asarray(d['mu2full'], float)
        return out
    method = str(d['interp_method']) if 'interp_method' in d.files else 'linear'
    nodes = np.asarray(d['nodes_trunc'], dtype=float)
    # Consistency check: the walls must have been measured in the coordinate
    # the likelihood centres its window on (see RIDGELINE_TRUNC_CENTRE).
    about = str(d['about']) if 'about' in d.files else 'ridgeline'
    expected = {'model': 'ridgeline', 'marginal': 'marginal'}.get(centre)
    if expected is not None and about != expected:
        raise ValueError(
            f"truncation model {filepath} was measured about the {about!r} but "
            f"RIDGELINE_TRUNC_CENTRE={centre!r} expects walls about the "
            f"{expected!r}; re-measure with measure_step1_edge.py --about {expected}")
    out = {}
    for col in [str(c) for c in d['colors']]:
        out[col] = (
            build_param_interp(nodes, np.asarray(d[f'{col}_kappa_values'], float),
                               method=method),
            build_param_interp(nodes, np.asarray(d[f'{col}_const_values'], float),
                               method=method),
        )
    return out


def log_window_mass(t: np.ndarray, d: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """log of the mass a Gaussian N(d, sigma^2) puts inside the window [-t, t].

        P = Phi((t - d)/sigma) - Phi((-t - d)/sigma)

    evaluated in log-space (scipy.special.log_ndtr), never as a difference of
    two saturated erf/ndtr values.  That matters: for a galaxy whose model
    mean sits many sigma outside its window, the naive
    ``0.5*(erf(x_hi) + erf(x_lo))`` cancels to exactly 0, gets clipped, and
    contributes 2*log(1e-300) ~ -1381 to the NLL instead of the proper
    Gaussian-tail penalty ~ -x^2 -- a fake reward of ~-1300 per galaxy that
    opens a spurious basin at large |b| (seen on the slope mock: NLL -1.45M
    vs -0.36M at the truth).  Symmetric in d, so the d < 0 half is mirrored.
    """
    from scipy.special import log_ndtr, ndtr
    t = np.asarray(t, float); d = np.asarray(d, float); sigma = np.asarray(sigma, float)
    dd = np.abs(d)                      # window is symmetric: mass(d) = mass(-d)
    x_hi = (t - dd) / sigma             # nearer edge
    x_lo = (-t - dd) / sigma            # farther edge, x_lo < x_hi always
    out = np.empty_like(x_hi)
    both_neg = x_hi <= 0.0
    # both edges in the lower tail: log[Phi(x_hi) - Phi(x_lo)]
    #   = log Phi(x_hi) + log(1 - exp(log Phi(x_lo) - log Phi(x_hi)))
    lh = log_ndtr(x_hi[both_neg]); ll = log_ndtr(x_lo[both_neg])
    out[both_neg] = lh + np.log1p(-np.exp(np.minimum(ll - lh, -1e-300)))
    # window straddles the mean: plain difference is accurate (P is O(1))
    m = ~both_neg
    out[m] = np.log(np.clip(ndtr(x_hi[m]) - ndtr(x_lo[m]), 1e-300, None))
    return out


def truncation_limits(truncation: Optional[Dict[str, Tuple[Callable, Callable]]],
                      colour_names: List[str], z_j: np.ndarray,
                      C_obs: np.ndarray) -> Optional[List[np.ndarray]]:
    """Evaluate the per-galaxy truncation limit ``t`` for each colour.

    Returns ``None`` when no truncation model is supplied, which restores the
    plain (untruncated) likelihood exactly.
    """
    if truncation is None:
        return None
    limits = []
    if truncation.get('__format__') == 'twostage':
        edges = truncation['__z_edges__']
        idx = np.clip(np.searchsorted(edges, np.asarray(z_j, float), side='right') - 1,
                      0, len(edges) - 2)
        for i, name in enumerate(colour_names):
            if name not in truncation:
                raise KeyError(f"two-stage window model has no entry for colour {name!r}")
            w = truncation[name]
            if not np.all(np.isfinite(w['mu2'][idx])):
                zz = np.asarray(z_j, float)[~np.isfinite(w['mu2'][idx])]
                raise ValueError(
                    f"two-stage window has no {name!r} entry for {zz.size} galaxies "
                    f"(z {zz.min():.3f}-{zz.max():.3f}): the colour was not in those "
                    f"bins' stage-2 cut -- restrict --z-max for this colour")
            C = np.asarray(C_obs[i], dtype=float)
            t2 = np.sqrt(np.clip(w['chi2_4'][idx] * (w['V2'][idx] + C), 1e-12, None))
            t1 = np.where(w['is_ref'][idx] > 0.5,
                          np.sqrt(np.clip(w['chi2_1'][idx] * (w['V1'][idx] + C), 1e-12, None)),
                          np.inf)
            # Stage-1 window is centred on mu1, stage-2 on mu2; they differ by
            # < 0.01 mag in practice, so one centre (mu2, the last cut applied)
            # is used and the stage-1 box is applied about it.
            limits.append({'mu': w['mu2'][idx], 't1': t1, 't2': t2,
                           'chi2_4': w['chi2_4'][idx], 'ndim': w['ndim'][idx]})
        return limits
    for i, name in enumerate(colour_names):
        if name not in truncation:
            raise KeyError(
                f"truncation model has no entry for colour {name!r}; "
                f"it covers {sorted(truncation)}")
        kappa_f, const_f = truncation[name]
        t_sq = kappa_f(z_j) * np.asarray(C_obs[i], dtype=float) + const_f(z_j)
        limits.append(np.sqrt(np.clip(t_sq, 1e-12, None)))
    return limits


def log_twostage_mass(win: Dict[str, np.ndarray], model: np.ndarray, sigma: np.ndarray,
                      gl_nodes: np.ndarray, gl_weights: np.ndarray) -> np.ndarray:
    """log surviving mass of N(model, sigma^2) under the two-stage step-1 cut.

    Selection function in this colour, about the window centre mu:
        S(delta) = 1(|delta| < t1) * F_chi2_{n-1}( chi2_4 * (1 - delta^2 / t2^2) )
    (n = colours in the bin's stage-2 cut: 4, or 3 where u-g was dropped;
    chi2_4 is that bin's stage-2 threshold, whatever its dof)
    -- the stage-1 box (reference colour only, t1 = inf otherwise) times the
    stage-2 chi^2_4 ellipsoid marginalised over the other three colours (the
    other axes at their means; verified against the data to ~0.02).
    P = int S(delta) N(delta; d, sigma^2) d delta with d = model - mu, done by
    Gauss-Legendre quadrature on [-ymax, ymax], ymax = min(t1, t2) (S = 0
    beyond t2).  F is smooth so 16 nodes are ample.

    The abscissae y, the taper and the weights depend only on the FIXED
    window (t1, t2, chi2_4), never on the fit parameters, so they are built
    once per colour and cached in ``win`` (``_gl_y``, ``_gl_wtaper``). That
    matters: chi2.cdf (a gammainc call on a 16 x N array) is ~90% of the
    objective's cost when done per evaluation -- 1.07 s vs 0.13 s per call
    for N = 270k, i.e. ~5 h vs ~40 min per colour under L-BFGS-B's numerical
    gradient (~48 evaluations per iteration), which is what timed out the
    first two-stage runs (jobs 870012/870023) at 12 h.
    """
    if '_gl_y' not in win:
        from scipy.stats import chi2 as _chi2
        t1, t2, k4 = win['t1'], win['t2'], win['chi2_4']
        ymax = np.minimum(t1, t2)                                   # (N,)
        y = ymax[None, :] * gl_nodes[:, None]                        # (K, N)
        dof = win.get('ndim', np.full_like(k4, 4.0)) - 1.0
        taper = _chi2.cdf(k4[None, :] * (1.0 - (y / t2[None, :]) ** 2), df=dof[None, :])
        win['_gl_y'] = y
        win['_gl_wtaper'] = ymax[None, :] * gl_weights[:, None] * taper
    y, wtaper = win['_gl_y'], win['_gl_wtaper']
    d = np.asarray(model, float) - win['mu']
    gauss = np.exp(-0.5 * ((y - d[None, :]) / sigma[None, :]) ** 2) / (np.sqrt(2 * np.pi) * sigma[None, :])
    P = np.sum(wtaper * gauss, axis=0)
    return np.log(np.clip(P, 1e-300, None))


def log_twostage_leak_mass(win: Dict[str, np.ndarray], model: np.ndarray, sigma: np.ndarray,
                           c_z: np.ndarray, residual: np.ndarray,
                           gl_nodes: np.ndarray, gl_weights: np.ndarray):
    """Two-stage survival mass WITH the stage-1 box leakage on a non-reference colour.

    With an intrinsic correlation r(ref, j) the stage-1 box |x_ref - mu1| < t1
    also truncates colour j (the rho=0.6 mock: i-z slope over-corrected +22%).
    Given x_j, the reference colour is Gaussian with
        mean = model_ref + k (x_j - model_j),   k = S_rj / sigma_j^2
        var  = S_rr - S_rj^2 / sigma_j^2
    S_rj = r c_ref c_j + C_err_rj, S_rr = c_ref^2 + C_err_rr, so the box keeps
    the fraction B(x_j) = Phi((mu1 + t1 - mean)/s) - Phi((mu1 - t1 - mean)/s).
    B depends on the fit parameters (c_j, model_j), so it enters the numerator
    (returned log B at the observed colour) as well as the survival mass
    P = int S2(delta) B N(delta; d, sigma^2) (returned log P). Galaxies whose
    reference colour IS j (``win['leak']['idx']`` excludes them) get the plain
    two-stage mass; their box is already in t1.
    Returns (log_P, log_B_obs), both (N,).
    """
    logP = log_twostage_mass(win, model, sigma, gl_nodes, gl_weights)
    lk = win['leak']
    if np.any(win.get('ndim', 4.0) != 4.0):
        raise NotImplementedError("box-leakage term assumes 4-colour stage-2 bins")
    L = lk['idx']
    logB = np.zeros_like(logP)
    if len(L) == 0:
        return logP, logB
    y, wtaper = win['_gl_y'][:, L], win['_gl_wtaper'][:, L]
    sig = sigma[L]
    d = model[L] - win['mu'][L]
    S_rj = lk['r'] * lk['c_ref'] * c_z[L] + lk['E_rj']
    k = S_rj / sig ** 2
    s = np.sqrt(np.clip(lk['S_rr'] - S_rj ** 2 / sig ** 2, 1e-12, None))
    hi, lo = lk['mu1'] + lk['t1'], lk['mu1'] - lk['t1']

    def box(e):
        mean = lk['model_ref'] + k * e
        return ndtr((hi - mean) / s) - ndtr((lo - mean) / s)

    gauss = np.exp(-0.5 * ((y - d[None, :]) / sig[None, :]) ** 2) / (np.sqrt(2 * np.pi) * sig[None, :])
    P = np.sum(wtaper * gauss * box(y - d[None, :]), axis=0)
    logP[L] = np.log(np.clip(P, 1e-300, None))
    logB[L] = np.log(np.clip(box(residual[L]), 1e-300, None))
    return logP, logB


def log_twostage_leak_exact(win: Dict[str, np.ndarray], model: np.ndarray, sigma: np.ndarray,
                            c_z: np.ndarray, gl_nodes: np.ndarray, gl_weights: np.ndarray):
    """EXACT two-stage selection on a non-reference colour j with intrinsic correlation.

    The selection on j is not (box leakage) x F_chi2_3: the F_chi2_3 taper
    assumes the reference colour follows the untruncated stage-2 Gaussian, but
    the box has cut it, so the product counts part of the truncation twice
    (rho=0.6 mock: b over-corrected +8-11% in gr/ri/ug). Integrating the
    reference colour out explicitly instead:
        S(x_j) = int dx_r N(x_r | x_j) 1(|x_r - mu1| < t1) F_chi2_2(chi2_4 - q(x_j, x_r))
        P      = int int N2((x_r, x_j); (model_r, model_j), S_rj) box F_chi2_2 d^2x
    with S_rj = [[c_r^2 + E_rr, r c_r c_j + E_rj], [., sigma_j^2]] and q the
    stage-2 Mahalanobis of the (r, j) block of G = V2 + C_err (the other two
    colours integrate to chi^2_2). Nodes/weights are fixed (cached in
    build_xcov_leak); only the Gaussians depend on the parameters.
    Galaxies whose reference colour IS j keep the plain two-stage mass.
    Returns log P - log S(x_j) per galaxy (to be added as ``log_erf``).
    """
    out = log_twostage_mass(win, model, sigma, gl_nodes, gl_weights)
    lk = win['leak']
    if np.any(win.get('ndim', 4.0) != 4.0):
        raise NotImplementedError("exact box-leakage term assumes 4-colour stage-2 bins (F_chi2_2)")
    L = lk['idx']
    if len(L) == 0:
        return out
    sj2 = sigma[L] ** 2
    Srj = lk['r'] * lk['c_ref'] * c_z[L] + lk['E_rj']
    Srr = lk['S_rr']
    dj = model[L] - lk['mu2_j']
    det = np.clip(Srr * sj2 - Srj ** 2, 1e-30, None)
    vr = lk['U_r'] - lk['d_r'][:, None]
    vj = lk['U_j'] - dj[:, None]
    q = (sj2[:, None] * vr ** 2 - 2 * Srj[:, None] * vr * vj + Srr[:, None] * vj ** 2) / det[:, None]
    P = np.sum(lk['W'] * np.exp(-0.5 * q), axis=1) / (2 * np.pi * np.sqrt(det))
    # conditional of u_r = x_r - mu2_r given the observed x_j
    mc = lk['d_r'] + Srj / sj2 * (lk['x_j'] - model[L])
    vc = np.clip(Srr - Srj ** 2 / sj2, 1e-30, None)
    Snum = np.sum(lk['Wnum'] * np.exp(-0.5 * (lk['Ur_num'] - mc[:, None]) ** 2 / vc[:, None]),
                  axis=1) / np.sqrt(2 * np.pi * vc)
    out[L] = np.log(np.clip(P, 1e-300, None)) - np.log(np.clip(Snum, 1e-300, None))
    return out


def build_xcov_leak(ridgeline_path: str, df, color_definitions, colour_names,
                    magnitude_col: str, z_spec_col: str, trunc_t: List[Dict],
                    truncation: Dict, smooth_mref: bool = True) -> None:
    """Attach the stage-1 box-leakage inputs to each two-stage window in ``trunc_t``.

    ``ridgeline_path`` is a merged 4-colour ridgeline with an r_cross block
    (src/fit_xcov_pairwise.py): it supplies the FROZEN reference-colour model
    a_ref + b_ref (m - m_ref), c_ref(z) and r(ref, j)(z). Photometric terms
    come from the full colour covariance of ``df`` (all four colours, shared
    bands), independent of which colours this run fits.
    """
    from .utils import build_mref_function, create_color_arrays_and_covariance
    all_defs = [('g', 'r'), ('r', 'i'), ('i', 'z'), ('u', 'g')]
    all_names = ['gr', 'ri', 'iz', 'ug']
    res, mref = load_params(ridgeline_path)
    if 'r_cross' not in res:
        raise ValueError(f"{ridgeline_path} has no r_cross block")
    spl = make_spline_functions(res, all_names)
    rcs = make_r_cross_splines(res, all_names)
    mref_f = build_mref_function(mref[0], mref[1], smooth_mref=smooth_mref)
    z = df[z_spec_col].to_numpy()
    m = df[magnitude_col].to_numpy()
    X, E = create_color_arrays_and_covariance(df, all_defs, magnitude_col)
    # exact (j, ref) integral when the window carries the full stage-2 matrices;
    # RIDGELINE_XCOV_LEAK_MODE=product forces the old approximate form
    import os as _os
    exact = ('__V2full__' in truncation
             and _os.environ.get('RIDGELINE_XCOV_LEAK_MODE', 'exact').lower() == 'exact')
    if exact:
        if truncation['__colors__'] != all_names:
            raise ValueError(f"window colour order {truncation['__colors__']} != {all_names}")
        V2full, mu2full = truncation['__V2full__'], truncation['__mu2full__']
    print(f"  xcov leak mode: {'EXACT (j, ref) integral' if exact else 'product B x F_chi2_3'}")
    rn = np.clip(z, res['r_cross']['nodes'][0], res['r_cross']['nodes'][-1])
    edges = truncation['__z_edges__']
    idx = np.clip(np.searchsorted(edges, z, side='right') - 1, 0, len(edges) - 2)
    is_ref = np.stack([truncation[c]['is_ref'][idx] for c in all_names], 1)
    ref = np.argmax(is_ref, axis=1)
    n = np.arange(len(z))
    mu1 = np.array([truncation[all_names[r]]['mu1'][b] for r, b in zip(ref, idx)])
    V1 = np.array([truncation[all_names[r]]['V1'][b] for r, b in zip(ref, idx)])
    chi1 = np.array([truncation[all_names[r]]['chi2_1'][b] for r, b in zip(ref, idx)])
    E_rr = E[n, ref, ref]
    t1 = np.sqrt(chi1 * (V1 + E_rr))
    model_all = np.stack([spl[c]['a'](z) + spl[c]['b'](z) * (m - mref_f(z)) for c in all_names], 1)
    c_all = np.abs(np.stack([spl[c]['c'](z) for c in all_names], 1))
    for w, name in zip(trunc_t, colour_names):
        j = all_names.index(name)
        L = np.where(ref != j)[0]
        rL = ref[L]
        r = np.zeros(len(L))
        for k in range(4):
            mk = rL == k
            if mk.any() and k != j:
                key = (min(k, j), max(k, j))
                r[mk] = np.clip(rcs[key](rn[L][mk]), -0.99, 0.99) if key in rcs else 0.0
        w['leak'] = {'idx': L, 'r': r, 'c_ref': c_all[L, rL],
                     'E_rj': E[L, rL, j], 'S_rr': c_all[L, rL] ** 2 + E_rr[L],
                     'model_ref': model_all[L, rL], 'mu1': mu1[L], 't1': t1[L]}
        if exact:
            from .cross_covariance_trunc import ellipse_box_quadrature, chord_quadrature
            b = idx[L]
            G = V2full[b] + E[L]
            Grr, Gjj, Grj = G[np.arange(len(L)), rL, rL], G[:, j, j], G[np.arange(len(L)), rL, j]
            k4 = truncation[name]['chi2_4'][b]
            mu2r = mu2full[b, rL]
            off = mu1[L] - mu2r
            ext = np.sqrt(k4 * Grr)
            lo = np.maximum(-ext, off - t1[L])
            hi = np.minimum(ext, off + t1[L])
            U_r, U_j, W = ellipse_box_quadrature(Grr, Gjj, Grj, k4, lo, hi, n_quad=12, df_rest=2)
            x_j = X[L, j]
            Ur_num, Wnum = chord_quadrature(x_j - mu2full[b, j], Gjj, Grr, Grj, k4,
                                            off - t1[L], off + t1[L], n_quad=16, df_rest=2)
            w['leak'].update({'exact': True, 'U_r': U_r, 'U_j': U_j, 'W': W,
                              'Ur_num': Ur_num, 'Wnum': Wnum, 'x_j': x_j,
                              'mu2_j': mu2full[b, j], 'd_r': model_all[L, rL] - mu2r})
        print(f"  xcov leak [{name}]: {len(L)} galaxies with a different reference colour, "
              f"median r(ref,{name}) = {np.median(r) if len(r) else float('nan'):.3f}")


def _interp_rows(grid: np.ndarray, table: np.ndarray,
                 rows: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation of per-row tabulated functions on a SHARED grid.

    ``table`` is (n_rows, n_grid); ``rows[j]`` picks the row for point ``x[j]``.
    Sharing one colour grid across all (z, magnitude) cells is what makes this
    a couple of fancy-index lookups instead of one ``np.interp`` per cell -- it
    runs inside the objective, so it is called thousands of times.
    Outside the grid the edge value is held (the pdf tapers to ~0 there and the
    cdf is flat at 0 / 1, which is what we want in both cases).
    """
    n = grid.size
    idx = np.clip(np.searchsorted(grid, x), 1, n - 1)
    x0, x1 = grid[idx - 1], grid[idx]
    y0, y1 = table[rows, idx - 1], table[rows, idx]
    y = y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    y = np.where(x <= grid[0], table[rows, 0], y)
    return np.where(x >= grid[-1], table[rows, -1], y)


def load_background(filepath: str) -> Dict:
    """Load the Eq.31 background model written by ``measure_background.py``.

    redMaPPer model contamination rather than cutting it (Rykoff et al. 2014,
    arXiv:1303.3562, Eq. 31)::

        P(c) = p_mem * G(c) + (1 - p_mem) * b(c, m_i)

    ``G`` is the (truncated, Eq.29) red-sequence Gaussian and ``b`` is the
    measured colour-magnitude distribution of NON-members. Eq.29 alone fixes
    the bias from cutting the data (it pulls c UP); Eq.31 fixes the bias from
    the cut not working perfectly (it pulls c DOWN). Both are needed -- with
    only Eq.29 the scatter absorbs the blue cloud, which is what makes
    c_gr(z=1) want ~0.5.

    ``b`` is a FIXED external input, exactly like the truncation limit ``t``.
    Fitting it jointly with c(z) would let the two trade against each other,
    the same oscillation that forbids iterating the stage-A/B edge measurement.

    Returns
    -------
    dict
        ``{'colors', 'z_nodes', 'mag_edges', 'per_colour': {name: {grid, pdf,
        cdf}}}``, with ``pdf``/``cdf`` shaped (n_z, n_mag, n_grid).
    """
    d = np.load(filepath, allow_pickle=True)
    names = [str(x) for x in d['colors']]
    out = {
        'colors': names,
        'z_nodes': np.asarray(d['z_nodes'], dtype=float),
        'mag_edges': np.asarray(d['mag_edges'], dtype=float),
        'source': str(d['source_parent']) if 'source_parent' in d.files else '',
        'per_colour': {},
    }
    for n in names:
        out['per_colour'][n] = {
            'grid': np.asarray(d[f'{n}_grid'], dtype=float),
            'pdf': np.asarray(d[f'{n}_pdf'], dtype=float),
            'cdf': np.asarray(d[f'{n}_cdf'], dtype=float),
        }
    return out


def background_terms(background: Optional[Dict], colour_names: List[str],
                     colors: np.ndarray, mi_j: np.ndarray, z_j: np.ndarray,
                     trunc_t: Optional[List[np.ndarray]] = None,
                     ref_colours: Optional[List[np.ndarray]] = None
                     ) -> Optional[List[Dict]]:
    """Precompute ``log b(c_j, m_j)``, the Eq.31 background term, per colour.

    Everything here is parameter-INDEPENDENT, so the objective does no
    interpolation at all -- it just sums a stored array.

    THE WINDOW MUST NOT MOVE WITH THE FIT. ``b`` has to be normalised over the
    step-1 acceptance window (the mixture is only a density if both components
    integrate to 1 over the same support), and the obvious implementation --
    centre the window on the model colour -- is UNBOUNDED. Slide the model away
    from the data and the background's mass inside the window goes to zero,
    so the renormalised ``b`` diverges and the fit is paid to abandon the
    ridgeline. Measured on a mock: the log-background sum grows without limit,
    1.7e5 at zero offset -> 3.9e6 at 1.2 mag, and the fit duly collapses to
    c = c_min, p = p_min.

    redMaPPer avoid this by construction -- their cut is "1.5 sigma about the
    MEDIAN color", a fixed data-derived centre (Rykoff et al. 2014 Sect. 6.4).
    We do the same: ``ref_colours`` is the stage-A ridgeline colour per galaxy,
    frozen, so the window is a fixed per-galaxy interval and the normalisation
    is a constant. Same reason ``t`` itself is frozen.

    Parameters
    ----------
    trunc_t : list of ndarray, optional
        Per-galaxy step-1 half-width. Without it the background is normalised
        over its full tabulated range (no truncation to correct for).
    ref_colours : list of ndarray, optional
        Per-galaxy FIXED window centre, one array per colour. Required whenever
        ``trunc_t`` is given -- there is no safe default, see above.

    Returns
    -------
    list of dict or None
        ``[{'log_b': ndarray}, ...]``; ``None`` when no background is supplied,
        which restores the pure red-sequence likelihood exactly.
    """
    if background is None:
        return None
    if trunc_t is not None and ref_colours is None:
        raise ValueError(
            "the Eq.31 background needs a FIXED window centre (ref_colours) "
            "whenever a truncation model is used; normalising over a window "
            "centred on the moving model makes the likelihood unbounded")
    z_nodes = background['z_nodes']
    edges = background['mag_edges']
    n_mag = len(edges) - 1
    iz = np.abs(np.asarray(z_j, float)[:, None] - z_nodes[None, :]).argmin(axis=1)
    im = np.clip(np.digitize(np.asarray(mi_j, float), edges[1:-1]), 0, n_mag - 1)
    cell = iz * n_mag + im

    terms = []
    for i, name in enumerate(colour_names):
        if name not in background['per_colour']:
            raise KeyError(
                f"background model has no entry for colour {name!r}; "
                f"it covers {sorted(background['per_colour'])}")
        blk = background['per_colour'][name]
        n_z, n_m, n_g = blk['pdf'].shape
        if n_m != n_mag:
            raise ValueError(f"background {name}: pdf has {n_m} magnitude bins "
                             f"but mag_edges implies {n_mag}")
        grid = blk['grid']
        pdf2d = blk['pdf'].reshape(n_z * n_m, n_g)
        b_pdf = _interp_rows(grid, pdf2d, cell, np.asarray(colors[i], float))
        # Floor: a colour where the background has no measured density simply
        # cannot be explained as contamination -- correct -- but log(0) is not.
        log_b = np.log(np.clip(b_pdf, 1e-12, None))

        if trunc_t is not None:
            cdf2d = blk['cdf'].reshape(n_z * n_m, n_g)
            ref = np.asarray(ref_colours[i], float)
            hi = _interp_rows(grid, cdf2d, cell, ref + trunc_t[i])
            lo = _interp_rows(grid, cdf2d, cell, ref - trunc_t[i])
            log_b = log_b - np.log(np.clip(hi - lo, 1e-6, None))
        terms.append({'log_b': log_b})
    return terms


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
                 loss: str = 'l2',
                 trunc_t: Optional[List[np.ndarray]] = None,
                 background_t: Optional[List[Dict]] = None):
        self.colors, self.mi_j, self.z_j = galaxy_data
        self.C_obs = C_obs
        self.mi_ref = mi_ref
        self.nodes = nodes
        self.n_colors = n_colors
        # Per-galaxy step-1 selection edge |delta| < t, one array per colour, or
        # None for the plain untruncated likelihood. See load_truncation().
        self.trunc_t = trunc_t
        if trunc_t is not None and loss == 'l1':
            raise NotImplementedError(
                "truncation correction is only derived for loss='l2' (Gaussian); "
                "a truncated Laplace needs a different normalisation")
        # Where the step-1 window sits, settable via RIDGELINE_TRUNC_CENTRE:
        #   'model'    (default, legacy): window |c - model(m)| < t, centred on
        #              the ridgeline at the galaxy's own magnitude. Right for
        #              Vakili's conditional stage-1 cut. The surviving mass is
        #              then independent of b -- this term corrects c only.
        #   'marginal': window |c - a(z)| < t, centred on the marginal mean
        #              (~ the intercept, since m_ref is the sample median
        #              magnitude) and the SAME for every magnitude. Right for
        #              the hybrid / redMaPPer component-mode stage-1 cut. The
        #              surviving mass now depends on d = b(z)(m - m_ref), so
        #              the normalisation has a gradient in b and un-flattens
        #              the slope that a magnitude-blind strip imprints
        #              (retained slope = lambda(k) b, lambda(1.5) ~ 0.55).
        # t must have been measured in the matching coordinate
        # (measure_step1_edge.py --about ridgeline | marginal).
        import os as _os
        self.trunc_centre = _os.environ.get("RIDGELINE_TRUNC_CENTRE", "model").lower()
        if self.trunc_centre not in ("model", "marginal", "twostage"):
            raise ValueError(f"RIDGELINE_TRUNC_CENTRE must be 'model', 'marginal' or "
                             f"'twostage', got {self.trunc_centre!r}")
        if self.trunc_centre == "twostage":
            if trunc_t is not None and not all(isinstance(t, dict) for t in trunc_t):
                raise ValueError("RIDGELINE_TRUNC_CENTRE=twostage needs a two-stage window "
                                 "model (src/build_twostage_truncation.py), not an edge npz")
            self._gl_nodes, self._gl_weights = np.polynomial.legendre.leggauss(16)
        if self.trunc_centre in ("marginal", "twostage") and background_t is not None:
            raise NotImplementedError(
                "RIDGELINE_TRUNC_CENTRE=marginal is only implemented for the pure "
                "truncated likelihood, not the Eq.31 background mixture")
        # Precomputed Eq.31 background terms, one dict per colour, or None for
        # the pure red-sequence likelihood. See background_terms().
        self.background_t = background_t
        if background_t is not None:
            if loss == 'l1':
                raise NotImplementedError(
                    "the Eq.31 mixture is only derived for loss='l2' (Gaussian)")
            if 'p' not in nodes:
                raise ValueError(
                    "a background model was supplied but nodes has no 'p' entry; "
                    "call setup_spline_nodes(..., delta_p=...)")
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
            if background_t is not None:
                self.param_names += [f'p_{i}_{j}'
                                     for j in range(len(nodes['p']))]

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

            log_erf = None
            if self.trunc_t is not None:
                # Step 1 kept only |residual| < t, so the data are a truncated
                # draw and the model must be normalised by the surviving mass
                # (redMaPPer Eq. 29). t is FIXED input, not a fit parameter --
                # that is what makes this term informative: shrinking c shrinks
                # sigma, raises t/sigma, drives erf -> 1 and its log -> 0, which
                # penalises exactly the collapse the plain Gaussian rewards.
                if self.trunc_centre == "twostage" and self.trunc_t[i].get('leak', {}).get('exact'):
                    log_erf = log_twostage_leak_exact(
                        self.trunc_t[i], model, sigma, c_z,
                        self._gl_nodes, self._gl_weights)
                elif self.trunc_centre == "twostage" and 'leak' in self.trunc_t[i]:
                    log_erf, log_box = log_twostage_leak_mass(
                        self.trunc_t[i], model, sigma, c_z, residual,
                        self._gl_nodes, self._gl_weights)
                    # the box acceptance at the observed colour depends on the
                    # parameters: it is part of the density, not a constant
                    log_erf = log_erf - log_box
                elif self.trunc_centre == "twostage":
                    log_erf = log_twostage_mass(self.trunc_t[i], model, sigma,
                                                self._gl_nodes, self._gl_weights)
                elif self.trunc_centre == "marginal":
                    # Window centred on the marginal mean a(z), not on the
                    # model: the galaxy's model mean sits d = b(m - m_ref)
                    # away from the window centre, so the surviving mass is
                    #   Phi((t - d)/sigma) - Phi((-t - d)/sigma)
                    # = 1/2 [erf((t - d)/sqrt2 sigma) + erf((t + d)/sqrt2 sigma)].
                    # Reduces to erf(t/sqrt2 sigma) at d = 0. Dividing the
                    # likelihood by this rewards putting mass OUTSIDE the
                    # window for galaxies far from m_ref, i.e. a steeper b
                    # than the strip-flattened sample mean -- the standard
                    # truncated-normal MLE, consistent for the slope.
                    d = b_z * (self.mi_j - self.mi_ref)
                    log_erf = log_window_mass(self.trunc_t[i], d, sigma)
                else:
                    frac = erf(self.trunc_t[i] / (np.sqrt(2.0) * sigma))
                    log_erf = np.log(np.clip(frac, 1e-300, None))

            if self.background_t is not None:
                # redMaPPer Eq. 31: explain the contaminants instead of letting
                # sigma widen to cover them. This REPLACES the pure NLL rather
                # than adding to it, so the plain chi^2 is never computed.
                nll = self._mixture_nll(i, p, residual, sigma, log_erf)
            else:
                if self.loss == 'l1':
                    # Laplace -2 log L. With scale b = sigma/sqrt(2) (so the
                    # Laplace variance equals sigma^2), -2logL =
                    # 2*sqrt(2)*|r|/sigma + 2*log(sigma) + const. Robust to the
                    # outlier tails that inflate the Gaussian c(z); c(z) stays a
                    # Gaussian-equivalent sigma.
                    nll = (2.0 * np.sqrt(2.0) * np.sum(np.abs(residual) / sigma)
                           + 2.0 * np.sum(np.log(sigma)))
                else:
                    # Gaussian -2 log L (chi^2 + 2 log sigma).
                    nll = np.sum(residual**2 / sigma**2) + 2 * np.sum(np.log(sigma))
                if log_erf is not None:
                    nll += 2 * np.sum(log_erf)
            total_loss += nll

        return total_loss


    def _mixture_nll(self, i: int, p: Dict[str, float], residual: np.ndarray,
                     sigma: np.ndarray,
                     log_erf: Optional[np.ndarray]) -> float:
        """-2 log L for the redMaPPer Eq.31 two-component mixture, one colour.

            P(c) = p_mem(z) * G(c) + (1 - p_mem(z)) * b(c, m)

        ``G`` is the same (optionally Eq.29-truncated) Gaussian the pure model
        uses; ``b`` is the frozen background density. Contaminants are
        EXPLAINED by ``b`` instead of forcing ``sigma`` to widen to cover them
        -- the bias Eq.29 alone cannot touch, because Eq.29 corrects for the
        cut being made and this corrects for the cut being imperfect.

        Normalisation is the part that is easy to get wrong: a mixture is only
        a density if both components integrate to 1 over the SAME support --
        the step-1 acceptance window. ``G`` is normalised there by the Eq.29
        erf; ``b`` is divided by its mass in the window inside
        :func:`background_terms`, against a FROZEN window centre. Letting that
        window follow the fitted model makes the likelihood unbounded; see the
        note there before changing it.

        The trailing ``-n*log(2*pi)`` makes this reduce EXACTLY to the pure
        expression as p_mem -> 1, so NLLs stay comparable with every fit made
        before the mixture existed.
        """
        p_vals = np.array([p[f'p_{i}_{j}'] for j in range(len(self.nodes['p']))])
        # A cubic interpolant can overshoot [0, 1] between nodes even when the
        # nodes themselves are bounded, so clip the CURVE, not just the nodes.
        p_z = np.clip(interpolate_parameters(self.nodes['p'], p_vals, self.z_j),
                      1e-9, 1.0)

        log_g = -0.5 * residual**2 / sigma**2 - np.log(sigma) - 0.5 * LOG_2PI
        if log_erf is not None:
            log_g = log_g - log_erf

        log_b = self.background_t[i]['log_b']

        with np.errstate(divide='ignore'):
            log_mix = np.logaddexp(np.log(p_z) + log_g,
                                   np.log1p(-p_z) + log_b)
        return -2.0 * float(np.sum(log_mix)) - len(self.z_j) * LOG_2PI


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
                 loss: str = 'l2',
                 trunc_t: Optional[List[np.ndarray]] = None,
                 background_t: Optional[List[Dict]] = None):
        super().__init__(galaxy_data, C_obs, mi_ref, nodes, n_colors, loss=loss,
                         background_t=background_t,
                         trunc_t=trunc_t)
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

        kinds = ['a', 'b', 'c'] + (['p'] if 'p' in model.nodes else [])
        for kind in kinds:
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
                    cap_last_node: bool = False,
                    trunc_t: Optional[List[np.ndarray]] = None,
                    background_t: Optional[List[Dict]] = None,
                    delta_p: Optional[float] = None) -> Dict:
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
                               cap_last_node=cap_last_node,
                               delta_p=(None if background_t is None
                                        else (delta_p or delta_c)))

    if colour_names is None:
        colour_names = [str(i) for i in range(n_colors)]

    # Choose model class
    if regularization_config:
        model = RedSequenceModelRegularized(galaxy_data, C_obs, mi_ref, nodes,
                                           n_colors, regularization_config,
                                           loss=loss, trunc_t=trunc_t,
                                           background_t=background_t)
    else:
        model = RedSequenceModel(galaxy_data, C_obs, mi_ref, nodes, n_colors,
                                 loss=loss, trunc_t=trunc_t,
                                 background_t=background_t)

    # Initial values and limits
    init = []
    limits = []
    for _ in range(n_colors):
        init += [1.0] * len(nodes['a'])
        limits += [(0.2, 3.5)] * len(nodes['a'])

        init += [0.0] * len(nodes['b'])
        limits += [(-0.5, 0.5)] * len(nodes['b'])

        init += [0.1] * len(nodes['c'])
        limits += [c_bounds()] * len(nodes['c'])

        if background_t is not None:
            init += [0.9] * len(nodes['p'])
            limits += [p_bounds()] * len(nodes['p'])

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


def _run_minuit_robust(model, init, limits, rail_rtol: float = 0.05):
    """Simplex-seeded MIGRAD with a railed-parameter rescue pass.

    MIGRAD's internal bound transform has zero gradient at box limits, so a
    parameter that drifts onto its bound inside a flat likelihood valley gets
    stuck there while the fit still reports success (seen on the last c(z)
    node of the r-i ridgeline). Strategy: simplex pre-fit + strategy=2, then
    if any parameter lands within ``rail_rtol`` of its bound, re-seed those
    parameters away from the bound and re-run MIGRAD; keep the better NLL.
    """
    m = Minuit(model, *init)
    m.errordef = 1.0
    m.limits = limits
    m.strategy = 2
    m.simplex()
    m.migrad()

    def railed_indices(values):
        out = []
        for k, (v, (lo, hi)) in enumerate(zip(values, limits)):
            span = hi - lo
            if v - lo < rail_rtol * span or hi - v < rail_rtol * span:
                out.append(k)
        return out

    railed = railed_indices(m.values)
    if railed:
        reinit = list(m.values)
        for k in railed:
            lo, hi = limits[k]
            # pull railed parameters to the inner quarter of their range
            reinit[k] = lo + 0.25 * (hi - lo) if reinit[k] - lo < hi - reinit[k] \
                else hi - 0.25 * (hi - lo)
        m2 = Minuit(model, *reinit)
        m2.errordef = 1.0
        m2.limits = limits
        m2.strategy = 2
        m2.migrad()
        print(f"  [minuit-robust] {len(railed)} railed param(s) "
              f"{[model.param_names[k] for k in railed]}: "
              f"rescue NLL={m2.fval:.1f} vs original NLL={m.fval:.1f} -> "
              f"{'rescue' if m2.fval < m.fval else 'original'} kept", flush=True)
        if m2.fval < m.fval:
            m = m2
    return m


def fit_red_sequence_single_colour(galaxy_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                   C_obs: np.ndarray, mi_ref: np.ndarray,
                                   z_min: float, z_max: float,
                                   delta_a: float, delta_b: float, delta_c: float,
                                   colour_names: Optional[List[str]] = None,
                                   regularization_config: Optional[Dict] = None,
                                   loss: str = 'l2',
                                   cap_last_node: bool = False,
                                   trunc_t: Optional[List[np.ndarray]] = None,
                                   background_t: Optional[List[Dict]] = None,
                                   delta_p: Optional[float] = None) -> Dict:
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
    # p(z) shares the c(z) spacing by default: the contaminated fraction is as
    # noisy to estimate as the scatter, so it gets the same coarse nodes.
    nodes = setup_spline_nodes(z_min, z_max, delta_a, delta_b, delta_c,
                               cap_last_node=cap_last_node,
                               delta_p=(None if background_t is None
                                        else (delta_p or delta_c)))

    if colour_names is None:
        colour_names = [str(i) for i in range(n_colors)]

    results = {'colors': colour_names}

    # Fit each color independently
    for i in range(n_colors):
        data_i = ([colors[i]], mi_j, z_j)

        t_i = None if trunc_t is None else [trunc_t[i]]
        bg_i = None if background_t is None else [background_t[i]]
        if regularization_config:
            model = RedSequenceModelRegularized(data_i, [C_obs[i]], mi_ref,
                                               nodes, 1, regularization_config,
                                               loss=loss, trunc_t=t_i,
                                               background_t=bg_i)
        else:
            model = RedSequenceModel(data_i, [C_obs[i]], mi_ref, nodes, 1,
                                     loss=loss, trunc_t=t_i, background_t=bg_i)

        init = ([1.0] * len(nodes['a']) +
                [0.0] * len(nodes['b']) +
                [0.1] * len(nodes['c']))
        import os as _os
        if _os.environ.get("RIDGELINE_A_INIT", "").lower() == "median":
            # Start a(z) at the binned median colour. The flat a=1.0 start can
            # sit ~5 sigma outside the selection window (r-i at z<0.4 has
            # a~0.3-0.45), where a truncated likelihood has a spurious basin:
            # a far mean with a huge sigma mimics an exponential slope across
            # the window (878330: exact-leak r-i fit stuck there, a=1.6,
            # c=0.27, NLL +8000 above the near-truth solution).
            _za, _col = np.asarray(z_j, float), np.asarray(colors[i], float)
            _h = 0.5 * (nodes['a'][1] - nodes['a'][0]) if len(nodes['a']) > 1 else 0.05
            for _k, _zn in enumerate(nodes['a']):
                _sel = np.abs(_za - _zn) <= _h
                if _sel.sum() < 20:
                    _sel = np.argsort(np.abs(_za - _zn))[:200]
                init[_k] = float(np.clip(np.median(_col[_sel]), 0.2, 3.5))
        limits = ([(0.2, 3.5)] * len(nodes['a']) +
                 [(-0.5, 0.5)] * len(nodes['b']) +
                 [c_bounds()] * len(nodes['c']))
        if bg_i is not None:
            # Start at 0.9: high purity is the prior for an already
            # red-SELECTED sample, and starting at 1.0 sits on the bound.
            # Settable via RIDGELINE_P_INIT because that prior is FALSE for an
            # unselected sample: fitting the parent directly (the no-cut
            # consistency test, where there is no truncation to correct) has a
            # true red fraction of ~0.2-0.5, so 0.9 starts the optimiser on the
            # wrong side of the mixture and, with the default p_min=0.5 floor,
            # on a bound above the true value. Lower BOTH together or the test
            # is rigged. Default is unchanged, so every cut-sample fit is
            # byte-identical.
            import os as _os2
            _p_init = float(_os2.environ.get("RIDGELINE_P_INIT", "0.9"))
            _p_lo, _p_hi = p_bounds()
            _p_init = min(max(_p_init, _p_lo), _p_hi)
            init += [_p_init] * len(nodes['p'])
            limits += [p_bounds()] * len(nodes['p'])

        import os as _os
        if _os.environ.get("RIDGELINE_OPTIMIZER", "").lower() == "lbfgsb":
            from scipy.optimize import minimize as _scipy_min
            r0 = _scipy_min(lambda x: model(*x), np.asarray(init, float),
                            method="L-BFGS-B", bounds=limits,
                            options={"maxiter": 20000, "maxfun": 200000,
                                     "ftol": 1e-12, "gtol": 1e-10})
            print(f"  [lbfgsb] colour {colour_names[i]}: success={r0.success} "
                  f"nit={r0.nit} NLL={r0.fun:.1f} ({r0.message})", flush=True)
            # success=True is reported at a bound just as at an interior
            # minimum, so a railed c(z) node is a CENSORED fit that looks
            # converged. Say so loudly -- see c_bounds().
            names = model.param_names
            _byname = dict(zip(names, np.asarray(r0.x, float)))
            # Log the fitted nodes per colour: the npz is only written after
            # ALL colours finish, so a wall-time kill on a later colour would
            # otherwise lose this one (jobs 870012/870023 lost their g-r).
            for _kind in ('a', 'b', 'c'):
                _vals = [_byname[f'{_kind}_0_{k}'] for k in range(len(nodes[_kind]))]
                print(f"  [lbfgsb] colour {colour_names[i]} {_kind}(z) nodes: "
                      + " ".join(f"{v:.4f}" for v in _vals), flush=True)
            for _kind, _bnd in (('c', c_bounds()),
                                ('p', p_bounds() if bg_i is not None else None)):
                if _bnd is None or _kind not in nodes:
                    continue
                _lo, _hi = _bnd
                _vals = [_byname[f'{_kind}_0_{k}'] for k in range(len(nodes[_kind]))]
                _rail = [(float(nodes[_kind][k]), float(v))
                         for k, v in enumerate(_vals)
                         if v >= _hi * (1 - 1e-6) or v <= _lo * (1 + 1e-6)]
                if _rail:
                    print(f"  [lbfgsb] WARNING colour {colour_names[i]}: "
                          f"{len(_rail)} {_kind}(z) node(s) AT A BOUND "
                          f"[{_lo:g}, {_hi:g}] -- censored, not converged: "
                          + ", ".join(f"z={z:.2f}:{_kind}={v:.4f}"
                                      for z, v in _rail), flush=True)
            params = dict(zip(names, r0.x))
            try:
                errors = dict(zip(names, np.sqrt(np.diag(
                    r0.hess_inv.todense()))))
            except Exception:
                errors = dict(zip(names, np.zeros(len(names))))
            sub = extract_named_params(model, params,
                                       [colour_names[i]], errors)
            results[colour_names[i]] = sub[colour_names[i]]
            continue

        m = _run_minuit_robust(model, init, limits)

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
        for kind in (['a', 'b', 'c'] + (['p'] if bg_i is not None else [])):
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
        limits += [c_bounds()] * len(nodes['c'])

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
                m_ref_values: Optional[np.ndarray] = None,
                m_ref_smooth: Optional[bool] = None,
                m_ref_smooth_s: Optional[float] = None) -> None:
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
    # Eq.31 membership p(z), present only when a background model was fitted.
    has_p = 'p' in results[first_color]
    if has_p:
        flat_dict['nodes_p'] = results[first_color]['p']['nodes']

    # Save values/errors per color
    for color in results['colors']:
        for kind in (['a', 'b', 'c'] + (['p'] if has_p else [])):
            flat_dict[f'{color}_{kind}_values'] = results[color][kind]['values']
            flat_dict[f'{color}_{kind}_errors'] = results[color][kind]['errors']

    if m_ref_z is not None and m_ref_values is not None:
        flat_dict['m_ref_z'] = m_ref_z
        flat_dict['m_ref_values'] = m_ref_values
        # How the fit turned the nodes into m_ref(z): loaders must rebuild the
        # SAME pivot (RedCatalogue.load_ridge_line reads these by default).
        if m_ref_smooth is not None:
            flat_dict['m_ref_smooth'] = np.array(bool(m_ref_smooth))
            flat_dict['m_ref_smooth_s'] = np.array(float(m_ref_smooth_s if m_ref_smooth_s is not None else 0.15))

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
    if 'nodes_p' in data.files:
        nodes['p'] = data['nodes_p']

    # Reconstruct per-color structure
    for color in colors:
        results[color] = {}
        for kind in (['a', 'b', 'c'] + (['p'] if 'p' in nodes else [])):
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
        kinds = ['a', 'b', 'c'] + (['p'] if 'p' in results[col] else [])
        for kind in kinds:
            z_nodes = np.asarray(results[col][kind]['nodes'])
            values = np.asarray(results[col][kind]['values'])
            # p(z) is the Eq.31 membership fraction, never smoothed: it is a
            # nuisance parameter read for diagnostics, not a ridgeline.
            spline_dict[col][kind] = build_param_interp(
                z_nodes, values, method=method,
                smooth_s=(0.0 if kind == 'p' else smooth_s))

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
                 cap_last_node: bool = False,
                 truncation: Optional[Dict[str, Tuple[Callable, Callable]]] = None,
                 background: Optional[Dict] = None,
                 background_ref: Optional[Dict] = None,
                 delta_p: Optional[float] = None):
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
        # Step-1 selection-edge model {colour: (kappa_f, const_f)} or None.
        self.truncation = truncation
        # Eq.31 background model (from load_background) or None, and the node
        # spacing for the membership p(z) it introduces (defaults to delta_c).
        self.background = background
        # Frozen stage-A splines {colour: {'a','b','c'}} giving the window
        # centre for the Eq.31 normalisation. See background_terms().
        self.background_ref = background_ref
        self.delta_p = delta_p

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

        trunc_t = truncation_limits(self.truncation, colour_names,
                                    galaxy_data[2], C_obs)
        # Optional stage-1 box leakage through the intrinsic correlations
        # (RIDGELINE_XCOV_LEAK = merged ridgeline with r_cross, from
        # src/fit_xcov_pairwise.py); two-stage windows only.
        import os as _os
        leak_path = _os.environ.get("RIDGELINE_XCOV_LEAK")
        if leak_path:
            if trunc_t is None or self.truncation.get('__format__') != 'twostage':
                raise ValueError("RIDGELINE_XCOV_LEAK needs a two-stage --truncation window")
            ctx = getattr(self, '_leak_ctx', None)
            if ctx is None:
                raise ValueError("RIDGELINE_XCOV_LEAK: no data context (set by "
                                 "RedCatalogue.fit_ridge_line)")
            if verbose:
                print(f"  Stage-1 box leakage via r_cross from {leak_path}")
            build_xcov_leak(leak_path, ctx['df'], ctx['color_definitions'], colour_names,
                            ctx['magnitude_col'], ctx['z_spec_col'], trunc_t,
                            self.truncation, smooth_mref=ctx['smooth_mref'])
        if verbose and trunc_t is not None:
            med = [(f"{n}:t1={np.median(t['t1'][np.isfinite(t['t1'])]) if np.isfinite(t['t1']).any() else float('inf'):.4f}"
                    f"/t2={np.median(t['t2']):.4f}") if isinstance(t, dict)
                   else f"{n}:{np.median(t):.4f}" for n, t in zip(colour_names, trunc_t)]
            import os as _os
            print(f"  Truncation-corrected fit (redMaPPer Eq.29); window centre = "
                  f"{_os.environ.get('RIDGELINE_TRUNC_CENTRE', 'model').lower()}; "
                  f"median t per colour: {', '.join(med)}")

        ref_colours = None
        if self.background is not None and trunc_t is not None:
            if self.background_ref is None:
                raise ValueError(
                    "Eq.31 with Eq.29 needs --background-ref: a frozen stage-A "
                    "ridgeline giving the centre of the step-1 window. Without "
                    "it the background normalisation follows the fitted model "
                    "and the likelihood is unbounded (background_terms).")
            z_j, mi_j = galaxy_data[2], galaxy_data[1]
            ref_colours = []
            for n in colour_names:
                sp = self.background_ref[n]
                ref_colours.append(sp['a'](z_j) + sp['b'](z_j) * (mi_j - mi_ref))
        background_t = background_terms(self.background, colour_names,
                                        galaxy_data[0], galaxy_data[1],
                                        galaxy_data[2], trunc_t=trunc_t,
                                        ref_colours=ref_colours)
        if verbose and background_t is not None:
            print(f"  Background mixture (redMaPPer Eq.31) on {len(colour_names)} "
                  f"colour(s); p(z) nodes every "
                  f"{self.delta_p or self.delta_c}, bounds {p_bounds()}")
            if trunc_t is None:
                print("  WARNING Eq.31 without Eq.29: the mixture is being "
                      "normalised over the FULL colour range, but step 1 "
                      "truncated the sample. Pass --truncation too.")

        if method == 'single':
            self.results = fit_red_sequence_single_colour(
                galaxy_data, C_obs, mi_ref,
                self.z_min, self.z_max,
                self.delta_a, self.delta_b, self.delta_c,
                colour_names, self.regularization_config,
                loss=self.loss, cap_last_node=self.cap_last_node,
                trunc_t=trunc_t, background_t=background_t,
                delta_p=self.delta_p,
            )
        elif method == 'joint':
            self.results = fit_red_sequence(
                galaxy_data, C_obs, mi_ref,
                self.z_min, self.z_max,
                self.delta_a, self.delta_b, self.delta_c,
                colour_names, self.regularization_config,
                loss=self.loss, cap_last_node=self.cap_last_node,
                trunc_t=trunc_t, background_t=background_t,
                delta_p=self.delta_p,
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
