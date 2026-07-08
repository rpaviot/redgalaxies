"""Iterative full-colour-space red-sequence selection (IterativeRedSelector).

A from-scratch alternative to the two-stage Vakili GMM. There is NO ridgeline,
NO step-1, and NO hardcoded ``epsilon`` / ``pi_min`` / ``color_switch_z``. The
red sequence is defined directly as the reddest component of a multi-component
XD GMM fit in the full feature space ``(m, g-r, r-i, i-z, v-H)``, per z-slice,
refined over a few iterations.

Two iteration philosophies are implemented, selectable via ``refit_on``:

``refit_on="pool"`` (default, the robust design)
    Every iteration re-fits the GMM on the **entire slice** — the candidate pool
    is never shrunk. The component count is held fixed and the reddest component
    is warm-started from the previous fit, so iterations *refine* the red
    definition rather than re-discovering it. Because the full blue/transition
    population is present in every fit as an *anchor*, "red" is always defined
    relative to "not-red" and the red component's intrinsic scatter is never
    clipped at its own wings: the boundary stabilises instead of collapsing.
    (Note: re-fitting identical full data is a fixed point of EM, so this
    typically converges at iteration 2 — that stability is the point.)

``refit_on="selected"`` (the naive peel — kept for comparison)
    Every iteration re-fits the GMM on **only the 2-sigma-selected reds** from the
    previous pass. This is iterative sigma-clipping: with no blue anchor the
    intrinsic scatter shrinks monotonically by construction, and both stopping
    criteria below fire regardless of whether the boundary is correct. Provided so
    the scatter-collapse pathology can be measured directly against ``"pool"``.

Per slice the loop stops when EITHER the reddest component's mean and intrinsic
scatter (in the red colour) are stable to ``mu_tol`` / ``sigma_tol``, OR its
weight exceeds ``weight_stop`` (default 0.98), OR the active set is unchanged, OR
``max_iter`` is reached. ``history`` records n_active, n_selected, red mean and
intrinsic scatter at every iteration of every slice — that is the
scatter-vs-iteration diagnostic.

The reddest component is the one with the highest mean in the *z-appropriate*
colour: as the 4000A break redshifts out of a baseline, the discriminating
colour follows it (g-r -> r-i -> i-z), set by ``red_color_switches``. The single
free statistical knob is ``n_sigma``.
"""
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.special import erf

from .gmm_selection import XDGMM
from .utils import create_color_arrays_and_covariance


def _fit_one_seed(X, Xerr, ncomp, seed, init_method='bayesian'):
    """Single XDGMM fit at one seed; returns (logL, mu, V, alpha).

    Module-level so it pickles cleanly for joblib's loky backend. ``init_method``
    selects how XDGMM seeds its EM (Bayesian-DP vs plain Gaussian mixture).
    """
    clf = XDGMM(ncomp, max_iter=400, random_state=seed,
                init_method=init_method).fit(X, Xerr)
    return float(clf.logL(X, Xerr)), clf.mu, clf.V, clf.alpha


def _sigma_to_conf(n_sigma: float) -> float:
    """1D-equivalent confidence level for an n-sigma cut (e.g. 2 -> 0.9545)."""
    return float(erf(n_sigma / np.sqrt(2.0)))


def _mahalanobis_d2(X, Xerr, mu, V) -> np.ndarray:
    """Squared joint Mahalanobis distance of each point to (mu, V+Xerr).

    Both the intrinsic covariance ``V`` and each galaxy's photometric error
    ``Xerr`` enter the distance. Compare to ``chi2.ppf(conf, df=n_features)`` for
    an n-sigma cut; storing D2 lets the sample be re-thresholded to any sigma
    without re-fitting.
    """
    d = X - mu
    Cinv = np.linalg.inv(V[None, :, :] + Xerr)
    return np.einsum('ni,nij,nj->n', d, Cinv, d)


def _mahalanobis_keep(X, Xerr, mu, V, n_sigma) -> np.ndarray:
    """Boolean mask: points within ``n_sigma`` joint Mahalanobis (1D-equiv conf)."""
    crit = chi2.ppf(_sigma_to_conf(n_sigma), df=X.shape[1])
    return _mahalanobis_d2(X, Xerr, mu, V) < crit


class IterativeRedSelector:
    """Iterative full-colour-space GMM red-sequence selection.

    Parameters
    ----------
    df : DataFrame
        Parent candidate catalogue (red AND blue) with ``mag_{band}`` /
        ``mag_{band}_err`` columns and a spectroscopic redshift column.
    color_definitions : list of (band1, band2)
        Colours as band1 - band2, e.g. ``[('g','r'),('r','i'),('i','z'),('v','h')]``.
    magnitude_col : str
        Reference magnitude included as the first feature (default ``mag_v2``).
    refit_on : {"pool", "selected"}
        Iteration philosophy (see module docstring). Default ``"pool"``.
    n_sigma : float
        Selection contour, joint Mahalanobis (default 2.0).
    n_components_first, n_components_rest : int
        Components for the first fit vs. all later fits. Default 4 then 2. NOTE:
        4 on the first pass is deliberate — a 3-component first fit is documented
        to let the blue tail bleed into the red component (NMAD +41%).
    z_bin_width : float
        Width of the (non-overlapping) z-slices (default 0.03).
    weight_stop : float
        Stop a slice once the red component's weight exceeds this (default 0.98).
    min_red_weight : float
        The red component is the reddest one whose mixture weight exceeds this
        (default 0.05). Excludes the negligible-weight, very broad catch-all
        components that best-by-logL adds to absorb outliers and that the plain
        reddest-mean rule would otherwise mistake for the red sequence.
    mu_tol, sigma_tol : float
        Relative convergence tolerances on the red component's mean and intrinsic
        scatter in the red colour (default 1e-3).
    max_iter : int
        Per-slice iteration cap (default 10).
    min_in_slice : int
        Skip slices with fewer galaxies than this (default 50).
    n_restarts : int
        Number of EM restarts (distinct seeds) for the FIRST fit of each slice;
        the highest-logL fit is kept. Guards against the multi-modal local optima
        that make a single fixed-seed fit unreliable on hard slices. The good
        (highest-logL) basin can be rare — e.g. ~10% of seeds at z~0.375 — so
        reliably hitting it needs enough restarts (~30 for >95%). Warm-started
        iterations are deterministic and not restarted. Default 1 (no restart).
    n_jobs : int
        Parallel workers for the restarts (joblib/loky). Default 1 (serial); -1
        uses all cores. Each worker is pinned to one BLAS thread to avoid
        oversubscription.
    xd_init : {'bayesian', 'gaussian'}
        How XDGMM seeds its EM. 'bayesian' (default) = BayesianGaussianMixture
        (Dirichlet-process prior, can shrink weak components); 'gaussian' = plain
        GaussianMixture (vanilla EM). The XD EM itself is unchanged — only the
        starting point differs, which can change which local optimum is reached.
    """

    def __init__(self, df: pd.DataFrame,
                 color_definitions: List[Tuple[str, str]],
                 magnitude_col: str = 'mag_v2',
                 z_spec_col: str = 'z_spec',
                 refit_on: str = 'pool',
                 n_sigma: float = 2.0,
                 red_color_index: int = 0,
                 red_color_switches: Optional[List[Tuple[float, int]]] = None,
                 n_components_first: int = 4,
                 n_components_rest: int = 2,
                 z_bin_width: float = 0.03,
                 z_min: Optional[float] = None,
                 z_max: Optional[float] = None,
                 weight_stop: float = 0.98,
                 min_red_weight: float = 0.05,
                 mu_tol: float = 1e-3,
                 sigma_tol: float = 1e-3,
                 max_iter: int = 10,
                 min_in_slice: int = 50,
                 n_restarts: int = 1,
                 n_jobs: int = 1,
                 xd_init: str = 'bayesian',
                 random_state: int = 0):
        if xd_init not in ('bayesian', 'gaussian'):
            raise ValueError("xd_init must be 'bayesian' or 'gaussian'")
        if refit_on not in ('pool', 'selected'):
            raise ValueError("refit_on must be 'pool' or 'selected'")
        self.df = df.reset_index(drop=True).copy()
        self.color_definitions = color_definitions
        self.colour_names = [f"{a}{b}" for a, b in color_definitions]
        self.magnitude_col = magnitude_col
        self.z_spec_col = z_spec_col
        self.refit_on = refit_on
        self.xd_init = xd_init
        self.n_sigma = n_sigma
        self.red_color_index = red_color_index
        self.n_components_first = n_components_first
        self.n_components_rest = n_components_rest
        self.z_bin_width = z_bin_width
        self.z_min = z_min if z_min is not None else float(self.df[z_spec_col].min())
        self.z_max = z_max if z_max is not None else float(self.df[z_spec_col].max())
        self.weight_stop = weight_stop
        self.min_red_weight = min_red_weight
        self.mu_tol = mu_tol
        self.sigma_tol = sigma_tol
        self.max_iter = max_iter
        self.min_in_slice = min_in_slice
        self.n_restarts = n_restarts
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Photometry for the full candidate pool (computed once).
        self.colors, self.color_cov = create_color_arrays_and_covariance(
            self.df, color_definitions, magnitude_col)            # (N, K), (N, K, K)
        self.mag = self.df[magnitude_col].values
        mag_str = magnitude_col.replace('mag_', '')
        self.mag_err = self.df[f"mag_{mag_str}_err"].values
        self.z = self.df[z_spec_col].values

        # Which colour defines "reddest" as a function of z. The 4000A break
        # leaves a given colour baseline as z rises, so the discriminating colour
        # must follow it (g-r -> r-i -> i-z). Default schedule built by NAME so it
        # adapts to the basis; switch redshifts are where each baseline straddles
        # the break (~0.45, ~0.80). Pass red_color_switches to override; falls
        # back to the constant red_color_index when no named colour matches.
        if red_color_switches is None:
            name_to_idx = {n: i for i, n in enumerate(self.colour_names)}
            sched = [(z0, name_to_idx[nm]) for z0, nm in
                     [(0.0, 'gr'), (0.40, 'ri'), (0.80, 'iz')] if nm in name_to_idx]
            self.red_color_switches = sched or [(0.0, self.red_color_index)]
        else:
            self.red_color_switches = sorted(red_color_switches)
        self.history: List[Dict] = []

    def _red_feat(self, z_mid: float) -> int:
        """Feature index (inside X = [mag, colours...]) of the red colour at z_mid."""
        idx = self.red_color_switches[0][1]
        for z0, ci in self.red_color_switches:
            if z_mid >= z0:
                idx = ci
        return 1 + idx

    # ---- feature assembly -----------------------------------------------
    def _features(self, idx: np.ndarray):
        """Joint (mag, all colours) vector + block-diagonal covariance for ``idx``.

        ``mag_v2`` shares no band with the colours, so the covariance is block
        diagonal: a mag variance block plus the correlated colour block.
        """
        K = len(self.color_definitions)
        X = np.column_stack([self.mag[idx], self.colors[idx]])        # (n, 1+K)
        Xerr = np.zeros((len(idx), 1 + K, 1 + K))
        Xerr[:, 0, 0] = self.mag_err[idx] ** 2
        Xerr[:, 1:, 1:] = self.color_cov[idx]
        return X, Xerr

    def _fit_best(self, X, Xerr, ncomp, init):
        """Fit XDGMM, keeping the highest-logL solution over ``n_restarts`` seeds.

        Warm-started fits (``init`` given) are deterministic, so a single fit is
        run regardless of ``n_restarts``. Otherwise the first fit of a slice is
        restarted from ``n_restarts`` distinct seeds and the best by log-likelihood
        is returned — the guard against multi-modal EM local optima.
        """
        im = self.xd_init
        if init is not None or self.n_restarts <= 1:
            return XDGMM(ncomp, max_iter=400, random_state=self.random_state,
                         init_method=im).fit(X, Xerr, init=init)

        seeds = [self.random_state + r for r in range(self.n_restarts)]
        if self.n_jobs == 1:
            results = [_fit_one_seed(X, Xerr, ncomp, s, im) for s in seeds]
        else:
            from joblib import Parallel, delayed
            try:
                from threadpoolctl import threadpool_limits
                limiter = threadpool_limits(limits=1, user_api="blas")
            except ImportError:
                limiter = None
            try:
                results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed(_fit_one_seed)(X, Xerr, ncomp, s, im) for s in seeds)
            finally:
                if limiter is not None:
                    limiter.unregister()

        # Keep the highest-logL fit; rehydrate an XDGMM with its parameters.
        best_logL, mu, V, alpha = max(results, key=lambda t: t[0])
        clf = XDGMM(ncomp, max_iter=400, random_state=self.random_state)
        clf.mu, clf.V, clf.alpha = mu, V, alpha
        return clf

    def _posterior_red(self, clf, red, X, Xerr) -> np.ndarray:
        """Posterior responsibility of the red component for each point."""
        logpa = clf.logprob_a(X, Xerr)                                # (n, ncomp)
        logpa -= logpa.max(axis=1, keepdims=True)
        q = np.exp(logpa)
        q /= q.sum(axis=1, keepdims=True)
        return q[:, red]

    # ---- per-slice loop --------------------------------------------------
    def _select_slice(self, slice_idx: np.ndarray, z_mid: float,
                      verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Iterate one z-slice. Returns (p_red, keep_2sigma) over ``slice_idx``.

        ``p_red`` is the posterior red responsibility from the converged fit
        (evaluated for every galaxy in the slice); ``keep_2sigma`` is the hard
        n_sigma selection. Galaxies dropped from the fit pool but never re-scored
        retain their last posterior.
        """
        n_slice = len(slice_idx)
        Xfull, Xerrfull = self._features(slice_idx)
        red_feat = self._red_feat(z_mid)             # z-appropriate red colour
        red_cname = self.colour_names[red_feat - 1]

        active = np.ones(n_slice, dtype=bool)        # in the current fit pool
        p_red = np.zeros(n_slice)
        keep = np.zeros(n_slice, dtype=bool)
        d2 = np.full(n_slice, np.inf)                # 5D Mahalanobis^2 to red comp
        p_red_c = np.zeros(n_slice)                  # colour-only posterior
        d2_c = np.full(n_slice, np.inf)              # colour-only Mahalanobis^2
        prev_mu = prev_sigma = None
        prev_params = None                           # (mu, V, alpha) for warm-start

        for it in range(1, self.max_iter + 1):
            a_idx = np.where(active)[0]
            if self.refit_on == 'pool':
                # FULL pool every iteration; component count fixed so the prior
                # red component can be warm-started (blue stays as anchor).
                ncomp = self.n_components_first
            else:
                ncomp = self.n_components_first if it == 1 else self.n_components_rest
            ncomp = min(ncomp, max(1, len(a_idx) // 10))
            if len(a_idx) < max(self.min_in_slice, ncomp * 5):
                break

            Xa, Xerra = Xfull[a_idx], Xerrfull[a_idx]
            # Warm-start only in pool mode, only when the component count matches
            # the carried params (so identity is preserved across iterations).
            init = (prev_params if (self.refit_on == 'pool' and prev_params is not None
                                    and prev_params[0].shape[0] == ncomp) else None)
            try:
                clf = self._fit_best(Xa, Xerra, ncomp, init)
            except Exception:
                break

            # Reddest component AMONG those with non-trivial weight. Without this
            # floor, best-by-logL adds a tiny (w~0.01) very broad catch-all to
            # absorb outliers, whose mean colour is the highest of all — the
            # plain argmax then mistakes that background for the red sequence and
            # its huge 2-sigma ellipse sweeps in most of the slice (observed at
            # z=0.12-0.22). Restricting to alpha > min_red_weight excludes it.
            elig = np.where(clf.alpha > self.min_red_weight)[0]
            if len(elig) == 0:                       # degenerate; fall back
                elig = np.arange(clf.mu.shape[0])
            red = int(elig[np.argmax(clf.mu[elig, red_feat])])
            red_mu = float(clf.mu[red, red_feat])
            # XD covariance is deconvolved -> diagonal is intrinsic scatter^2.
            red_sigma = float(np.sqrt(max(clf.V[red, red_feat, red_feat], 0.0)))
            red_w = float(clf.alpha[red])

            # Score and select the WHOLE slice from this fit.
            p_red = self._posterior_red(clf, red, Xfull, Xerrfull)
            d2 = _mahalanobis_d2(Xfull, Xerrfull, clf.mu[red], clf.V[red])
            crit = chi2.ppf(_sigma_to_conf(self.n_sigma), df=Xfull.shape[1])
            keep = d2 < crit

            # Colour-only (magnitude marginalised) membership: the colour
            # sub-block of the red component. A Gaussian mixture's marginal over
            # the dropped axis is just the retained sub-blocks (same weights), so
            # this is the proper "is it red in colour, regardless of magnitude"
            # cut — it leaves the magnitude lever arm (for b(z)) untouched.
            clf_c = XDGMM(clf.mu.shape[0])
            clf_c.mu, clf_c.V, clf_c.alpha = clf.mu[:, 1:], clf.V[:, 1:, 1:], clf.alpha
            Xc, Xcerr = Xfull[:, 1:], Xerrfull[:, 1:, 1:]
            p_red_c = self._posterior_red(clf_c, red, Xc, Xcerr)
            d2_c = _mahalanobis_d2(Xc, Xcerr, clf_c.mu[red], clf_c.V[red])

            d_mu = abs(red_mu - prev_mu) if prev_mu is not None else np.inf
            d_sig = (abs(red_sigma - prev_sigma) / max(prev_sigma, 1e-9)
                     if prev_sigma is not None else np.inf)
            self.history.append({
                'z_mid': z_mid, 'iter': it, 'refit_on': self.refit_on,
                'n_active': int(active.sum()), 'n_selected': int(keep.sum()),
                'ncomp': ncomp, 'red_w': red_w, 'red_color': red_cname,
                'red_mu': red_mu, 'red_sigma': red_sigma,
                'd_mu': d_mu, 'd_sigma': d_sig,
            })
            if verbose:
                tag = '  (sigma shrinking)' if (prev_sigma is not None
                                                and red_sigma < prev_sigma - 1e-6) else ''
                print(f"    z={z_mid:.3f} it{it} ncomp={ncomp} "
                      f"n_act={active.sum()} n_sel={keep.sum()} "
                      f"w={red_w:.3f} mu={red_mu:.4f} sig={red_sigma:.4f} "
                      f"Δmu={d_mu:.4f} Δsig={d_sig:.4f}{tag}")

            # Stopping criteria.
            if red_w > self.weight_stop:
                break
            if d_mu < self.mu_tol and d_sig < self.sigma_tol:
                break

            # Advance the pool.
            if self.refit_on == 'selected':
                # Peel: next fit sees only the 2-sigma reds (no blue anchor).
                new_active = keep.copy()
                if new_active.sum() == active.sum() and np.array_equal(new_active, active):
                    break
                active = new_active
            else:
                # Pool: the fit ALWAYS sees the whole slice; only the warm-start
                # (the carried red component) refines between iterations. Blue is
                # never removed, so the boundary can stabilise without the red
                # scatter collapsing.
                prev_params = (clf.mu.copy(), clf.V.copy(), clf.alpha.copy())
            prev_mu, prev_sigma = red_mu, red_sigma

        return p_red, keep, d2, p_red_c, d2_c

    # ---- driver ----------------------------------------------------------
    def select(self, verbose: bool = True) -> pd.DataFrame:
        """Run the per-slice iteration over the whole catalogue.

        Returns the parent frame with new columns: ``p_red`` (posterior red
        responsibility from each slice's converged fit), ``red_2sigma`` (the hard
        n_sigma membership), ``red_maha_d2`` (squared Mahalanobis distance to the
        red component) and ``red_maha_df`` (its dof = n_features), so the sample
        can be re-thresholded to any sigma without re-fitting. Rows outside any
        processed slice get p_red=0.
        """
        p_red = np.zeros(len(self.df))
        keep = np.zeros(len(self.df), dtype=bool)
        d2 = np.full(len(self.df), np.inf)
        p_red_c = np.zeros(len(self.df))
        d2_c = np.full(len(self.df), np.inf)

        n_steps = int(np.ceil((self.z_max - self.z_min) / self.z_bin_width - 1e-9))
        edges = self.z_min + np.arange(n_steps + 1) * self.z_bin_width
        if verbose:
            print(f"IterativeRedSelector(refit_on={self.refit_on!r}): "
                  f"{len(edges) - 1} slices, Δz={self.z_bin_width}, "
                  f"range [{self.z_min:.3f}, {self.z_max:.3f}]")

        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = np.where((self.z >= lo) & (self.z < hi))[0]
            if len(idx) < self.min_in_slice:
                continue
            pr, kp, dd, prc, ddc = self._select_slice(idx, 0.5 * (lo + hi), verbose)
            p_red[idx] = pr
            keep[idx] = kp
            d2[idx] = dd
            p_red_c[idx] = prc
            d2_c[idx] = ddc

        out = self.df.copy()
        out['p_red'] = p_red
        out['red_2sigma'] = keep
        out['red_maha_d2'] = d2
        out['red_maha_df'] = 1 + len(self.color_definitions)
        # Colour-only (magnitude-marginalised) membership — preserves the b(z)
        # magnitude lever arm; df = n_colours.
        out['p_red_colour'] = p_red_c
        out['red_maha_d2_colour'] = d2_c
        out['red_maha_df_colour'] = len(self.color_definitions)
        if verbose:
            print(f"  total {self.n_sigma}-sigma red: {keep.sum()} / {len(out)}")
        return out
