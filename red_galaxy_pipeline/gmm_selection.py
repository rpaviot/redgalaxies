"""
Gaussian Mixture Model (GMM) selection for red sequence galaxies.

Implements the Extreme Deconvolution (XD) methodology from Vakili et al. (2019),
Section 3.2, for identifying red sequence galaxies using a two-stage GMM approach.
"""

import numpy as np
import pandas as pd
from time import time
from scipy import linalg
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.utils import check_random_state
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from red_galaxy_pipeline.utils import create_color_arrays_and_covariance

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp


def log_multivariate_gaussian(x, mu, V, Vinv=None, method=1, fast=False):
    """
    Evaluate log N(x|mu, V) with array broadcasting.

    Parameters
    ----------
    x : array_like
        Input data points
    mu : array_like
        Mean of the Gaussian distribution
    V : array_like
        Covariance matrix of the Gaussian distribution
    Vinv : array_like, optional
        Pre-computed inverse of V (default: None)
    method : int, optional
        Computation method: 0 for Cholesky, 1 for direct inverse (default: 1)

    Returns
    -------
    log_prob : ndarray
        Log probability of x under the Gaussian N(mu, V)
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)

    ndim = x.shape[-1]
    x_mu = x - mu

    if V.shape[-2:] != (ndim, ndim):
        raise ValueError("Shape of (x-mu) and V do not match")

    Vshape = V.shape
    V = V.reshape([-1, ndim, ndim])

    if Vinv is not None:
        assert Vinv.shape == Vshape
        method = 1

    if method == 0:
        Vchol = np.array([linalg.cholesky(V[i], lower=True)
                          for i in range(V.shape[0])])
        VcholI = np.array([linalg.inv(Vchol[i])
                          for i in range(V.shape[0])])
        logdet = np.array([2 * np.sum(np.log(np.diagonal(Vchol[i])))
                           for i in range(V.shape[0])])

        VcholI = VcholI.reshape(Vshape)
        logdet = logdet.reshape(Vshape[:-2])

        VcIx = np.sum(VcholI * x_mu.reshape(x_mu.shape[:-1]
                                            + (1,) + x_mu.shape[-1:]), -1)
        xVIx = np.sum(VcIx ** 2, -1)

    elif method == 1:
        if fast:
            # Batched numpy: one BLAS call instead of n*K Python-level calls.
            V_batched = V  # shape (M, d, d) where M = prod(Vshape[:-2])
            if Vinv is None:
                Vinv = np.linalg.inv(V_batched).reshape(Vshape)
            else:
                assert Vinv.shape == Vshape
            _, logdet = np.linalg.slogdet(V_batched)
            logdet = logdet.reshape(Vshape[:-2])
        else:
            if Vinv is None:
                Vinv = np.array([linalg.inv(V[i])
                                 for i in range(V.shape[0])]).reshape(Vshape)
            else:
                assert Vinv.shape == Vshape

            logdet = np.log(np.array([linalg.det(V[i])
                                      for i in range(V.shape[0])]))
            logdet = logdet.reshape(Vshape[:-2])

        xVI = np.sum(x_mu.reshape(x_mu.shape + (1,)) * Vinv, -2)
        xVIx = np.sum(xVI * x_mu, -1)

    else:
        raise ValueError(f"unrecognized method {method}")

    return -0.5 * ndim * np.log(2 * np.pi) - 0.5 * (logdet + xVIx)


class XDGMM(BaseEstimator):
    """
    Extreme Deconvolution Gaussian Mixture Model.

    Fit an XD model to heteroscedastic data (different error per point).
    This implementation follows Bovy et al. (2011) arXiv:0905.2979.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components
    max_iter : int, optional
        Number of EM iterations (default: 100)
    tol : float, optional
        Stopping criterion for EM iterations (default: 1e-5)
    verbose : bool, optional
        Print iteration information
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(self, n_components, max_iter=100, tol=1e-5, verbose=False,
                 random_state=0, fast=True, init_method='bayesian'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.fast = fast
        # How EM is seeded: 'bayesian' = BayesianGaussianMixture (Dirichlet-process
        # prior; can shrink/prune weak components), 'gaussian' = plain
        # GaussianMixture (vanilla EM, no component pruning). The seed determines
        # which local optimum the XD EM converges to.
        if init_method not in ('bayesian', 'gaussian'):
            raise ValueError("init_method must be 'bayesian' or 'gaussian'")
        self.init_method = init_method

        # Model parameters: set by fit() method
        self.V = None
        self.mu = None
        self.alpha = None

    def fit(self, X, Xerr, R=None, init=None):
        """
        Fit the XD model to data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Input data
        Xerr : array_like, shape (n_samples, n_features, n_features)
            Error covariance matrices for each data point
        R : array_like, optional
            Transformation matrix (not implemented)
        init : tuple (mu, V, alpha), optional
            Warm-start parameters. When given, EM is seeded from these instead of
            the default k-means BayesianGaussianMixture init — used to carry a
            component identity (e.g. the red component) across successive fits on
            the same data so iterations refine rather than re-discover.

        Returns
        -------
        self : object
            Fitted model
        """
        if R is not None:
            raise NotImplementedError("mixing matrix R is not yet implemented")

        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape

        # Assume full covariances
        assert Xerr.shape == (n_samples, n_features, n_features)

        if init is not None:
            mu0, V0, alpha0 = init
            self.mu = np.asarray(mu0, dtype=float).copy()
            self.V = np.asarray(V0, dtype=float).copy()
            self.alpha = np.asarray(alpha0, dtype=float).copy()
        else:
            # Seed EM with a plain-data mixture fit (no errors, but fast). The
            # init_method picks the seeding model; both feed the same XD EM below.
            common = dict(n_components=self.n_components, max_iter=500,
                          covariance_type='full', init_params='kmeans',
                          random_state=self.random_state, tol=self.tol * 100)
            if self.init_method == 'gaussian':
                gmm = GaussianMixture(**common).fit(X)
            else:
                gmm = BayesianGaussianMixture(**common).fit(X)

            self.mu = gmm.means_
            self.alpha = gmm.weights_
            self.V = gmm.covariances_

        logL = self.logL(X, Xerr)

        for i in range(self.max_iter):
            t0 = time()
            self._EMstep(X, Xerr)
            logL_next = self.logL(X, Xerr)
            t1 = time()

            if self.verbose:
                print(f"{i + 1}: log(L) = {logL_next:.5g}")
                print(f"    ({t1 - t0:.2g} sec)")

            if logL_next < logL + self.tol:
                break
            logL = logL_next

        return self

    def logprob_a(self, X, Xerr):
        """
        Evaluate log probability for each component.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Data points
        Xerr : array_like, shape (n_samples, n_features, n_features)
            Error covariances

        Returns
        -------
        log_prob : ndarray, shape (n_samples, n_components)
            Log probability of each point under each component
        """
        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape

        assert Xerr.shape == (n_samples, n_features, n_features)

        X = X[:, np.newaxis, :]
        Xerr = Xerr[:, np.newaxis, :, :]
        T = Xerr + self.V

        return log_multivariate_gaussian(X, self.mu, T, fast=self.fast) + np.log(self.alpha)

    def logL(self, X, Xerr):
        """
        Compute log-likelihood of data given the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Data
        Xerr : array_like, shape (n_samples, n_features, n_features)
            Error covariances

        Returns
        -------
        logL : float
            Log-likelihood
        """
        return np.sum(logsumexp(self.logprob_a(X, Xerr), -1))

    def _EMstep(self, X, Xerr):
        """
        Perform one E-M step (Eq. 16 of Bovy et al. 2011).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Input data
        Xerr : array_like, shape (n_samples, n_features, n_features)
            Error covariance matrices for each data point
        """
        n_samples, n_features = X.shape

        X = X[:, np.newaxis, :]
        Xerr = Xerr[:, np.newaxis, :, :]

        w_m = X - self.mu
        T = Xerr + self.V

        # Compute inverse of each covariance matrix T
        Tshape = T.shape
        if self.fast:
            # Batched: one BLAS call for all n*K matrices.
            Tinv = np.linalg.inv(T)
        else:
            T = T.reshape([n_samples * self.n_components, n_features, n_features])
            Tinv = np.array([linalg.inv(T[i]) for i in range(T.shape[0])]).reshape(Tshape)
            T = T.reshape(Tshape)

        # E-step: compute responsibilities q via logsumexp for numerical stability.
        # The previous direct form `q = N*alpha / (N @ alpha)` underflowed to 0/0 = NaN
        # for points where exp(logL) underflowed under both components (poisoned
        # mu/V on subsequent iterations and silently killed stage 2 on some bins).
        log_N = log_multivariate_gaussian(X, self.mu, T, Vinv=Tinv, fast=self.fast)
        log_q = log_N + np.log(self.alpha)
        log_q -= logsumexp(log_q, axis=-1, keepdims=True)
        q = np.exp(log_q)

        tmp = np.sum(Tinv * w_m[:, :, np.newaxis, :], -1)
        b = self.mu + np.sum(self.V * tmp[:, :, np.newaxis, :], -1)

        tmp = np.sum(Tinv[:, :, :, :, np.newaxis] * self.V[:, np.newaxis, :, :], -2)
        B = self.V - np.sum(self.V[:, :, :, np.newaxis] * tmp[:, :, np.newaxis, :, :], -2)

        # M-step: compute alpha, m, V
        qj = q.sum(0)
        self.alpha = qj / n_samples
        self.mu = np.sum(q[:, :, np.newaxis] * b, 0) / qj[:, np.newaxis]

        m_b = self.mu - b
        tmp = m_b[:, :, np.newaxis, :] * m_b[:, :, :, np.newaxis]
        tmp += B
        tmp *= q[:, :, np.newaxis, np.newaxis]
        self.V = tmp.sum(0) / qj[:, np.newaxis, np.newaxis]


def _run_bin_task(task, magnitude_col, z_spec_col, sigma_multiplier,
                  n_realizations=1, base_seed=0,
                  n_components_stage1=4, stage2_single_gaussian=False,
                  stage2_degeneracy_eps=0.0,
                  stage2_degeneracy_mode="reject",
                  stage1_chisq_sigma=0.866,
                  stage2_chisq_sigma=0.866,
                  pi_min=0.05,
                  stage2_component="reddest",
                  stage1_mode="ridgeline",
                  stage2_on_parent=False):
    """Run two-stage XD GMM on one z-bin, N times, returning soft membership p_red.

    stage2_on_parent=True: the stage-1 cut only picks the galaxies the stage-2
    XD is FITTED on; the stage-2 ellipsoid is then applied to every galaxy of
    the bin, so the stage-1 box is not part of the selection (the step-2
    truncation term is then the stage-2 taper alone, no box, no leak).

    For n_realizations=1 (default), behaviour matches the original hard
    selection: returned DataFrame is the selected red galaxies with p_red=1.

    For n_realizations>1, each row of df_bin is included if it was selected
    in at least one realization, with p_red = (#selected) / (#successful runs).
    """
    z_low, z_high, z_mid, df_bin, colors_bin = task

    # Track selection counts on the bin's index (preserved through filtering).
    selection_count = pd.Series(0, index=df_bin.index, dtype=int)
    stage1_count = pd.Series(0, index=df_bin.index, dtype=int)
    m_refs = []
    n_success = 0
    bin_fit = None

    for k in range(n_realizations):
        selector_bin = RedSequenceSelector(
            df_bin,
            magnitude_col,
            colors_bin,
            z_spec_col=z_spec_col,
            sigma_multiplier=sigma_multiplier,
            use_binning=False,
            n_jobs=1,
            random_state=base_seed + k,
            n_components_stage1=n_components_stage1,
            stage2_single_gaussian=stage2_single_gaussian,
            stage2_degeneracy_eps=stage2_degeneracy_eps,
            stage2_degeneracy_mode=stage2_degeneracy_mode,
            stage1_chisq_sigma=stage1_chisq_sigma,
            stage2_chisq_sigma=stage2_chisq_sigma,
            pi_min=pi_min,
            stage2_component=stage2_component,
            stage1_mode=stage1_mode,
        )
        try:
            selector_bin._gmm_stage1(verbose=False)
            selector_bin._gmm_stage2(verbose=False)
        except Exception:
            continue
        if len(selector_bin.df) == 0:
            continue
        selected_idx = selector_bin.df.index
        if stage2_on_parent:
            fit2 = selector_bin.stage2_fit
            if fit2 is None:
                raise RuntimeError("stage2_on_parent needs the 2-component stage-2 fit "
                                   "(stage2_single_gaussian / degeneracy fallback store no window)")
            Xp, Cp = create_color_arrays_and_covariance(df_bin, colors_bin, magnitude_col)
            r = Xp - np.asarray(fit2["mu"], float)
            G = np.asarray(fit2["V"], float)[None] + Cp
            good = np.isfinite(r).all(axis=1) & np.isfinite(G).all(axis=(1, 2))
            D2 = np.full(len(df_bin), np.inf)
            D2[good] = np.einsum("ni,ni->n", r[good],
                                 np.linalg.solve(G[good], r[good][..., None])[..., 0])
            selected_idx = df_bin.index[D2 < fit2["chi2_crit"]]
            fit2["n_pass_parent"] = int(len(selected_idx))
        selection_count.loc[selected_idx] += 1
        if selector_bin.stage1_pass_index is not None:
            stage1_count.loc[selector_bin.stage1_pass_index] += 1
        m_refs.append(np.median(selector_bin.df[magnitude_col].values))
        n_success += 1
        # Keep the LAST successful realisation's windows (exact for
        # n_realizations=1, representative otherwise).
        bin_fit = {"z_low": float(z_low), "z_high": float(z_high), "z_mid": float(z_mid),
                   "colors": [f"{a}{b}" for a, b in colors_bin],
                   "seed": int(base_seed + k), "n_realizations_ok": n_success,
                   "stage2_on_parent": bool(stage2_on_parent),
                   "stage1": selector_bin.stage1_fit, "stage2": selector_bin.stage2_fit}

    if n_success == 0:
        return pd.DataFrame(), np.nan, None

    out = df_bin.copy()
    out["p_red"] = selection_count.values / float(n_success)
    out["p_stage1"] = stage1_count.values / float(n_success)
    # Keep stage-1 passers that stage 2 rejected too (p_red=0, p_stage1>0):
    # downstream reads p_red >= p_red_min so they are inert there, and the
    # step-1 diagnostics need them to see what stage 2 removes.
    out = out[(out["p_red"] > 0) | (out["p_stage1"] > 0)].copy()
    return out, float(np.median(m_refs)), bin_fit


class RedSequenceSelector:
    """
    Two-stage GMM red sequence selection (Vakili et al. 2019 Section 3.2).

    Parameters
    ----------
    df : DataFrame
        Input galaxy catalog
    magnitude_col : str
        Name of the reference magnitude column
    color_definitions : list of tuples
        List of (band1, band2) tuples defining colors as band1 - band2
    z_spec_col : str, optional
        Name of the spectroscopic redshift column (default: 'z_spec')
    sigma_multiplier : float, optional
        Sigma multiplier for chi-squared filtering (default: 2.0)
    z_bin_width : float, optional
        Width of redshift bins (default: 0.03)
    color_switch_z : float, optional
        Redshift at which to switch color order (default: 0.5)
    color_switch_z2 : float, optional
        Second switch: at z_low >= color_switch_z2 the colour with index
        ``high_z_first`` in ``color_definitions`` (default 2 = i-z in the
        gr/ri/iz/ug basis) becomes the stage-1 reference colour, with the
        previous reference colours following it (i-z, r-i, g-r, u-g).
        None (default) keeps the single switch.
    high_z_first : int, optional
        Index (in ``color_definitions``) of the colour put first above
        ``color_switch_z2`` (default: 2).
    drop_u_z : float, optional
        Bins with z_low >= drop_u_z leave the u-g colour out entirely (stage 2
        becomes 3-D and u NaN rows survive the per-bin NaN drop). Above z~0.4
        u-g carries almost no redshift information while the u DETECTION
        requirement removes 6% (z=0.40) to ~40% (z>0.8) of red spec-z galaxies.
        None (default) keeps u-g in every bin.
    use_binning : bool, optional
        Whether to use redshift binning (default: True)
    stage2_component : {"reddest", "max_weight"}, optional
        Which stage-2 XD component is the red sequence: the one with the
        reddest first-colour mean among components with weight > pi_min
        (default, historical behaviour) or the one with the largest weight.
    """

    def __init__(self, df, magnitude_col, color_definitions,
                 z_spec_col='z_spec', sigma_multiplier=2.0,
                 z_bin_width=0.03, color_switch_z=0.42, use_binning=True,
                 color_switch_z2=None, high_z_first=2, drop_u_z=None,
                 z_min=None, z_max=None, n_jobs=1, n_realizations=1,
                 random_state=None,
                 n_components_stage1=4, stage2_single_gaussian=False,
                 stage2_degeneracy_eps=0.0,
                 stage2_degeneracy_mode="reject",
                 stage1_chisq_sigma=0.866,
                 stage2_chisq_sigma=0.866,
                 pi_min=0.05,
                 stage2_component="reddest",
                 z_step=None,
                 stage1_mode="ridgeline",
                 red_jump_max=None,
                 stage2_on_parent=False):
        self.df = df.copy()
        # the stage-2 ellipsoid cuts the whole bin; stage 1 only picks its fit sample
        self.stage2_on_parent = stage2_on_parent
        self.pi_min = pi_min
        if stage2_component not in ("reddest", "max_weight"):
            raise ValueError(f"stage2_component must be 'reddest' or 'max_weight', got {stage2_component!r}")
        self.stage2_component = stage2_component
        self.z_step = z_step
        self.stage1_chisq_sigma = stage1_chisq_sigma
        self.stage2_chisq_sigma = stage2_chisq_sigma
        if stage1_mode not in ("ridgeline", "component"):
            raise ValueError(f"stage1_mode must be 'ridgeline' or 'component', "
                             f"got {stage1_mode!r}")
        self.stage1_mode = stage1_mode
        # disjoint mode: reject a bin whose red stage-2 mean in the reference
        # colour jumps by more than this from the previous accepted bin (None = off)
        self.red_jump_max = red_jump_max
        self.magnitude_col = magnitude_col
        self.color_definitions = color_definitions
        self.z_spec_col = z_spec_col
        self.sigma_multiplier = sigma_multiplier
        self.n_colors = len(color_definitions)
        self.z_bin_width = z_bin_width
        self.color_switch_z = color_switch_z
        self.color_switch_z2 = color_switch_z2
        self.high_z_first = high_z_first
        self.drop_u_z = drop_u_z
        self.use_binning = use_binning
        self.z_min = z_min
        self.z_max = z_max
        self.n_jobs = n_jobs
        self.n_realizations = n_realizations
        self.random_state = random_state
        self.n_components_stage1 = n_components_stage1
        self.stage2_single_gaussian = stage2_single_gaussian
        self.stage2_degeneracy_eps = stage2_degeneracy_eps
        self.stage2_degeneracy_mode = stage2_degeneracy_mode

        # Extract magnitude name (e.g., 'v2' from 'mag_v2')
        self.mag_str = magnitude_col.replace('mag_', '')

        # Storage for per-bin reference magnitudes
        self.m_ref_per_bin = []
        # Fitted selection windows, filled by the stage methods (component-mode
        # stage 1 and stage 2). Exported per bin by _select_with_binning as
        # self.bin_fits so the step-1 cut is known analytically downstream
        # (truncation half-width t^2 = chi2_crit * (V + C_i)) and can be
        # overlaid on red-vs-parent colour diagnostics.
        self.stage1_fit = None
        self.stage2_fit = None
        self.stage1_pass_index = None

        # Preprocess and build covariance matrices (if not using binning)
        if not use_binning:
            self._build_data_vectors()

    def _build_data_vectors(self):
        """
        Build color vectors and covariance matrices.

        Constructs the data arrays needed for GMM fitting, including
        color arrays, covariance matrices, and combined data vectors.
        """
        # Get reference magnitude
        self.mag_ref = self.df[self.magnitude_col].values

        # Get color array and full covariance
        self.color_array, self.color_covariance = create_color_arrays_and_covariance(
            self.df, self.color_definitions, self.magnitude_col
        )

        # Build magnitude-color covariance (for stage 1)
        self._build_magc_covariance()

        # Build combined data vector [mag, colors]
        self.data_vector = np.column_stack([self.mag_ref, self.color_array])

        # Preprocess: remove NaN and outliers
        self._preprocess_data()

    def _build_magc_covariance(self):
        """
        Build 2D covariance matrix for (magnitude, first_color) space.

        Constructs per-galaxy covariance matrices for Stage 1 GMM,
        including correlation between magnitude and color errors.
        """
        n_galaxies = len(self.df)
        ref_mag_err_col = f'mag_{self.mag_str}_err'
        mag_ref_err = self.df[ref_mag_err_col].values

        band1, band2 = self.color_definitions[0]  # First color
        mag1_err_col = f'mag_{band1}_err'
        mag2_err_col = f'mag_{band2}_err'

        self.magc_covariance = np.zeros((n_galaxies, 2, 2))
        self.magc_covariance[:, 0, 0] = mag_ref_err**2
        self.magc_covariance[:, 1, 1] = self.df[mag1_err_col]**2 + self.df[mag2_err_col]**2

        # Off-diagonal: check if color shares band with reference magnitude
        common_bands = set(self.mag_str) & set([band1, band2])
        if common_bands:
            common_band = list(common_bands)[0]
            common_err_col = f'mag_{common_band}_err'
            # Note: original code has flux ratio term - simplified here
            self.magc_covariance[:, 0, 1] = self.df[common_err_col]**2
            self.magc_covariance[:, 1, 0] = self.df[common_err_col]**2

    def _apply_mask(self, mask):
        """
        Apply boolean mask to all internal data arrays.

        Parameters
        ----------
        mask : array_like of bool
            Boolean mask to apply
        """
        self.df = self.df[mask]
        self.data_vector = self.data_vector[mask]
        self.color_array = self.color_array[mask]
        self.color_covariance = self.color_covariance[mask]
        self.magc_covariance = self.magc_covariance[mask]
        self.mag_ref = self.mag_ref[mask]

    def _preprocess_data(self):
        """
        Remove NaN values and outliers using DBSCAN.

        Applies NaN filtering and DBSCAN clustering in (mag, color) space
        to remove outliers before GMM fitting.
        """
        nan_mask = ~np.isnan(self.data_vector).any(axis=1)
        self._apply_mask(nan_mask)

        # Remove outliers using DBSCAN in (mag, first_color) space
        X = self.data_vector[:, 0:2]
        min_samples = 10
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
        distances, _ = nbrs.kneighbors(X)

        max_dist = np.quantile(distances, 0.98)
        clustering = DBSCAN(eps=max_dist, min_samples=10, p=1, algorithm='kd_tree').fit(X)
        self._apply_mask(clustering.labels_ != -1)

    def select_red_sequence(self, verbose=False):
        """
        Execute two-stage red sequence selection.

        Parameters
        ----------
        verbose : bool, optional
            Print progress information (default: False)

        Returns
        -------
        df : DataFrame
            DataFrame containing selected red sequence galaxies
        """
        if self.use_binning:
            if self.z_step is not None and self.z_step < self.z_bin_width:
                return self._select_with_rolling_window(verbose=verbose)
            return self._select_with_binning(verbose=verbose)
        else:
            return self._select_no_binning(verbose=verbose)

    def _select_no_binning(self, verbose=False):
        """
        Run GMM on full sample without redshift binning.

        Parameters
        ----------
        verbose : bool, optional
            Print progress information (default: False)

        Returns
        -------
        df : DataFrame
            DataFrame containing selected red sequence galaxies
        """
        # Stage 1: GMM in (magnitude, first_color) space
        self._gmm_stage1(verbose=verbose)

        # Stage 2: GMM in multi-color space
        self._gmm_stage2(verbose=verbose)

        if verbose:
            print(f"Final red sequence sample: {len(self.df)} galaxies")

        return self.df

    def _select_with_binning(self, verbose=False):
        """
        Run GMM independently on redshift bins.

        Parameters
        ----------
        verbose : bool, optional
            Print progress information (default: False)

        Returns
        -------
        df : DataFrame
            DataFrame containing selected red sequence galaxies from all bins
        """
        # Create redshift bins. Span the configured [z_min, z_max] (already
        # padded by RedCatalogue); fall back to the data range if not given.
        # Use an explicit step count instead of np.arange's stop argument so
        # the last edge is guaranteed to be >= z_max (np.arange's float-stop
        # behaviour can drop or add the upper edge unpredictably).
        z_min = self.z_min if self.z_min is not None else self.df[self.z_spec_col].min()
        z_max = self.z_max if self.z_max is not None else self.df[self.z_spec_col].max()
        n_steps = int(np.ceil((z_max - z_min) / self.z_bin_width - 1e-9))
        bins_z = z_min + np.arange(n_steps + 1) * self.z_bin_width

        if verbose:
            print(f"Running GMM on {len(bins_z)-1} redshift bins (Δz={self.z_bin_width})")
            print(f"Redshift range: {z_min:.3f} to {z_max:.3f} "
                  f"(bin edges {bins_z[0]:.4f}..{bins_z[-1]:.4f})")

        # Bin the dataframe
        df_temp = self.df.copy()
        df_temp['z_bins'] = pd.cut(df_temp[self.z_spec_col], bins=bins_z)

        # Build per-bin tasks. Done here (not inside workers) so that the
        # color-switch logic stays sequential and trivially correct.
        tasks = []  # (z_low, z_high, z_mid, df_bin, colors_bin)
        for bin_label, group in df_temp.groupby('z_bins', observed=True):
            if len(group) == 0:
                continue
            z_low = bin_label.left
            z_mid = (bin_label.left + bin_label.right) / 2.0
            colors_bin = self.colors_for_bin(z_low)
            tasks.append((z_low, bin_label.right, z_mid,
                          group.drop('z_bins', axis=1), colors_bin))

        if verbose:
            for z_low, z_high, _, group, colors_bin in tasks:
                cstr = ', '.join([f"{c[0]}{c[1]}" for c in colors_bin])
                print(f"\n  Bin z=[{z_low:.3f}, {z_high:.3f}): "
                      f"{len(group)} galaxies, colors={cstr}")

        # Per-bin execution (sequential or joblib-parallel).
        base_seed = self.random_state if self.random_state is not None else 0
        if self.n_jobs == 1:
            results = [_run_bin_task(
                t, self.magnitude_col, self.z_spec_col, self.sigma_multiplier,
                n_realizations=self.n_realizations,
                base_seed=base_seed + 1000 * i,
                n_components_stage1=self.n_components_stage1,
                stage2_single_gaussian=self.stage2_single_gaussian,
                stage2_degeneracy_eps=self.stage2_degeneracy_eps,
                stage2_degeneracy_mode=self.stage2_degeneracy_mode,
                stage1_chisq_sigma=self.stage1_chisq_sigma,
                stage1_mode=self.stage1_mode,
                stage2_on_parent=self.stage2_on_parent,
                stage2_chisq_sigma=self.stage2_chisq_sigma,
                pi_min=self.pi_min,
                stage2_component=self.stage2_component,
            ) for i, t in enumerate(tasks)]
        else:
            from joblib import Parallel, delayed
            try:
                from threadpoolctl import threadpool_limits
                limiter = threadpool_limits(limits=1, user_api="blas")
            except ImportError:
                limiter = None
            try:
                results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed(_run_bin_task)(
                        t, self.magnitude_col, self.z_spec_col,
                        self.sigma_multiplier,
                        n_realizations=self.n_realizations,
                        base_seed=base_seed + 1000 * i,
                        n_components_stage1=self.n_components_stage1,
                        stage2_single_gaussian=self.stage2_single_gaussian,
                        stage2_degeneracy_eps=self.stage2_degeneracy_eps,
                        stage2_degeneracy_mode=self.stage2_degeneracy_mode,
                        stage1_chisq_sigma=self.stage1_chisq_sigma,
                        stage1_mode=self.stage1_mode,
                        stage2_on_parent=self.stage2_on_parent,
                        stage2_chisq_sigma=self.stage2_chisq_sigma,
                        pi_min=self.pi_min,
                        stage2_component=self.stage2_component,
                    ) for i, t in enumerate(tasks)
                )
            finally:
                if limiter is not None:
                    limiter.unregister()

        # Reassemble in order.
        df_red_list = []
        self.m_ref_per_bin = []
        self.bin_fits = []
        self.rejected_bins = []
        prev_mu2 = None          # stage-2 means (by colour name) of the last accepted bin
        for (_, _, z_mid, _, _), (df_bin_red, m_ref, bin_fit) in zip(tasks, results):
            # Continuity guard: when the red component is near pi_min (red
            # fraction ~5% at z>1.1) it can be masked and the reddest
            # REMAINING component -- the blue cloud -- taken as red (z2 bin
            # [1.18,1.20): r-i 0.78 -> 0.14, 13k blue galaxies flagged red).
            # Reject a bin whose reference-colour stage-2 mean jumps by more
            # than red_jump_max from the last accepted bin.
            if (self.red_jump_max is not None and bin_fit is not None
                    and bin_fit.get("stage2") and prev_mu2 is not None):
                ref = bin_fit["colors"][0]
                mu_ref = bin_fit["stage2"]["mu"][0]
                if ref in prev_mu2 and abs(mu_ref - prev_mu2[ref]) > self.red_jump_max:
                    self.rejected_bins.append({"z_low": bin_fit["z_low"], "ref": ref,
                                               "mu2_ref": float(mu_ref),
                                               "prev_mu2_ref": float(prev_mu2[ref])})
                    if verbose:
                        print(f"  z_mid={z_mid:.3f}: REJECTED by continuity guard "
                              f"({ref} stage-2 mean {mu_ref:.3f} vs {prev_mu2[ref]:.3f})")
                    continue
            if bin_fit is not None and bin_fit.get("stage2"):
                prev_mu2 = dict(zip(bin_fit["colors"], bin_fit["stage2"]["mu"]))
            if bin_fit is not None:
                bin_fit["m_ref"] = float(m_ref)
                bin_fit["n_red"] = int(len(df_bin_red))
                self.bin_fits.append(bin_fit)
            if len(df_bin_red) > 0:
                df_red_list.append(df_bin_red)
                self.m_ref_per_bin.append((z_mid, m_ref))
                if verbose:
                    print(f"  z_mid={z_mid:.3f}: selected {len(df_bin_red)} "
                          f"red galaxies, m_ref={m_ref:.2f}")

        # Concatenate all bins
        if len(df_red_list) == 0:
            if verbose:
                print("\nWARNING: No red galaxies selected in any bin!")
            return pd.DataFrame()

        self.df = pd.concat(df_red_list, ignore_index=True)

        if verbose:
            print(f"\nFinal red sequence sample: {len(self.df)} galaxies")

        return self.df

    def _select_with_rolling_window(self, verbose=False):
        """Rolling-window two-stage GMM selection.

        Windows of width ``z_bin_width`` step by ``z_step`` (< z_bin_width),
        so each galaxy participates in K = ceil(z_bin_width / z_step) windows
        (fewer near the edges). Per-galaxy p_red is averaged across all
        windows containing it: ``p_red = sum(p_red_window) / n_eligible``.

        This couples neighbouring bins via overlap and removes the
        per-bin centroid jitter that drives high-frequency wiggles in c(z).
        """
        z_min = self.z_min if self.z_min is not None else self.df[self.z_spec_col].min()
        z_max = self.z_max if self.z_max is not None else self.df[self.z_spec_col].max()
        w = float(self.z_bin_width)
        s = float(self.z_step)

        # Window starts so that windows tile [z_min, z_max] with overlap s.
        n_steps = int(np.ceil(max(z_max - z_min - w, 0.0) / s + 1e-9))
        starts = z_min + np.arange(n_steps + 1) * s
        # Ensure the last window reaches z_max exactly.
        if starts[-1] + w < z_max - 1e-9:
            starts = np.append(starts, z_max - w)

        # Use a unique integer index for safe accumulation.
        df_work = self.df.reset_index(drop=True)
        z_vals = df_work[self.z_spec_col].values

        n_eligible = np.zeros(len(df_work), dtype=np.int32)
        p_red_sum = np.zeros(len(df_work), dtype=np.float64)

        tasks = []
        for zlo in starts:
            zhi = zlo + w
            zmid = 0.5 * (zlo + zhi)
            in_win = (z_vals >= zlo) & (z_vals < zhi)
            if not in_win.any():
                continue
            group = df_work.loc[in_win]
            colors_bin = self.colors_for_bin(zlo)
            tasks.append((zlo, zhi, zmid, group, colors_bin))

        if verbose:
            print(f"Rolling-window GMM: width={w:.3f}, step={s:.3f}, "
                  f"{len(tasks)} windows, range [{z_min:.3f}, {z_max:.3f}]")

        base_seed = self.random_state if self.random_state is not None else 0
        if self.n_jobs == 1:
            results = [_run_bin_task(
                t, self.magnitude_col, self.z_spec_col, self.sigma_multiplier,
                n_realizations=self.n_realizations,
                base_seed=base_seed + 1000 * i,
                n_components_stage1=self.n_components_stage1,
                stage2_single_gaussian=self.stage2_single_gaussian,
                stage2_degeneracy_eps=self.stage2_degeneracy_eps,
                stage2_degeneracy_mode=self.stage2_degeneracy_mode,
                stage1_chisq_sigma=self.stage1_chisq_sigma,
                stage1_mode=self.stage1_mode,
                stage2_on_parent=self.stage2_on_parent,
                stage2_chisq_sigma=self.stage2_chisq_sigma,
                pi_min=self.pi_min,
                stage2_component=self.stage2_component,
            ) for i, t in enumerate(tasks)]
        else:
            from joblib import Parallel, delayed
            try:
                from threadpoolctl import threadpool_limits
                limiter = threadpool_limits(limits=1, user_api="blas")
            except ImportError:
                limiter = None
            try:
                results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed(_run_bin_task)(
                        t, self.magnitude_col, self.z_spec_col,
                        self.sigma_multiplier,
                        n_realizations=self.n_realizations,
                        base_seed=base_seed + 1000 * i,
                        n_components_stage1=self.n_components_stage1,
                        stage2_single_gaussian=self.stage2_single_gaussian,
                        stage2_degeneracy_eps=self.stage2_degeneracy_eps,
                        stage2_degeneracy_mode=self.stage2_degeneracy_mode,
                        stage1_chisq_sigma=self.stage1_chisq_sigma,
                        stage1_mode=self.stage1_mode,
                        stage2_on_parent=self.stage2_on_parent,
                        stage2_chisq_sigma=self.stage2_chisq_sigma,
                        pi_min=self.pi_min,
                        stage2_component=self.stage2_component,
                    ) for i, t in enumerate(tasks)
                )
            finally:
                if limiter is not None:
                    limiter.unregister()

        self.m_ref_per_bin = []
        self.bin_fits = [r[2] for r in results if r[2] is not None]
        for (zlo, zhi, zmid, group, _), (df_win, m_ref, _fit) in zip(tasks, results):
            elig_idx = group.index.values
            n_eligible[elig_idx] += 1
            if len(df_win) > 0:
                p_red_sum[df_win.index.values] += df_win["p_red"].values
                self.m_ref_per_bin.append((zmid, m_ref))

        with np.errstate(invalid='ignore', divide='ignore'):
            p_red = np.where(n_eligible > 0, p_red_sum / n_eligible, 0.0)

        out = df_work.copy()
        out["p_red"] = p_red
        out["n_eligible_windows"] = n_eligible
        out = out[out["p_red"] > 0].reset_index(drop=True)

        if verbose:
            print(f"  selected (p_red>0): {len(out)} / {len(df_work)}")
        self.df = out
        return out

    def _select_single_bin(self, df_bin, colors_bin, verbose=False):
        """
        Run two-stage GMM selection on a single redshift bin.

        Parameters
        ----------
        df_bin : DataFrame
            Galaxies in this redshift bin
        colors_bin : list of tuples
            Color definitions for this bin
        verbose : bool
            Print progress

        Returns
        -------
        df_red : DataFrame
            Selected red galaxies
        m_ref : float
            Reference magnitude for this bin
        """
        # Create temporary selector for this bin
        selector_bin = RedSequenceSelector(
            df_bin,
            self.magnitude_col,
            colors_bin,
            z_spec_col=self.z_spec_col,
            sigma_multiplier=self.sigma_multiplier,
            use_binning=False,  # Don't recurse!
            stage1_mode=self.stage1_mode,
        )

        # Run GMM stages
        try:
            selector_bin._gmm_stage1(verbose=False)
            selector_bin._gmm_stage2(verbose=False)

            # Get reference magnitude (median of selected galaxies)
            m_ref = np.median(selector_bin.df[self.magnitude_col].values)

            return selector_bin.df, m_ref

        except Exception as e:
            if verbose:
                print(f"    ⚠ GMM failed for this bin: {e}")
            return pd.DataFrame(), np.nan

    def colors_for_bin(self, z_low):
        """Colour order used in the bin starting at ``z_low``.

        Element 0 is the stage-1 reference colour (and the colour whose
        component mean identifies the red population in stage 2):
          z_low <  color_switch_z                : as given   (g-r first)
          color_switch_z <= z_low < switch_z2    : colour[1] first (r-i)
          z_low >= color_switch_z2 (if set)      : colour[high_z_first] first,
                                                   then r-i, g-r, the rest.
        Colours involving u are then dropped if z_low >= drop_u_z (if set).
        """
        defs = list(self.color_definitions)
        if self.color_switch_z2 is not None and z_low >= self.color_switch_z2 - 1e-9:
            k = self.high_z_first
            out = [defs[k]] + [d for i, d in enumerate(defs) if i != k and i in (0, 1)][::-1] \
                + [d for i, d in enumerate(defs) if i not in (0, 1, k)]
        elif z_low < self.color_switch_z:
            out = defs
        else:
            out = [defs[1], defs[0]] + defs[2:]
        if self.drop_u_z is not None and z_low >= self.drop_u_z - 1e-9:
            out = [d for d in out if 'u' not in d]
        return out

    def _gmm_stage1(self, verbose=False):
        """
        Stage 1: 2D GMM in (magnitude, color) space (Section 3.2 of Vakili+2019).

        Fits a 4-component XD GMM and selects galaxies belonging to the
        red sequence component based on chi-squared filtering.

        Parameters
        ----------
        verbose : bool, optional
            Print progress information (default: False)
        """
        if self.stage1_mode == "component":
            return self._gmm_stage1_component(verbose=verbose)

        X = self.data_vector[:, 0:2]
        Xerr = self.magc_covariance

        # Fit XD GMM (n_components configurable; default 4)
        clf = XDGMM(n_components=self.n_components_stage1, max_iter=2000,
                    tol=1e-10, verbose=False,
                    random_state=self.random_state)
        clf.fit(X, Xerr)

        # Select component with highest mean color (with minimum weight threshold)
        weights = clf.alpha
        pi_min = self.pi_min
        mean_color = np.where(weights > pi_min, clf.mu[:, 1], -np.inf)
        red_population = np.argmax(mean_color)

        frac_red = weights[red_population]
        if verbose:
            print(f"Stage 1: Red fraction = {frac_red:.2f}, mean colors = {mean_color}")

        # Apply chi-squared filtering based on red component
        self._filter_by_chisq_2d(clf, red_population, sigma=self.stage1_chisq_sigma)

    def _gmm_stage1_component(self, verbose=False):
        """Stage 1, redMaPPer flavour (Rykoff et al. 2014, Sect 6.2 step 1).

        Fits the GMM in the REFERENCE COLOUR ALONE and cuts about the
        component MEAN with the component's MARGINAL width:

            (c - mu_red)^2 / (V_red + C_phot)  <  chi2.ppf(sigma, df=1)

        Contrast `_gmm_stage1` + `_filter_by_chisq_2d`, which fit in
        (m, c) and cut about the fitted RIDGELINE with the CONDITIONAL width:

            (c - [c_ref + (V12/V11)(m - m_ref)])^2 / (S_mod^2 + C_phot) < ...
            S_mod^2 = V22 - V12^2/V11   <=   V22

        Two consequences, both deliberate. The cut no longer depends on a(z),
        b(z) or S_mod, so step 1 stops imprinting the quantities step 2 goes on
        to measure -- that is the whole point of this mode. And because the
        marginal width is never smaller than the conditional one, the cut is
        looser at fixed sigma, though it also loses the magnitude tilt, so at
        matched sigma the two select samples of very similar size (measured on
        the UNIONS z sample: 284,912 vs 303,598, agreeing on 89-97% of galaxies
        per z bin).

        The chi-squared threshold is IDENTICAL to ridgeline mode -- same
        `stage1_chisq_sigma`, same 1 dof -- so a sigma ladder means the same
        thing in both modes and the two are directly comparable.
        """
        c = self.data_vector[:, 1]
        cvar = self.magc_covariance[:, 1, 1]

        clf = XDGMM(n_components=self.n_components_stage1, max_iter=2000,
                    tol=1e-10, verbose=False,
                    random_state=self.random_state)
        clf.fit(c.reshape(-1, 1), cvar.reshape(-1, 1, 1))

        # Minimum-weight guard BEFORE the argmax: a collapsed EM component
        # (weight ~0, variance ~2) otherwise wins on its reddest mean and the
        # cut then keeps the whole bin.
        mean_color = np.where(clf.alpha > self.pi_min, clf.mu[:, 0], -np.inf)
        if not np.isfinite(mean_color).any():
            self._apply_mask(np.zeros(len(self.df), dtype=bool))
            if verbose:
                print("Stage 1 (component): no component above pi_min -- bin rejected")
            return
        red_population = int(np.argmax(mean_color))

        mu = float(clf.mu[red_population, 0])
        V_red = max(float(clf.V[red_population, 0, 0]), 1e-8)
        if verbose:
            print(f"Stage 1 (component): red fraction = "
                  f"{clf.alpha[red_population]:.2f}, mu = {mu:.3f}, "
                  f"sigma_int = {np.sqrt(V_red):.4f}")

        critical_value = chi2.ppf(self.stage1_chisq_sigma, df=1)
        chi2_vals = (c - mu) ** 2 / (V_red + cvar)
        mask = chi2_vals < critical_value
        self.stage1_fit = {
            "mode": "component", "mu": mu, "V": V_red,
            "pi": float(clf.alpha[red_population]), "chi2_crit": float(critical_value),
            "n_in": int(len(c)), "n_pass": int(mask.sum()),
            "components_mu": clf.mu[:, 0].tolist(), "components_V": clf.V[:, 0, 0].tolist(),
            "components_pi": clf.alpha.tolist(),
        }
        self.stage1_pass_index = self.df.index[mask]
        self._apply_mask(mask)

    def _filter_by_chisq_2d(self, clf, red_population, sigma):
        """
        Filter galaxies based on chi-squared distance in 2D (mag, color) space.

        Parameters
        ----------
        clf : XDGMM
            Fitted GMM model
        red_population : int
            Index of the red sequence component
        sigma : float
            Chi-squared confidence level for filtering
        """
        Cov = clf.V[red_population]
        mean = clf.mu[red_population]

        # Predict color from magnitude (Eq. 17 of Vakili+2019)
        mi_ref, color_ref = mean
        V_11, V_12 = Cov[0, 0], Cov[0, 1]
        color_mod = color_ref + (V_12 / V_11) * (self.data_vector[:, 0] - mi_ref)

        # Intrinsic scatter (Eq. 18)
        S_mod_sq = Cov[1, 1] - Cov[0, 1]**2 / Cov[0, 0]

        # Chi-squared filtering
        critical_value = chi2.ppf(sigma, df=1)
        chi2_vals = (color_mod - self.data_vector[:, 1])**2 / (S_mod_sq + self.magc_covariance[:, 1, 1])
        self._apply_mask(chi2_vals < critical_value)

    def _fit_single_gaussian(self, X, Xerr):
        """
        Fit single Gaussian to data with heteroscedastic errors.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Input data
        Xerr : array_like, shape (n_samples, n_features, n_features)
            Error covariance matrices for each data point

        Returns
        -------
        SingleGaussian : object
            Object with mu, V, and alpha attributes matching XDGMM format
        """
        n_samples, n_features = X.shape

        # Compute weighted mean using inverse error weights
        # For simplicity, use the diagonal of error covariance as weights
        weights = 1.0 / (np.trace(Xerr, axis1=1, axis2=2) / n_features + 1e-10)
        weights /= weights.sum()

        mu = np.sum(weights[:, np.newaxis] * X, axis=0)

        # Compute sample covariance
        X_centered = X - mu
        cov_sample = np.einsum('ni,nj->ij', weights[:, np.newaxis] * X_centered, X_centered)

        # Average error covariance
        cov_error = np.mean(Xerr, axis=0)

        # Total covariance (intrinsic + error)
        cov_total = cov_sample - cov_error
        cov_total = np.maximum(cov_total, 1e-6 * np.eye(n_features))  # Ensure positive definite

        # Return in format compatible with XDGMM
        class SingleGaussian:
            def __init__(self, mu, V):
                self.mu = mu[np.newaxis, :]  # Shape (1, n_features)
                self.V = V[np.newaxis, :, :]  # Shape (1, n_features, n_features)
                self.alpha = np.array([1.0])

        return SingleGaussian(mu, cov_total)

    def _gmm_stage2(self, verbose=False):
        """
        Stage 2: 3D GMM in color-color space (Section 3.2 of Vakili+2019).

        Fits a 2-component XD GMM in multi-color space and selects galaxies
        belonging to the red sequence component based on chi-squared filtering.

        Parameters
        ----------
        verbose : bool, optional
            Print progress information (default: False)
        """
        X = self.color_array  # All colors
        Xerr = self.color_covariance

        if self.stage2_single_gaussian:
            # Single-Gaussian fit on all bin galaxies (skip 2-component XD).
            # After stage-1 chi^2 cut, the bin is already red-dominated; a
            # single Gaussian lands on the red ridge and the chi^2 cut below
            # cleans residual contamination. Avoids degenerate-component
            # branching in the color-transition zone (z~0.4-0.5).
            clf = self._fit_single_gaussian(X, Xerr)
            self._filter_by_chisq_nd(X, clf, 0, sigma=self.stage2_chisq_sigma)
            if verbose:
                print("Stage 2: single-Gaussian fit")
            return

        # Fit 2-component XD GMM
        clf = XDGMM(n_components=2, max_iter=2000, tol=1e-5, verbose=False,
                    random_state=self.random_state)
        clf.fit(X, Xerr)

        weights = clf.alpha
        pi_min = self.pi_min

        # Select component with highest mean in first color.
        # Previously, when both components had nearly-identical mean color
        # (within epsilon=0.03), the code fell back to _fit_single_gaussian,
        # which collapsed onto a near-degenerate intrinsic covariance and the
        # chi^2 cut then rejected the entire bin (e.g. z=[0.19, 0.22)).
        # If the 2-component XD can't tell red from non-red apart, both
        # components are red-enough; argmax is the right answer.
        mean_color = np.where(weights > pi_min, clf.mu[:, 0], -np.inf)
        red_population = int(np.argmax(mean_color))
        # stage2_component="max_weight": the stage-1 cut has already isolated
        # the red peak, so the dominant component is the red sequence. With
        # "reddest", a broad minority component (pi ~0.05-0.09, the
        # non-Gaussian wings) whose mean is marginally redder can be picked;
        # with stage2_on_parent its wide ellipsoid then admits blue galaxies
        # (Euclid DR1 widedeep, 2026-10-01: 6/54 slices, all seeds).
        if self.stage2_component == "max_weight":
            red_population = int(np.argmax(weights))
        frac_red = weights[red_population]

        # Optional near-degeneracy fallback. Two modes:
        #   `reject`   — drop the bin entirely (cleanest, but too aggressive
        #                in red-dominated bins where the 2-component XD is
        #                already crammed close together).
        #   `single`   — verbatim restore of the pre-2026-05-22 behaviour:
        #                fall back to `_fit_single_gaussian` + chi^2 filter.
        # Both triggered only if both components are above pi_min in weight
        # and their first-color means are within `stage2_degeneracy_eps`.
        if self.stage2_degeneracy_eps > 0:
            valid_mask = weights > pi_min
            valid = clf.mu[:, 0][valid_mask]
            if len(valid) >= 2 and (valid.max() - valid.min()) < self.stage2_degeneracy_eps:
                if self.stage2_degeneracy_mode == "single":
                    clf_sg = self._fit_single_gaussian(X, Xerr)
                    self._filter_by_chisq_nd(X, clf_sg, 0, sigma=self.stage2_chisq_sigma)
                    if verbose:
                        print(f"Stage 2: degenerate (Δμ<{self.stage2_degeneracy_eps}) — single-Gaussian fallback")
                    return
                else:  # reject
                    self._apply_mask(np.zeros(len(self.df), dtype=bool))
                    if verbose:
                        print(f"Stage 2: degenerate (Δμ<{self.stage2_degeneracy_eps}) — bin rejected")
                    return

        if verbose:
            print(f"Stage 2: Red fraction = {frac_red:.2f}")

        self.stage2_fit = {
            "mu": clf.mu[red_population].tolist(),
            "V": clf.V[red_population].tolist(),
            "pi": float(frac_red),
            "chi2_crit": float(chi2.ppf(self.stage2_chisq_sigma, df=X.shape[1])),
            "n_in": int(len(X)),
        }
        # Filter by chi-squared in color space
        self._filter_by_chisq_nd(X, clf, red_population, sigma=self.stage2_chisq_sigma)
        self.stage2_fit["n_pass"] = int(len(self.df))

    def _filter_by_chisq_nd(self, X, clf, red_population, sigma):
        """
        Filter galaxies based on chi-squared distance in N-D color space.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Color data
        clf : XDGMM
            Fitted GMM model
        red_population : int
            Index of the red sequence component
        sigma : float
            Chi-squared confidence level for filtering
        """
        Cov = clf.V[red_population]
        mean = clf.mu[red_population]

        # Mahalanobis distance with intrinsic + observed covariance
        C_inv = np.linalg.inv(Cov + self.color_covariance)
        critical_value = chi2.ppf(sigma, df=X.shape[1])
        D_squared = np.einsum('...i,...ij,...j->...', X - mean, C_inv, X - mean)
        self._apply_mask(D_squared < critical_value)
