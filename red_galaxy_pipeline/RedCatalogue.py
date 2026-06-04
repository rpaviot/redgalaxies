"""
RedCatalogue - Main orchestration class for red galaxy pipeline.

Unified interface for red galaxy identification and photometric redshift estimation
based on Vakili et al. (2019) methodology.

Usage modes:
1. Full pipeline: GMM selection → ridge line fitting → photo-z
2. Skip GMM: Assume input is already red galaxies
3. Load saved parameters: Use pre-fitted ridge line parameters
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional, Dict, Callable
from scipy.interpolate import CubicSpline, splrep, splev

from red_galaxy_pipeline.utils import (
    parse_colors, create_magnitude_reference, create_color_arrays_and_covariance
)
from red_galaxy_pipeline.gmm_selection import RedSequenceSelector
from red_galaxy_pipeline.red_sequence_fitting import (
    RedSequenceFitter, save_params, load_params, make_spline_functions,
    make_r_cross_splines,
)
from red_galaxy_pipeline.photoz_estimation import (
    PhotoZFitter, create_schechter_from_params,
    fit_redshift_offset, bias_and_scatter, save_offset_function, load_offset_function
)


class RedCatalogue:
    """
    Main class for red galaxy identification and photo-z estimation.

    Implements the full pipeline from Vakili et al. (2019):
    1. Red sequence selection via Extreme Deconvolution GMM
    2. Ridge line parameter fitting: a(z), b(z), c(z)
    3. Photometric redshift estimation with Schechter priors
    """

    def __init__(self,
                 colors: Union[List[str], List[Tuple[str, str]]],
                 magnitude_col: str = 'mag_v2',
                 z_spec_col: str = 'z_spec',
                 z_min: float = 0.05,
                 z_max: float = 0.95,
                 z_pad: float = 0.05,
                 delta_a: float = 0.05,
                 delta_b: float = 0.10,
                 delta_c: float = 0.15,
                 schechter_params: Optional[Dict] = None,
                 regularization_config: Optional[Dict] = None,
                 cosmology = None,
                 z_bin_width: float = 0.03,
                 color_switch_z: float = 0.42,
                 use_gmm_binning: bool = True,
                 auto_compute_luminosity: bool = True,
                 node_z_min: Optional[float] = None,
                 node_z_max: Optional[float] = None,
                 interp_method: Optional[str] = None,
                 smooth_abc_s: float = 0.0,
                 loss: str = 'l2'):
        """
        Parameters
        ----------
        colors : list
            Color definitions, either as:
            - List of strings: ['gi', 'rz', 'iy']
            - List of tuples: [('g','i'), ('r','z'), ('i','y')]
        magnitude_col : str
            Reference magnitude column name in input catalogs
        z_spec_col : str
            Spectroscopic redshift column name
        z_min, z_max : float
            Redshift range for fitting and photo-z
        delta_a, delta_b, delta_c : float
            Spline node spacings for a(z), b(z), c(z)
        schechter_params : Dict
            Schechter function parameters as arrays. If None, uses simple defaults.
            Dict must contain:
            - 'z_nodes': ndarray of redshift samples
            - 'm_star': ndarray of m*(z) values
            Optional:
            - 'alpha': ndarray of α(z) values (default: constant -1.0)
            - 'phi': ndarray of φ(z) values (default: constant 1.0)
            Example: {'z_nodes': z_array, 'm_star': mstar_array, 'alpha': alpha_array}
        regularization_config : dict, optional
            Regularization for ridge line fitting
        cosmology : astropy cosmology, optional
            Cosmology for photo-z volume prior
        z_bin_width : float
            Redshift bin width for GMM selection (default: 0.03)
        color_switch_z : float
            Redshift at which to swap first two colors in GMM (default: 0.5)
        use_gmm_binning : bool
            If True, run GMM on redshift bins. If False, run on full sample (default: True)
        auto_compute_luminosity : bool
            If True, automatically compute L/L* ratio for red galaxies and add as column
            to df_red (default: True). Uses Schechter m*(z) to compute luminosity ratios.

        Examples
        --------
        >>> # Basic usage with string color definitions
        >>> cat = RedCatalogue(colors=['gi', 'rz', 'iy'])
        >>> df_red = cat.run_full_pipeline(df_spec)
        >>> df_photoz = cat.estimate_photoz(df_photometric)

        >>> # Advanced usage with custom Schechter parameters
        >>> import numpy as np
        >>> z_nodes = np.linspace(0.05, 0.95, 20)
        >>> m_star = 20.0 + 2.0 * np.log(1 + z_nodes)  # Passive evolution
        >>> alpha = -1.0 - 0.1 * z_nodes
        >>> schechter = {'z_nodes': z_nodes, 'm_star': m_star, 'alpha': alpha}
        >>> cat = RedCatalogue(
        ...     colors=[('g','i'), ('r','z'), ('i','y')],
        ...     schechter_params=schechter
        ... )
        """
        # Parse color definitions
        self.color_names, self.color_definitions = parse_colors(colors)

        # Configuration
        self.magnitude_col = magnitude_col
        self.z_spec_col = z_spec_col
        self.z_min = z_min
        self.z_max = z_max
        self.z_pad = z_pad
        self.delta_a = delta_a
        self.delta_b = delta_b
        self.delta_c = delta_c
        # Use provided Schechter params or None (let photoz_estimation use defaults)
        self.schechter_params = schechter_params
        self.regularization_config = regularization_config
        self.cosmology = cosmology
        self.z_bin_width = z_bin_width
        self.color_switch_z = color_switch_z
        self.use_gmm_binning = use_gmm_binning
        self.auto_compute_luminosity = auto_compute_luminosity
        # Ridge-line spline node range (science window). If unset, falls back to
        # the data range as before. Decoupling these lets a/b/c(z) nodes live on
        # [0.1, 1.0] while the fit still uses padded data in [0.05, 1.05], so the
        # boundary nodes are pinned by galaxies straddling them instead of being
        # planted in the sparse padding region.
        self.node_z_min = node_z_min
        self.node_z_max = node_z_max
        self.interp_method = interp_method
        self.smooth_abc_s = float(smooth_abc_s or 0.0)
        # Ridge-fit data-fidelity loss: 'l2' (Gaussian, default) or 'l1'
        # (Laplace, robust to outlier tails that inflate c(z)).
        self.loss = loss

        # State variables
        self.df_red = None  # Selected red galaxies
        self.fit_results = None  # Ridge line parameters
        self.splines = None  # Spline functions
        self.m_ref_func = None  # Reference magnitude function
        self.m_ref_z = None  # Reference magnitude redshift nodes
        self.m_ref_values = None  # Reference magnitude values
        self.fitter = None  # RedSequenceFitter instance
        self.photoz_fitter = None  # PhotoZFitter instance
        self.offset_func = None  # Photo-z offset function

    def select_red_sequence(self, df: pd.DataFrame,
                           sigma_multiplier: float = 2.0,
                           save_filepath: Optional[str] = None,
                           n_jobs: int = 1,
                           n_realizations: int = 1,
                           random_state: Optional[int] = None,
                           verbose: bool = True) -> pd.DataFrame:
        """
        Step 1: Select red sequence galaxies using XD GMM.

        Implements two-stage selection from Vakili et al. (2019) Section 3.2:
        - Stage 1: 2D GMM in (magnitude, first_color) space
        - Stage 2: N-D GMM in multi-color space

        Parameters
        ----------
        df : DataFrame
            Input galaxy catalog with spectroscopic redshifts
        sigma_multiplier : float
            Selection threshold in units of sigma
        save_filepath : str, optional
            Full path to save red catalogue (e.g., 'folder/red_catalogue_Q1.parquet').
            If None, catalogue is not saved automatically.
        verbose : bool
            Print selection statistics

        Returns
        -------
        DataFrame
            Selected red sequence galaxies
        """
        # Pad the user's [z_min, z_max] science window by ±z_pad so the tails
        # are available for quantifying scatter at the limits downstream.
        z_lo = self.z_min - self.z_pad
        z_hi = self.z_max + self.z_pad
        df_cut = df[(df[self.z_spec_col] >= z_lo) &
                   (df[self.z_spec_col] <= z_hi)].copy()

        if verbose:
            print(f"Input: {len(df)} galaxies")
            print(f"After z cuts [{z_lo:.4f}, {z_hi:.4f}] "
                  f"(science [{self.z_min}, {self.z_max}] ±{self.z_pad}): "
                  f"{len(df_cut)} galaxies")

        # Run XD GMM selection
        selector = RedSequenceSelector(
            df_cut,
            self.magnitude_col,
            self.color_definitions,
            self.z_spec_col,
            sigma_multiplier=sigma_multiplier,
            z_bin_width=self.z_bin_width,
            color_switch_z=self.color_switch_z,
            use_binning=self.use_gmm_binning,
            z_min=z_lo,
            z_max=z_hi,
            n_jobs=n_jobs,
            n_realizations=n_realizations,
            random_state=random_state,
        )

        self.df_red = selector.select_red_sequence(verbose=verbose)

        if verbose:
            print(f"Selected red sequence: {len(self.df_red)} galaxies")
            print(f"  ({100*len(self.df_red)/len(df_cut):.1f}% of z-cut sample)")

        # Compute L/L* if enabled
        if self.auto_compute_luminosity:
            self._add_luminosity_ratio(self.df_red, verbose=verbose)

        # Save if filepath provided
        if save_filepath is not None:
            self.save_red_catalogue(save_filepath, verbose=verbose)

        return self.df_red

    def fit_ridge_line(self, df: Optional[pd.DataFrame] = None,
                      method: str = 'single',
                      smooth_mref: bool = False,
                      smooth_s: float = 0.15,
                      fit_cross_covariance: bool = False,
                      delta_r: Optional[float] = None,
                      r_prior_width: float = 0.45,
                      cross_cov_iterations: int = 1,
                      cross_cov_tol: float = 1e-3,
                      save_filepath: Optional[str] = None,
                      verbose: bool = True) -> Dict:
        """
        Step 2: Fit red sequence ridge line parameters a(z), b(z), c(z).

        Implements spline fitting from Vakili et al. (2019) Section 3.3.

        Parameters
        ----------
        df : DataFrame, optional
            Red galaxy catalog. If None, uses self.df_red from step 1.
        method : str
            Fitting method:
            - 'single': Independent fits for each color (more stable)
            - 'joint': Simultaneous fit across all colors
        smooth_mref : bool
            If True, use splrep smoothing for reference magnitude (default: False)
        smooth_s : float
            Smoothing parameter for splrep when smooth_mref=True (default: 0.15)
        save_filepath : str, optional
            Full path to save ridge line parameters (e.g., 'folder/ridgeline_Euclid_Q1.npz').
            If None, ridge line is not saved automatically.
        verbose : bool
            Print fitting statistics

        Returns
        -------
        dict
            Fitted parameters with keys 'nodes', 'a_params', 'b_params', 'c_params'
        """
        if df is None:
            if self.df_red is None:
                raise ValueError("No red galaxy sample. Run select_red_sequence first.")
            df = self.df_red
        else:
            # Apply z cuts if using custom dataframe
            df = df[(df[self.z_spec_col] >= self.z_min) &
                   (df[self.z_spec_col] <= self.z_max)].copy()

        # Spline node range. By default, clamp to the actual galaxy redshift
        # range so the spline doesn't extend into empty regions opened up by
        # step1's padding + binning rounding. If node_z_min/node_z_max are given,
        # plant the nodes on that (narrower) science window instead while still
        # fitting on the full padded sample — galaxies in the padding then pin
        # the boundary nodes rather than spawning runaway extrapolated nodes.
        z_data_min = float(df[self.z_spec_col].min())
        z_data_max = float(df[self.z_spec_col].max())
        z_node_min = self.node_z_min if self.node_z_min is not None else z_data_min
        z_node_max = self.node_z_max if self.node_z_max is not None else z_data_max

        if verbose:
            print(f"Fitting ridge line with {len(df)} red galaxies")
            print(f"  Method: {method}")
            print(f"  Node spacings: Δa={self.delta_a}, Δb={self.delta_b}, Δc={self.delta_c}")
            print(f"  Node range: [{z_node_min:.4f}, {z_node_max:.4f}] "
                  f"(data [{z_data_min:.4f}, {z_data_max:.4f}], "
                  f"science window [{self.z_min}, {self.z_max}])")
            print(f"  Interpolation: {self.interp_method or 'cubic (default)'}"
                  + (f", post-hoc smooth s={self.smooth_abc_s}"
                     if self.smooth_abc_s else ""))
            if smooth_mref:
                print(f"  Using smoothed m_ref (s={smooth_s})")

        # Create reference magnitude function and store underlying data
        self.m_ref_func, self.m_ref_z, self.m_ref_values = create_magnitude_reference(
            df, self.z_spec_col, self.magnitude_col,
            smooth_mref=smooth_mref, smooth_s=smooth_s, return_data=True
        )

        # Initialize fitter
        self.fitter = RedSequenceFitter(
            z_min=z_node_min,
            z_max=z_node_max,
            delta_a=self.delta_a,
            delta_b=self.delta_b,
            delta_c=self.delta_c,
            regularization_config=self.regularization_config,
            interp_method=self.interp_method,
            smooth_abc_s=self.smooth_abc_s,
            loss=self.loss,
        )

        # Setup data (pass m_ref_func to use proper reference magnitude)
        galaxy_data, C_obs, mi_ref = self.fitter.setup_galaxy_data(
            df, self.color_definitions, self.magnitude_col, self.z_spec_col,
            m_ref_func=self.m_ref_func
        )

        # Fit (Stage A: a/b/c per color; optional Stage B: r(z) cross-cov)
        self.fit_results = self.fitter.fit(
            galaxy_data, C_obs, mi_ref, self.color_names, method=method,
            fit_cross_covariance=fit_cross_covariance,
            color_definitions=self.color_definitions,
            delta_r=delta_r,
            r_prior_width=r_prior_width,
            cross_cov_iterations=cross_cov_iterations,
            cross_cov_tol=cross_cov_tol,
            verbose=verbose,
        )

        # Build spline functions
        self.splines = make_spline_functions(self.fit_results, self.color_names)

        if verbose:
            print(f"✓ Ridge line fitting complete")
            for kind in ['a', 'b', 'c']:
                # Get nodes from first color (all colors share same nodes)
                nodes = self.fit_results[self.color_names[0]][kind]['nodes']
                print(f"  {kind}(z): {len(nodes)} nodes from z={nodes[0]:.2f} to {nodes[-1]:.2f}")

        # Save if filepath provided
        if save_filepath is not None:
            self.save_ridge_line(save_filepath, verbose=verbose)

        return self.fit_results

    def load_ridge_line(self, filepath: str,
                        smooth_mref: bool = False,
                        smooth_s: float = 0.15,
                        verbose: bool = True) -> None:
        """
        Load pre-fitted ridge line parameters from disk.

        Parameters
        ----------
        filepath : str
            Full path to ridge line file (e.g., 'folder/ridgeline_Euclid_Q1.npz')
        smooth_mref : bool
            If True, apply splrep smoothing to reference magnitude (default: False)
        smooth_s : float
            Smoothing parameter for splrep when smooth_mref=True (default: 0.15)
        verbose : bool
            Print loading statistics
        """
        self.fit_results, m_ref_data = load_params(filepath)
        self.splines = make_spline_functions(self.fit_results, self.color_names)

        # Reconstruct reference magnitude function if data is available
        if m_ref_data is not None:
            m_ref_z, m_ref_values = m_ref_data
            self.m_ref_z = m_ref_z
            self.m_ref_values = m_ref_values
            if smooth_mref:
                tck = splrep(m_ref_z, m_ref_values, s=smooth_s)
                self.m_ref_func = lambda z: splev(z, tck)
            else:
                self.m_ref_func = CubicSpline(m_ref_z, m_ref_values)
        else:
            self.m_ref_func = None
            self.m_ref_z = None
            self.m_ref_values = None

        if verbose:
            print(f"Loaded ridge line parameters from {filepath}")
            for kind in ['a', 'b', 'c']:
                # Get nodes from first color (all colors share same nodes)
                nodes = self.fit_results[self.color_names[0]][kind]['nodes']
                print(f"  {kind}(z): {len(nodes)} nodes")
            if m_ref_data is not None:
                print(f"  m_ref(z): {len(m_ref_data[0])} nodes" +
                      (" (smoothed)" if smooth_mref else ""))
            else:
                print("  Warning: No reference magnitude function in saved file")

    def save_ridge_line(self, filepath: str, verbose: bool = True) -> None:
        """
        Save fitted ridge line parameters to disk.

        Parameters
        ----------
        filepath : str
            Full path to output file (e.g., 'folder/ridgeline_Euclid_Q1.npz')
        verbose : bool
            Print saving statistics
        """
        if self.fit_results is None:
            raise ValueError("No ridge line parameters to save. Run fit_ridge_line first.")

        # Use stored m_ref data arrays
        m_ref_z = getattr(self, 'm_ref_z', None)
        m_ref_values = getattr(self, 'm_ref_values', None)

        save_params(self.fit_results, filepath, m_ref_z=m_ref_z, m_ref_values=m_ref_values)

        if verbose:
            print(f"Saved ridge line parameters to {filepath}")
            if m_ref_z is not None:
                print(f"  Including reference magnitude function ({len(m_ref_z)} nodes)")

    def estimate_photoz(self, df: pd.DataFrame,
                       z_init: float = 0.5,
                       method: str = 'iminuit',
                       n_jobs: int = 1,
                       return_chi2: bool = False,
                       offset: bool = False,
                       verbose: bool = True) -> pd.DataFrame:
        """
        Step 3: Estimate photometric redshifts.

        Implements Bayesian photo-z with Schechter priors.

        Parameters
        ----------
        df : DataFrame
            Photometric galaxy catalog
        z_init : float
            Initial redshift guess for all galaxies (only used for 'iminuit' method)
        method : str
            Optimization method: 'iminuit' (default, fast local optimizer)
            or 'differential_evolution' (more robust global optimizer)
        n_jobs : int
            Number of parallel jobs. -1 uses all CPU cores, 1 = sequential (default).
        return_chi2 : bool
            If True, add 'chi2' column to output DataFrame
        offset : bool
            If True, apply offset correction. Uses self.offset_func if available,
            otherwise raises an error (run calibrate_photoz_offset first).
        verbose : bool
            Print progress

        Returns
        -------
        DataFrame
            Input catalog with added 'z_phot' column (and 'chi2' if return_chi2=True)

        Examples
        --------
        >>> # Basic usage
        >>> df_phot = cat.estimate_photoz(df_photometric)
        >>>
        >>> # With offset correction (after calibration)
        >>> cat.calibrate_photoz_offset(df_spec)
        >>> df_phot = cat.estimate_photoz(df_photometric, offset=True)
        """
        if self.splines is None:
            raise ValueError("Ridge line not fitted. Run fit_ridge_line or load_ridge_line first.")

        if self.m_ref_func is None:
            # Create default m_ref_func if not available
            if self.df_red is not None:
                self.m_ref_func = create_magnitude_reference(
                    self.df_red, self.z_spec_col, self.magnitude_col
                )
            else:
                raise ValueError("No reference magnitude function. Run select_red_sequence or fit_ridge_line first.")

        # Check offset function if offset requested
        offset_func = None
        if offset:
            if self.offset_func is None:
                raise ValueError("Offset requested but no offset function available. Run calibrate_photoz_offset first.")
            offset_func = self.offset_func

        if verbose:
            msg = f"Estimating photo-z for {len(df)} galaxies using {method}"
            if n_jobs != 1:
                msg += f" ({n_jobs if n_jobs > 0 else 'all'} CPU cores)"
            if offset:
                msg += " with offset correction"
            print(msg)

        # Build optional r(z) cross-covariance splines from fit results
        r_splines = make_r_cross_splines(self.fit_results, self.color_names) \
            if self.fit_results is not None else {}

        # Initialize photo-z fitter
        self.photoz_fitter = PhotoZFitter(
            spline_functions=self.splines,
            m_ref_func=self.m_ref_func,
            colour_names=self.color_names,
            schechter_params=self.schechter_params,
            cosmology=self.cosmology,
            z_min=self.z_min,
            z_max=self.z_max,
            offset_func=offset_func,
            apply_offset=offset,
            r_splines=r_splines,
        )

        # Estimate photo-z
        df_out = self.photoz_fitter.estimate_photoz(
            df, self.color_definitions, self.magnitude_col, z_init, method,
            n_jobs, return_chi2, verbose
        )

        if verbose:
            valid_z = np.isfinite(df_out['z_phot'])
            print(f"✓ Photo-z estimation complete")
            print(f"  Valid estimates: {valid_z.sum()}/{len(df)} ({100*valid_z.sum()/len(df):.1f}%)")

        # Add L/L* if enabled and valid photo-z exist
        if self.auto_compute_luminosity and self.schechter_params is not None:
            valid_mask = np.isfinite(df_out['z_phot'])
            if valid_mask.sum() > 0:
                # Compute L/L* using photo-z (not spec-z, since this is photometric)
                schechter = create_schechter_from_params(self.schechter_params)
                l_ratio = np.full(len(df_out), np.nan)
                l_ratio[valid_mask] = schechter.compute_luminosity_ratio(
                    df_out.loc[valid_mask, self.magnitude_col].values,
                    df_out.loc[valid_mask, 'z_phot'].values
                )
                df_out['l_ratio'] = l_ratio

                if verbose:
                    valid_l = np.isfinite(l_ratio)
                    print(f"  Computed L/L* ratio for {valid_l.sum()} galaxies")
                    if valid_l.sum() > 0:
                        print(f"    Median L/L*: {np.nanmedian(l_ratio):.3f}")
                        pct_bright = 100 * np.sum(l_ratio > 0.5) / valid_l.sum()
                        pct_lum = 100 * np.sum(l_ratio > 1.0) / valid_l.sum()
                        print(f"    L/L* > 0.5: {np.sum(l_ratio > 0.5)} galaxies ({pct_bright:.1f}%)")
                        print(f"    L/L* > 1.0: {np.sum(l_ratio > 1.0)} galaxies ({pct_lum:.1f}%)")

        return df_out

    def refine_red_sequence(self, df: Optional[pd.DataFrame] = None,
                           sigma_multiplier: float = 2.0,
                           smooth_mref: bool = False,
                           smooth_s: float = 0.15,
                           verbose: bool = True) -> pd.DataFrame:
        """
        Post-process the red sequence selection by applying chi-squared filtering
        using the fitted ridge line parameters. This refines the initial GMM selection.

        Based on step2.ipynb methodology where the full spec-z catalog is filtered
        using the fitted red sequence model.

        Parameters
        ----------
        df : DataFrame, optional
            Galaxy catalog to filter. If None, uses the full spec-z catalog.
            Should contain color and magnitude columns.
        sigma_multiplier : float
            Chi-squared threshold (default: 2.0 for ~86% confidence)
        smooth_mref : bool
            If True, smooth the reference magnitude using splrep
        smooth_s : float
            Smoothing parameter for splrep (default: 0.15)
        verbose : bool
            Print filtering statistics

        Returns
        -------
        DataFrame
            Refined red sequence galaxies

        Notes
        -----
        This method requires that ridge line fitting has been completed.
        The filtering computes chi-squared distance in each color independently:

        chi2 = (color_obs - color_model)^2 / (c(z)^2 + color_err)

        where color_model = a(z) + b(z) * (m - m_ref(z))

        A galaxy is selected if chi2 < sigma_multiplier for ALL colors.
        """
        if self.fit_results is None or self.splines is None:
            raise ValueError("Ridge line not fitted. Run fit_ridge_line first.")

        if df is None:
            raise ValueError("Must provide a DataFrame to filter")

        # Apply redshift cuts
        df_filtered = df[(df[self.z_spec_col] >= self.z_min) &
                        (df[self.z_spec_col] <= self.z_max)].copy()

        if verbose:
            print(f"Refining red sequence with chi-squared filtering")
            print(f"  Input: {len(df_filtered)} galaxies")
            print(f"  Sigma threshold: {sigma_multiplier:.1f}")

        # Create reference magnitude function (optionally smoothed)
        if smooth_mref:
            self.m_ref_func = create_magnitude_reference(
                df_filtered, self.z_spec_col, self.magnitude_col,
                smooth_mref=True, smooth_s=smooth_s
            )
            mi_ref = self.m_ref_func(df_filtered[self.z_spec_col].values)

            if verbose:
                print(f"  Using smoothed m_ref (s={smooth_s})")
        else:
            # Use existing m_ref function or create new one
            if self.m_ref_func is None:
                self.m_ref_func = create_magnitude_reference(
                    df_filtered, self.z_spec_col, self.magnitude_col
                )
            mi_ref = self.m_ref_func(df_filtered[self.z_spec_col].values)

        # Create color arrays with error covariance
        color_array, color_covariance = create_color_arrays_and_covariance(
            df_filtered, self.color_definitions, self.magnitude_col
        )

        # Apply chi-squared filtering for each color
        combined_mask = np.ones(len(df_filtered), dtype=bool)

        for i, color_name in enumerate(self.color_names):
            # Get spline parameters for this color
            a_z = self.splines[color_name]['a'](df_filtered[self.z_spec_col].values)
            b_z = self.splines[color_name]['b'](df_filtered[self.z_spec_col].values)
            c_z = self.splines[color_name]['c'](df_filtered[self.z_spec_col].values)

            # Expected color from model
            expected_color = a_z + b_z * (df_filtered[self.magnitude_col].values - mi_ref)

            # Observed color
            observed_color = color_array[:, i]

            # Color error (diagonal of covariance)
            color_err = color_covariance[:, i, i]

            # Chi-squared distance
            sigma_sq = c_z**2 + color_err
            chi2 = (observed_color - expected_color)**2 / sigma_sq

            # Apply threshold
            mask_i = chi2 < sigma_multiplier
            combined_mask &= mask_i

            if verbose:
                print(f"  {color_name}: {mask_i.sum()}/{len(df_filtered)} pass ({100*mask_i.sum()/len(df_filtered):.1f}%)")

        # Apply combined mask
        df_refined = df_filtered[combined_mask].copy()

        if verbose:
            print(f"  Final refined sample: {len(df_refined)} galaxies ({100*len(df_refined)/len(df_filtered):.1f}%)")

        # Update internal state
        self.df_red = df_refined

        # Compute L/L* if enabled
        if self.auto_compute_luminosity:
            self._add_luminosity_ratio(self.df_red, verbose=verbose)

        return df_refined

    def run_full_pipeline(self, df_spec: pd.DataFrame,
                         sigma_multiplier: float = 2.0,
                         fit_method: str = 'single',
                         refine: bool = False,
                         refine_sigma: float = 2.0,
                         smooth_mref: bool = False,
                         smooth_s: float = 0.15,
                         save_ridgeline_filepath: Optional[str] = None,
                         verbose: bool = True) -> pd.DataFrame:
        """
        Run the complete pipeline: selection → fitting → optional refinement.

        Parameters
        ----------
        df_spec : DataFrame
            Spectroscopic galaxy catalog for training
        sigma_multiplier : float
            Selection threshold for GMM (Stage 1)
        fit_method : str
            Ridge line fitting method ('single' or 'joint')
        refine : bool
            If True, apply post-processing chi-squared filtering to refine selection
        refine_sigma : float
            Chi-squared threshold for refinement step (if refine=True)
        smooth_mref : bool
            If True, smooth reference magnitude with splrep (if refine=True)
        smooth_s : float
            Smoothing parameter for splrep (if smooth_mref=True)
        save_ridgeline_filepath : str, optional
            Full path to save ridge line parameters (e.g., 'folder/ridgeline.npz').
            If None, ridge line is not saved automatically.
        verbose : bool
            Print progress

        Returns
        -------
        DataFrame
            Selected red sequence galaxies with fitted parameters stored

        Examples
        --------
        >>> # Basic pipeline
        >>> cat = RedCatalogue(colors=['gi', 'rz', 'iy'])
        >>> df_red = cat.run_full_pipeline(df_spec)

        >>> # With post-processing refinement and saving
        >>> df_red = cat.run_full_pipeline(
        ...     df_spec,
        ...     refine=True,
        ...     refine_sigma=2.0,
        ...     smooth_mref=True,
        ...     smooth_s=0.15,
        ...     save_ridgeline_filepath='output/ridgeline.npz'
        ... )
        """
        if verbose:
            print("=" * 60)
            print("Red Galaxy Pipeline - Full Run")
            print("=" * 60)

        # Step 1: Select red sequence
        if verbose:
            print("\nStep 1: Red Sequence Selection (GMM)")
            print("-" * 60)
        self.select_red_sequence(df_spec, sigma_multiplier, verbose=verbose)

        # Step 2: Fit ridge line
        if verbose:
            print("\nStep 2: Ridge Line Fitting")
            print("-" * 60)
        self.fit_ridge_line(method=fit_method, verbose=verbose)

        # Optional Step 3: Refine selection using fitted model
        if refine:
            if verbose:
                print("\nStep 3: Refining Selection with Chi-Squared Filter")
                print("-" * 60)
            self.refine_red_sequence(
                df=df_spec,
                sigma_multiplier=refine_sigma,
                smooth_mref=smooth_mref,
                smooth_s=smooth_s,
                verbose=verbose
            )

            # Re-fit with refined sample
            if verbose:
                print("\nStep 4: Re-fitting Ridge Line with Refined Sample")
                print("-" * 60)
            self.fit_ridge_line(method=fit_method, verbose=verbose)

        # Save if filepath provided
        if save_ridgeline_filepath is not None:
            self.save_ridge_line(save_ridgeline_filepath, verbose=verbose)

        if verbose:
            print("\n" + "=" * 60)
            print("Pipeline complete!")
            print("=" * 60)

        return self.df_red

    def calibrate_photoz_offset(self, df_spec: pd.DataFrame,
                                n_nodes: int = 10,
                                train_fraction: float = 0.5,
                                method: str = 'differential_evolution',
                                n_jobs: int = -1,
                                random_seed: int = 42,
                                save_filepath: Optional[str] = None,
                                compute_luminosity_diagnostics: bool = False,
                                l_thresholds: List[float] = [0.5, 1.0],
                                offset_interp: str = 'cubic',
                                verbose: bool = True) -> Tuple[Callable, Dict]:
        """
        Calibrate photo-z offset function using spectroscopic sample.

        This method fits an offset function Δz(z) to minimize systematic bias in
        photo-z estimates. It uses a train/validation split to avoid overfitting.
        The fitted offset function is stored in self.offset_func.

        Workflow:
        1. Split spec-z sample into train (default 50%) and validation sets
        2. Run photo-z estimation on training set (without offset)
        3. Fit cubic spline offset function: Δz(z_phot) to minimize |z_spec - (z_phot + Δz)|
        4. Validate on held-out set and compute bias/scatter metrics
        5. Optionally save offset function to provided filepath

        Parameters
        ----------
        df_spec : DataFrame
            Spectroscopic galaxy catalog (must have z_spec column)
        n_nodes : int
            Number of redshift nodes for offset spline (default: 10).
            Creates bins from z_min to z_max with n_nodes points.
        train_fraction : float
            Fraction of data to use for training (default: 0.50)
        method : str
            Optimization method for photo-z: 'iminuit' or 'differential_evolution' (default)
        n_jobs : int
            Number of parallel jobs for photo-z estimation. -1 = all cores (default).
        random_seed : int
            Random seed for train/validation split (default: 42)
        save_filepath : str, optional
            Full path to save offset function (e.g., 'folder/offset_Euclid_Q1.npz').
            If None, offset is not saved automatically.
        compute_luminosity_diagnostics : bool
            If True, compute bias/scatter stratified by luminosity ratio L/L*.
            Following Vakili et al. (2019), evaluates performance for different
            luminosity samples (all, L/L*>0.5, L/L*>1.0). Default: False.
        l_thresholds : list of float
            Luminosity thresholds to evaluate when compute_luminosity_diagnostics=True.
            Default: [0.5, 1.0] (dense and luminous samples).
        verbose : bool
            Print progress and diagnostics

        Returns
        -------
        offset_func : CubicSpline
            Fitted offset function Δz(z). Also stored in self.offset_func.
        metrics : dict
            Dictionary containing:
            - 'train_bias', 'train_scatter': Pre-offset training metrics
            - 'val_bias', 'val_scatter': Post-offset validation metrics
            - 'train_z_bins', 'val_z_bins': Redshift bin centers

        Examples
        --------
        >>> # Calibrate and save offset
        >>> offset_func, metrics = cat.calibrate_photoz_offset(
        ...     df_spec, save_filepath='output/offset_Euclid_Q1.npz'
        ... )
        >>> print(f"Mean bias before: {np.nanmean(metrics['train_bias']):.4f}")
        >>> print(f"Mean bias after: {np.nanmean(metrics['val_bias']):.4f}")
        """
        if self.splines is None:
            raise ValueError("Ridge line not fitted. Run fit_ridge_line() first.")

        # Create z_bins from n_nodes
        z_bins = np.linspace(self.z_min, self.z_max, n_nodes)

        if verbose:
            print("=" * 60)
            print("Photo-z Offset Calibration")
            print("=" * 60)
            print(f"Using {n_nodes} nodes from z={self.z_min:.2f} to z={self.z_max:.2f}")

        # Split into train/validation
        n_galaxies = len(df_spec)
        rng = np.random.default_rng(seed=random_seed)
        all_indices = np.arange(n_galaxies)
        n_train = int(n_galaxies * train_fraction)
        train_indices = rng.choice(all_indices, size=n_train, replace=False)
        val_indices = np.setdiff1d(all_indices, train_indices)

        df_train = df_spec.iloc[train_indices].copy()
        df_val = df_spec.iloc[val_indices].copy()

        if verbose:
            print(f"\nTrain set: {len(df_train)} galaxies ({100*train_fraction:.1f}%)")
            print(f"Validation set: {len(df_val)} galaxies ({100*(1-train_fraction):.1f}%)")

        # Get true redshifts
        z_spec_train = df_train[self.z_spec_col].values
        z_spec_val = df_val[self.z_spec_col].values

        # Run photo-z on training set (without offset)
        if verbose:
            print(f"\nEstimating photo-z on training set using {method}...")
            if n_jobs != 1:
                print(f"  Using {n_jobs if n_jobs > 0 else 'all'} CPU cores")

        # Optional cross-cov r(z) splines for downstream photo-z chi^2
        r_splines = make_r_cross_splines(self.fit_results, self.color_names) \
            if self.fit_results is not None else {}

        # Create temporary PhotoZFitter without offset
        temp_fitter = PhotoZFitter(
            spline_functions=self.splines,
            m_ref_func=self.m_ref_func,
            colour_names=self.color_names,
            schechter_params=self.schechter_params,
            cosmology=self.cosmology,
            z_min=self.z_min,
            z_max=self.z_max,
            offset_func=None,
            apply_offset=False,
            r_splines=r_splines,
        )

        df_train_phot = temp_fitter.estimate_photoz(
            df_train, self.color_definitions, self.magnitude_col,
            z_init=0.5, method=method, n_jobs=n_jobs, verbose=verbose
        )
        z_photo_train = df_train_phot['z_phot'].values

        # Compute pre-offset bias and scatter
        z_centers_train, bias_train, scatter_train, n_train = bias_and_scatter(
            z_bins, z_spec_train, z_photo_train
        )

        # Compute luminosity diagnostics if requested
        train_lum_diag = None
        if compute_luminosity_diagnostics:
            m_obs_train = df_train[self.magnitude_col].values
            schechter = create_schechter_from_params(self.schechter_params)
            # Use z_spec for L/L* computation (true luminosity, not photometric)
            from red_galaxy_pipeline.photoz_estimation import compute_luminosity_diagnostics as _compute_lum_diag
            train_lum_diag = _compute_lum_diag(
                z_photo_train, z_spec_train, m_obs_train, schechter,
                z_bins, l_thresholds, use_spec_for_lum=True
            )

        if verbose:
            print("\nPre-offset training metrics (all galaxies):")
            print(f"  Mean bias: {np.nanmean(bias_train):.4f}")
            print(f"  Mean NMAD scatter: {np.nanmean(scatter_train):.4f}")

            if compute_luminosity_diagnostics:
                for l_thresh in l_thresholds:
                    key = f'l_{l_thresh}'.replace('.', '')
                    lum_data = train_lum_diag[key]
                    n_lum = lum_data['n_galaxies']
                    pct = 100 * n_lum / n_train
                    print(f"\nPre-offset training metrics (L/L* > {l_thresh}):")
                    print(f"  Mean bias: {np.nanmean(lum_data['bias']):.4f}  [N={n_lum}, {pct:.1f}% of sample]")
                    print(f"  Mean NMAD scatter: {np.nanmean(lum_data['scatter']):.4f}")

        # Fit offset function using z_bins as nodes
        if verbose:
            print(f"\nFitting offset function with {len(z_bins)} nodes...")

        # iminuit returns NaN on failed fits; drop those before L-BFGS-B
        # so the offset spline parameter search doesn't blow up.
        valid = np.isfinite(z_photo_train) & np.isfinite(z_spec_train)
        n_dropped = int((~valid).sum())
        if n_dropped > 0 and verbose:
            print(f"  Dropping {n_dropped} galaxies with NaN photo-z "
                  f"({100*n_dropped/len(valid):.2f}%)")
        offset_func = fit_redshift_offset(
            z_photo_train[valid], z_spec_train[valid], z_bins,
            interp=offset_interp
        )

        # Validate on validation set
        if len(df_val) > 0:
            if verbose:
                print("\nValidating on held-out set...")

            # Create fitter WITH offset
            val_fitter = PhotoZFitter(
                spline_functions=self.splines,
                m_ref_func=self.m_ref_func,
                colour_names=self.color_names,
                schechter_params=self.schechter_params,
                cosmology=self.cosmology,
                z_min=self.z_min,
                z_max=self.z_max,
                offset_func=offset_func,
                apply_offset=True,
                r_splines=r_splines,
            )

            df_val_phot = val_fitter.estimate_photoz(
                df_val, self.color_definitions, self.magnitude_col,
                z_init=0.5, method=method, n_jobs=n_jobs, verbose=False
            )
            z_photo_val = df_val_phot['z_phot'].values

            z_centers_val, bias_val, scatter_val, n_val = bias_and_scatter(
                z_bins, z_spec_val, z_photo_val
            )

            # Compute luminosity diagnostics for validation set
            val_lum_diag = None
            if compute_luminosity_diagnostics:
                m_obs_val = df_val[self.magnitude_col].values
                schechter = create_schechter_from_params(self.schechter_params)
                # Use z_photo for L/L* (offset already applied in validation)
                val_lum_diag = _compute_lum_diag(
                    z_photo_val, z_spec_val, m_obs_val, schechter,
                    z_bins, l_thresholds, use_spec_for_lum=False
                )

            if verbose:
                print("\nPost-offset validation metrics (all galaxies):")
                print(f"  Mean bias: {np.nanmean(bias_val):.4f}")
                print(f"  Mean NMAD scatter: {np.nanmean(scatter_val):.4f}")

                if compute_luminosity_diagnostics:
                    for l_thresh in l_thresholds:
                        key = f'l_{l_thresh}'.replace('.', '')
                        lum_data = val_lum_diag[key]
                        n_lum = lum_data['n_galaxies']
                        pct = 100 * n_lum / n_val
                        print(f"\nPost-offset validation metrics (L/L* > {l_thresh}):")
                        print(f"  Mean bias: {np.nanmean(lum_data['bias']):.4f}  [N={n_lum}, {pct:.1f}% of sample]")
                        print(f"  Mean NMAD scatter: {np.nanmean(lum_data['scatter']):.4f}")
        else:
            # No validation set - use training metrics
            if verbose:
                print("\nWarning: No validation set (train_fraction too high)")
            z_centers_val = z_centers_train
            z_photo_train_corr = z_photo_train + offset_func(z_photo_train)
            z_centers_val, bias_val, scatter_val, n_val = bias_and_scatter(
                z_bins, z_spec_train, z_photo_train_corr
            )
            val_lum_diag = None

        # Store calibration data for later use by save_offset()
        train_metrics = {
            'bias': bias_train,
            'scatter': scatter_train,
            'z_bins': z_centers_train
        }
        val_metrics = {
            'bias': bias_val,
            'scatter': scatter_val,
            'z_bins': z_centers_val
        }

        # Prepare luminosity diagnostics for saving
        lum_diag_to_save = None
        if compute_luminosity_diagnostics:
            lum_diag_to_save = {
                'train': train_lum_diag,
                'val': val_lum_diag if val_lum_diag is not None else {}
            }

        # Store offset function and calibration data
        self.offset_func = offset_func
        self._offset_calibration_data = {
            'z_nodes': z_bins,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'luminosity_diagnostics': lum_diag_to_save,
            'offset_interp': offset_interp,
        }

        # Save if filepath provided
        if save_filepath is not None:
            self.save_offset(save_filepath, verbose=verbose)

        # Compile metrics
        metrics = {
            'train_bias': bias_train,
            'train_scatter': scatter_train,
            'train_z_bins': z_centers_train,
            'val_bias': bias_val,
            'val_scatter': scatter_val,
            'val_z_bins': z_centers_val
        }

        # Add luminosity diagnostics to returned metrics
        if compute_luminosity_diagnostics:
            for l_thresh in l_thresholds:
                key = f'l_{l_thresh}'.replace('.', '')
                # Training luminosity metrics
                metrics[f'train_{key}_bias'] = train_lum_diag[key]['bias']
                metrics[f'train_{key}_scatter'] = train_lum_diag[key]['scatter']
                metrics[f'train_{key}_n'] = train_lum_diag[key]['n_galaxies']
                # Validation luminosity metrics
                if val_lum_diag:
                    metrics[f'val_{key}_bias'] = val_lum_diag[key]['bias']
                    metrics[f'val_{key}_scatter'] = val_lum_diag[key]['scatter']
                    metrics[f'val_{key}_n'] = val_lum_diag[key]['n_galaxies']

        if verbose:
            print("\n" + "=" * 60)
            print("Offset calibration complete!")
            print("=" * 60)

        return offset_func, metrics

    def compute_photoz_diagnostics_by_luminosity(self,
                                                 df_with_photoz: pd.DataFrame,
                                                 z_phot_col: str = 'z_phot',
                                                 z_spec_col: Optional[str] = None,
                                                 magnitude_col: Optional[str] = None,
                                                 z_bins: Optional[np.ndarray] = None,
                                                 l_thresholds: List[float] = [0.5, 1.0]) -> pd.DataFrame:
        """
        Compute photo-z diagnostics stratified by luminosity ratio L/L*.

        Convenience method for post-hoc analysis of photo-z quality as a function
        of galaxy luminosity. Useful for understanding which galaxies have reliable
        photo-z estimates and for selecting samples by luminosity.

        Parameters
        ----------
        df_with_photoz : DataFrame
            Catalog with photo-z estimates (must have z_phot_col column)
        z_phot_col : str
            Column name for photometric redshifts (default: 'z_phot')
        z_spec_col : str, optional
            Column name for spectroscopic redshifts. If None, uses self.z_spec_col.
        magnitude_col : str, optional
            Column name for magnitudes. If None, uses self.magnitude_col.
        z_bins : ndarray, optional
            Bin edges for redshift. If None, uses 10 bins from z_min to z_max.
        l_thresholds : list of float
            Luminosity thresholds to evaluate. Default: [0.5, 1.0]

        Returns
        -------
        diagnostics_df : DataFrame
            Summary DataFrame with columns:
            - 'sample': sample name ('all', 'l_05', 'l_10', etc.)
            - 'n_galaxies': number of galaxies in sample
            - 'fraction': fraction of total sample
            - 'mean_bias': mean of bias array
            - 'mean_scatter': mean of scatter array (NMAD)

        Examples
        --------
        >>> # After running photo-z
        >>> df_result = cat.estimate_photoz(df_photometric)
        >>>
        >>> # Compute luminosity diagnostics
        >>> diag_df = cat.compute_photoz_diagnostics_by_luminosity(df_result)
        >>> print(diag_df)
        >>>
        >>> # Show which sample has best performance
        >>> best = diag_df.loc[diag_df['mean_scatter'].idxmin()]
        >>> print(f"Best sample: {best['sample']} (NMAD={best['mean_scatter']:.4f})")
        """
        from red_galaxy_pipeline.photoz_estimation import compute_luminosity_diagnostics as _compute_lum_diag

        # Use defaults if not provided
        if z_spec_col is None:
            z_spec_col = self.z_spec_col
        if magnitude_col is None:
            magnitude_col = self.magnitude_col
        if z_bins is None:
            z_bins = np.linspace(self.z_min, self.z_max, 11)

        # Check required columns
        if z_phot_col not in df_with_photoz.columns:
            raise ValueError(f"Column '{z_phot_col}' not found in DataFrame")
        if z_spec_col not in df_with_photoz.columns:
            raise ValueError(f"Column '{z_spec_col}' not found. This method requires spec-z for validation.")

        # Extract data
        z_photo = df_with_photoz[z_phot_col].values
        z_spec = df_with_photoz[z_spec_col].values
        m_obs = df_with_photoz[magnitude_col].values

        # Get Schechter function
        schechter = create_schechter_from_params(self.schechter_params)

        # Compute diagnostics
        diagnostics = _compute_lum_diag(
            z_photo, z_spec, m_obs, schechter, z_bins, l_thresholds
        )

        # Convert to DataFrame for easier analysis
        summary_data = []
        n_total = len(df_with_photoz)

        # All galaxies
        all_diag = diagnostics['all']
        summary_data.append({
            'sample': 'all',
            'n_galaxies': all_diag['n_galaxies'],
            'fraction': 1.0,
            'mean_bias': np.nanmean(all_diag['bias']),
            'mean_scatter': np.nanmean(all_diag['scatter'])
        })

        # Luminosity-selected samples
        for l_thresh in l_thresholds:
            key = f'l_{l_thresh}'.replace('.', '')
            lum_diag = diagnostics[key]
            summary_data.append({
                'sample': f'L/L*>{l_thresh}',
                'n_galaxies': lum_diag['n_galaxies'],
                'fraction': lum_diag['n_galaxies'] / n_total,
                'mean_bias': np.nanmean(lum_diag['bias']),
                'mean_scatter': np.nanmean(lum_diag['scatter'])
            })

        return pd.DataFrame(summary_data)

    def get_config(self) -> Dict:
        """
        Get pipeline configuration.

        Returns
        -------
        dict
            Configuration parameters
        """
        return {
            'colors': self.color_names,
            'color_definitions': self.color_definitions,
            'magnitude_col': self.magnitude_col,
            'z_spec_col': self.z_spec_col,
            'z_min': self.z_min,
            'z_max': self.z_max,
            'delta_a': self.delta_a,
            'delta_b': self.delta_b,
            'delta_c': self.delta_c,
            'schechter_params': self.schechter_params,
            'regularization': self.regularization_config,
            'auto_compute_luminosity': self.auto_compute_luminosity
        }

    def _add_luminosity_ratio(self, df: pd.DataFrame, verbose: bool = True) -> None:
        """
        Add L/L* ratio to the red galaxy catalog.

        Parameters
        ----------
        df : DataFrame
            Red galaxy catalog (modified in-place)
        verbose : bool
            Print statistics about luminosity distribution
        """
        if self.schechter_params is None:
            if verbose:
                print("  Note: Schechter parameters not provided. Skipping L/L* computation.")
            return

        # Create Schechter function
        schechter = create_schechter_from_params(self.schechter_params)

        # Compute L/L* for all galaxies
        l_ratio = schechter.compute_luminosity_ratio(
            df[self.magnitude_col].values,
            df[self.z_spec_col].values
        )

        # Add as column
        df['l_ratio'] = l_ratio

        if verbose:
            print(f"  Computed L/L* ratio:")
            print(f"    Median L/L*: {np.median(l_ratio):.3f}")
            pct_bright = 100 * np.sum(l_ratio > 0.5) / len(l_ratio)
            pct_lum = 100 * np.sum(l_ratio > 1.0) / len(l_ratio)
            print(f"    L/L* > 0.5: {np.sum(l_ratio > 0.5)} galaxies ({pct_bright:.1f}%)")
            print(f"    L/L* > 1.0: {np.sum(l_ratio > 1.0)} galaxies ({pct_lum:.1f}%)")

    def load_red_catalogue(self, filepath: str,
                          compute_luminosity: Optional[bool] = None,
                          verbose: bool = True) -> pd.DataFrame:
        """
        Load a pre-selected red galaxy catalog from a parquet file.

        Use this method when you already have a red sequence sample
        (e.g., from a previous GMM run or a different selection method)
        and want to skip the GMM selection step.

        Parameters
        ----------
        filepath : str
            Full path to parquet file (e.g., 'folder/red_catalogue_Q1.parquet')
        compute_luminosity : bool, optional
            If True, compute L/L* and add to catalog. If None, uses
            self.auto_compute_luminosity setting (default).
        verbose : bool
            Print loading statistics

        Returns
        -------
        DataFrame
            The loaded red catalog (with L/L* added if requested)

        Examples
        --------
        >>> cat = RedCatalogue(colors=['gi', 'rz', 'iy'])
        >>> cat.load_red_catalogue('output/red_catalogue_Q1.parquet')
        >>> cat.fit_ridge_line()  # Skip GMM, go straight to fitting

        >>> # Load and disable automatic L/L* computation
        >>> cat.load_red_catalogue('output/red_catalogue_Q1.parquet', compute_luminosity=False)
        """
        # Load from parquet file
        df_red = pd.read_parquet(filepath)

        # Apply redshift cuts
        df_cut = df_red[(df_red[self.z_spec_col] >= self.z_min) &
                       (df_red[self.z_spec_col] <= self.z_max)].copy()

        if verbose:
            print("=" * 60)
            print("Loading Pre-Selected Red Catalogue")
            print("=" * 60)
            print(f"File: {filepath}")
            print(f"Input: {len(df_red)} galaxies")
            print(f"After z cuts [{self.z_min}, {self.z_max}]: {len(df_cut)} galaxies")

        # Set internal state
        self.df_red = df_cut

        # Compute L/L* if requested
        if compute_luminosity is None:
            compute_luminosity = self.auto_compute_luminosity

        if compute_luminosity:
            self._add_luminosity_ratio(self.df_red, verbose=verbose)

        if verbose:
            print("=" * 60)
            print("Red catalogue loaded successfully!")
            print("=" * 60)

        return self.df_red

    def save_red_catalogue(self, filepath: str, verbose: bool = True) -> None:
        """
        Save red galaxy catalogue to parquet file.

        Parameters
        ----------
        filepath : str
            Output file path (should end with .parquet)
        verbose : bool
            Print save statistics

        Examples
        --------
        >>> cat.select_red_sequence(df_spec)
        >>> cat.save_red_catalogue('red_galaxies.parquet')
        """
        if self.df_red is None:
            raise ValueError("No red catalogue to save. Run select_red_sequence or load_red_catalogue first.")

        self.df_red.to_parquet(filepath, index=False)

        if verbose:
            print(f"Saved {len(self.df_red)} red galaxies to {filepath}")

    def set_schechter_params(self, schechter_params: Dict, verbose: bool = True) -> None:
        """
        Set Schechter function parameters.

        Use this method to set Schechter parameters after initialization.

        Parameters
        ----------
        schechter_params : dict
            Dictionary containing Schechter parameters:
            - 'z_nodes': ndarray of redshift samples (required)
            - 'm_star': ndarray of m*(z) values (required)
            - 'alpha': ndarray of α(z) values (optional)
            - 'phi': ndarray of φ(z) values (optional)
        verbose : bool
            Print summary statistics

        Examples
        --------
        >>> cat = RedCatalogue(colors=['gi', 'rz', 'iy'])
        >>> schechter_params = {'z_nodes': z_gal, 'm_star': mi_star}
        >>> cat.set_schechter_params(schechter_params)
        >>> cat.estimate_photoz(df_photometric)
        """
        if 'z_nodes' not in schechter_params or 'm_star' not in schechter_params:
            raise ValueError("schechter_params must contain 'z_nodes' and 'm_star' arrays")

        self.schechter_params = schechter_params

        if verbose:
            n_nodes = len(schechter_params['z_nodes'])
            z_range = f"[{schechter_params['z_nodes'][0]:.2f}, {schechter_params['z_nodes'][-1]:.2f}]"
            print(f"Set Schechter parameters: {n_nodes} redshift nodes, z = {z_range}")
            if 'alpha' in schechter_params:
                print(f"  alpha(z) included")
            if 'phi' in schechter_params:
                print(f"  phi(z) included")

    def load_offset(self, filepath: str) -> Tuple[CubicSpline, Dict]:
        """
        Load a saved offset function from disk.

        Parameters
        ----------
        filepath : str
            Full path to offset file (e.g., 'folder/offset_Euclid_Q1.npz')

        Returns
        -------
        offset_func : CubicSpline
            Fitted offset function
        metrics : dict
            Calibration metrics
        """
        self.offset_func, self.metric = load_offset_function(filepath)
        return self.offset_func, self.metric

    def save_offset(self, filepath: str, verbose: bool = True) -> None:
        """
        Save offset function to disk.

        Parameters
        ----------
        filepath : str
            Full path to output file (e.g., 'folder/offset_Euclid_Q1.npz')
        verbose : bool
            Print saving statistics

        Notes
        -----
        Requires that calibrate_photoz_offset() has been run first to compute
        the offset function and calibration metrics.
        """
        if self.offset_func is None:
            raise ValueError("No offset function to save. Run calibrate_photoz_offset first.")

        if not hasattr(self, '_offset_calibration_data'):
            raise ValueError("No calibration data available. Run calibrate_photoz_offset first.")

        data = self._offset_calibration_data
        save_offset_function(
            self.offset_func,
            data['z_nodes'],
            data['train_metrics'],
            data['val_metrics'],
            filepath,
            data.get('luminosity_diagnostics'),
            interp=data.get('offset_interp', 'cubic'),
        )

        if verbose:
            print(f"Saved offset function to {filepath}")

