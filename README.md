# Red Galaxy Pipeline

Red sequence galaxy identification and photometric redshift estimation based on [Vakili et al. (2019)](https://arxiv.org/abs/1909.11736).

## Installation

```bash
pip install -e .
```

## Overview

This pipeline implements:

1. **GMM Selection** (Section 3.2): Two-stage Gaussian Mixture Model selection using Extreme Deconvolution
2. **Ridge Line Fitting** (Section 3.3): Red sequence color-magnitude relation fitting
3. **Photo-z Estimation** (Section 3.4-3.7): Bayesian photometric redshifts with Schechter luminosity function priors

## Color Model

The red sequence color is modeled as:

```
color(z, m) = a(z) + b(z) * (m - m_ref(z)) with an intrinsic scatter in colour C_int(z).
```

Where:
- `a(z)`: Zero-point color evolution (cubic spline)
- `b(z)`: Color-magnitude slope evolution (cubic spline)
- `c(z)`: Intrinsic scatter evolution (cubic spline)
- `m_ref(z)`: Reference magnitude (median magnitude per redshift bin)

## Modules

### `gmm_selection.py`

Two-stage GMM red sequence selection:

- **Stage 1**: 2D GMM in (magnitude, first_color) space identifies the red component
- **Stage 2**: N-D GMM in multi-color space refines the selection

Key class: `RedSequenceSelector`

### `red_sequence_fitting.py`

Fits the ridge line parameters a(z), b(z), c(z) using cubic spline interpolation over redshift nodes.

Key functions:
- `fit_red_sequence()`: Joint fit across all colors
- `fit_red_sequence_single_colour()`: Independent fits per color (more stable)
- `save_params()` / `load_params()`: Save/load fitted parameters

### `photoz_estimation.py`

Bayesian photo-z estimation with priors:
- Schechter luminosity function p(m|z)
- Comoving volume p(z)

**Optimization methods:**
- `iminuit` (default): Fast local minimization
- `differential_evolution`: Robust global optimization (slower but avoids local minima)

**Parallel processing:**
```python
df_result = cat.estimate_photoz(df, n_jobs=-1)  # Use all cores
```

### `RedCatalogue.py`

High-level interface combining all pipeline stages.

## Schechter Function

The Schechter function is specified via arrays sampled at redshift nodes:

**Required:**
- `z_nodes`: Redshift sampling array
- `m_star`: Characteristic magnitude m*(z)

**Optional (default to constants):**
- `alpha`: Faint-end slope, default -1.0
- `phi`: Normalization, default 1.0

## Offset Calibration

To minimize systematic photo-z bias:

1. **Calibrate** on spec-z sample using `calibrate_photoz_offset()`
2. **Apply** to photometric catalog with `apply_offset=True`

The offset function is a cubic spline fit to minimize |z_spec - z_photo|.

## Output Format

Ridge line parameters are saved as `.npz` files:

```
colors          # List of color names ['gi', 'rz', 'iy']
nodes_a         # Redshift nodes for a(z) - stored once
nodes_b         # Redshift nodes for b(z) - stored once
nodes_c         # Redshift nodes for c(z) - stored once
{color}_a_values, {color}_a_errors  # Per-color a(z) values
{color}_b_values, {color}_b_errors  # Per-color b(z) values
{color}_c_values, {color}_c_errors  # Per-color c(z) values
m_ref_z, m_ref_values              # Reference magnitude spline
```

## References

Vakili, M., et al. (2019). "A photometric survey of red sequence galaxies for the Vera C. Rubin Observatory Legacy Survey of Space and Time." MNRAS, 487, 3160-3176.
