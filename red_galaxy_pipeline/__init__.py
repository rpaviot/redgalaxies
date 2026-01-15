"""
Red Galaxy Pipeline

A unified pipeline for identifying and characterizing red-sequence galaxies
using photometric data, based on the methodology of Vakili et al. (2019).

Main components:
- RedCatalogue: Main class orchestrating the full pipeline
- GMM selection: Extreme Deconvolution for red sequence identification
- Red sequence fitting: Spline fitting for color evolution a(z), b(z), c(z)
- Photo-z estimation: Photometric redshift with Schechter function priors
"""

# Core class
from red_galaxy_pipeline.RedCatalogue import RedCatalogue

# Utils - functions used by multiple modules
from red_galaxy_pipeline.utils import (
    parse_colors,
    create_magnitude_reference,
    create_color_arrays_and_covariance
)

# GMM Selection
from red_galaxy_pipeline.gmm_selection import RedSequenceSelector

# Red Sequence Fitting
from red_galaxy_pipeline.red_sequence_fitting import (
    RedSequenceFitter,
    save_params,
    load_params,
    make_spline_functions
)

# Photo-z Estimation
from red_galaxy_pipeline.photoz_estimation import (
    PhotoZFitter,
    PhotoZEstimator,
    SchechterFunction,
    create_schechter_from_params,
    fit_redshift_offset,
    bias_and_scatter,
    save_offset_function,
    load_offset_function,
    compute_luminosity_diagnostics
)

__version__ = '1.0.0'
__all__ = [
    'RedCatalogue',
    # utils
    'parse_colors',
    'create_magnitude_reference',
    'create_color_arrays_and_covariance',
    # gmm
    'RedSequenceSelector',
    # fitting
    'RedSequenceFitter',
    'save_params',
    'load_params',
    'make_spline_functions',
    # photoz
    'PhotoZFitter',
    'PhotoZEstimator',
    'SchechterFunction',
    'create_schechter_from_params',
    'fit_redshift_offset',
    'bias_and_scatter',
    'save_offset_function',
    'load_offset_function',
    'compute_luminosity_diagnostics',
]
