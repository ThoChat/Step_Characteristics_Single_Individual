"""Analysis of the first recovery step of the DeCIMo single-individual experiment.

Package layout
--------------
config               Paths, sampling period, detection parameters.
data_io              Reading the open-source dataset.
pipeline             Per-trial processing -> one-row-per-trial first-step table.

methods/             The biomechanical methods (manuscript Sections 2.1-2.3)
    kinematics       Time derivative, perturbation timing, step detection.
    geometry         Segment intersection, foot polygons, heading, AP/ML decomposition.
    classification   Unified stepping strategy classification.

plotting/            Figures in the style of the previous work
    style            Palette, markers, fonts, legends.
    figures          Figures of the analysis and of the test report.

validation/          Checks of the method (used by tests/ and notebooks/02_Analysis_Test)
    previous_results Comparison with the previous analysis results and raw files.
    previous_code    Reference copies of the previous functions (needs shapely).
    synthetic        Synthetic trials with known answers.
"""

from . import config, data_io, methods, pipeline, plotting  # noqa: F401
