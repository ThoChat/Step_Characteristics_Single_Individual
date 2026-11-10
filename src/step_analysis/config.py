"""Project-wide constants and paths.

All paths can be overridden with environment variables so that the code runs
on another machine without editing the source:

- ``DECIMO_PROJECT_ROOT``     : repository root (default: found from this file).
- ``DECIMO_DATA_DIR``         : folder containing ``S*_NI`` / ``S*_SI`` trial folders.
- ``DECIMO_PREVIOUS_DIR``     : folder of the previous analysis (``DeCIMo-analysis``),
                                only needed for the cross-validation.
- ``DECIMO_PREVIOUS_RAW_DIR`` : raw CSV exports used by the previous analysis,
                                only needed for the cross-validation.

Outputs: ``outputs/tables`` (cached per-trial table), ``outputs/figures`` (PDF).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# src/step_analysis/config.py -> repository root (requires an editable install: pip install -e .)
PROJECT_ROOT = Path(os.environ.get("DECIMO_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_ROOT = PROJECT_ROOT / "data"
DATA_DIR = Path(os.environ.get("DECIMO_DATA_DIR", DATA_ROOT / "DeCIMo_OpenSource_data"))
ANATOMICAL_INFO_FILE = DATA_ROOT / "Anatomical_Info.csv"
PUSHES_NAMES_FILE = DATA_ROOT / "Pushes_Names.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIRST_STEP_TABLE_FILE = TABLES_DIR / "first_step_table.csv"

# Previous analysis (used only to cross-validate this implementation)
PREVIOUS_DIR = Path(
    os.environ.get("DECIMO_PREVIOUS_DIR", "/Users/thochat/Desktop/DeCIMo-analysis")
)
# Raw CSV exports used by the previous analysis (original pseudonym folder names)
PREVIOUS_RAW_DIR = Path(
    os.environ.get(
        "DECIMO_PREVIOUS_RAW_DIR", "/Users/thochat/Desktop/Previous works/DeCIMo/Data_DeCIMo"
    )
)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
DT = 1 / 200  # Motion capture sampling period (s), Qualisys at 200 Hz

# Sensory conditions. The index is the ``SCIndex`` used in the previous analysis
# ("NA" in the previous code == "NI" in the open data).
CONDITIONS = ("NI", "SI")

# Instructed perturbation angles (deg) as written in the push names (PH0_1, ...)
INSTRUCTED_ANGLES = ("0", "45", "90", "135", "180")

# Markers loaded for each trial (CSV_export/<name>.csv)
FOOT_MARKERS = ("TOE", "TARI", "HEE", "TAR")  # order of the foot polygon vertices
TRIAL_MARKERS = (
    "RTOE", "RTARI", "RHEE", "RTAR",
    "LTOE", "LTARI", "LHEE", "LTAR",
    "RSHO", "LSHO",
    "COM",
    "ForceSensor",
)

# ---------------------------------------------------------------------------
# Step detection parameters (manuscript Section 2.1.2.2, previous StepDetector)
# ---------------------------------------------------------------------------
STEP_DETECTION_PARAMS = dict(
    dist_threshold=2e-2,  # m, minimal toe displacement to be a step
    abduction_threshold=3e-2,  # m, max(dist) - final(dist) above this => leg abduction
    successful_abduction_threshold=5e-2,  # m, final dist below this => successful abduction
    local_max_threshold=0.3,  # fraction of max transversal speed for a peak to count
    trans_vel_threshold=0.1,  # fraction of max transversal speed for inflexion points
    epsilon=0.01,  # fraction of max transversal speed, accepted gradient at inflexion
    frames_ignored=50,  # frames ignored at the beginning and end of the trial
)

# Perturbation detection (previous ComputePerturbationDuration / ComputeImpulse)
PERTURBATION_THRESHOLD = 0.25  # fraction of max force under which the push starts/ends
NO_PUSH_IMPULSE = 1.0  # N.s, trials with a smaller impulse are discarded ("No push")

# Window used to estimate the participant heading at perturbation initiation
HEADING_WINDOW_FRAMES = 20  # 0.1 s ending at the perturbation onset

# Stepping strategies studied in this project (Unified classification)
LATERAL_STRATEGIES = ("UMS", "LSS", "UCS")
