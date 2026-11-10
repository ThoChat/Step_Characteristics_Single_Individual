"""Reading the DeCIMo open-source dataset.

Data layout (one folder per participant and sensory condition)::

    <DATA_DIR>/S<i>_<NI|SI>/<Push>_<rep>/CSV_export/<Marker>.csv

Each marker file is a comma-separated array of shape (3, n_frames) holding the
x, y, z lab coordinates (m) of the marker. ``ForceSensor.csv`` and ``Time.csv``
have shape (1, n_frames). Lab axes: +x = participant's right, +y = forward
(direction of sight), +z = up.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_participants(path=config.ANATOMICAL_INFO_FILE):
    """Load the anthropometric table of the participants.

    Parameters
    ----------
    path : str or Path, optional
        CSV file with columns ``Pseudo, Gender, Age, Mass, Height, Laterality``
        (comma separated). Defaults to ``data/Anatomical_Info.csv``.

    Returns
    -------
    pandas.DataFrame
        One row per participant, indexed by ``Pseudo`` (``"S0"`` ... ``"S20"``),
        file row order preserved. ``Height`` is the standing height in m and
        ``Mass`` the body mass in kg.
    """
    return pd.read_csv(path).set_index("Pseudo")


def read_push_names(path=config.PUSHES_NAMES_FILE):
    """List every trial name of one block, in the order of the previous analysis.

    Parameters
    ----------
    path : str or Path, optional
        CSV file with one header line followed by one push type per line
        (``PH0``, ``PH45``, ...). Defaults to ``data/Pushes_Names.csv``.

    Returns
    -------
    list of str
        Push names with repetition suffix: ``["PH0_1", "PH0_2", "PH45_1", ...]``
        (30 names), as built by the previous ``ReadPushes`` loop.
    """
    with open(path, "r") as file:
        lines = [line.strip() for line in file.readlines()[1:] if line.strip()]
    return [f"{push}_{rep}" for push in lines for rep in (1, 2)]


def instructed_angle(push_name):
    """Extract the instructed perturbation angle from a push name.

    Parameters
    ----------
    push_name : str
        Push name such as ``"PH135_2"`` (P + intensity H/M/L + angle + _rep).

    Returns
    -------
    str
        Instructed angle in degrees as a string, e.g. ``"135"``.
    """
    match = re.match(r"^P[HML](\d+)_\d+$", push_name)
    if match is None:
        raise ValueError(f"Unexpected push name: {push_name!r}")
    return match.group(1)


def trial_dir(participant, condition, push_name, data_dir=None):
    """Path to the ``CSV_export`` folder of one trial.

    Parameters
    ----------
    participant : str
        Participant code, e.g. ``"S4"``.
    condition : str
        Sensory condition, ``"NI"`` or ``"SI"``.
    push_name : str
        Trial (folder) name, e.g. ``"PH45_1"``.
    data_dir : str or Path, optional
        Root of the open data. Defaults to :data:`config.DATA_DIR`.

    Returns
    -------
    pathlib.Path
        ``<data_dir>/<participant>_<condition>/<push_name>/CSV_export``.
    """
    data_dir = Path(config.DATA_DIR if data_dir is None else data_dir)
    return data_dir / f"{participant}_{condition}" / push_name / "CSV_export"


def load_trial(participant, condition, push_name, markers=config.TRIAL_MARKERS, data_dir=None):
    """Load the marker / CoM / force time series of one trial.

    Parameters
    ----------
    participant : str
        Participant code, e.g. ``"S4"``.
    condition : str
        Sensory condition, ``"NI"`` or ``"SI"``.
    push_name : str
        Trial (folder) name, e.g. ``"PH45_1"``.
    markers : iterable of str, optional
        Names of the CSV files to load (without extension). Defaults to
        :data:`config.TRIAL_MARKERS` (feet, shoulders, CoM, force sensor).
    data_dir : str or Path, optional
        Root of the open data. Defaults to :data:`config.DATA_DIR`.

    Returns
    -------
    dict of str -> numpy.ndarray
        ``data[name]`` has shape (3, n_frames) for 3D quantities (m), and shape
        (n_frames,) for 1D signals (``ForceSensor`` in N, ``Time`` in s).
    """
    return load_csv_export(trial_dir(participant, condition, push_name, data_dir), markers)


def load_csv_export(folder, markers=config.TRIAL_MARKERS):
    """Load marker / CoM / force CSV files from one ``CSV_export`` folder.

    Parameters
    ----------
    folder : str or Path
        ``CSV_export`` folder of one trial (open-data or previous layout).
    markers : iterable of str, optional
        Names of the CSV files to load (without extension).

    Returns
    -------
    dict of str -> numpy.ndarray
        Same format as :func:`load_trial`.
    """
    folder = Path(folder)
    data = {}
    for name in markers:
        array = np.loadtxt(folder / f"{name}.csv", delimiter=",", ndmin=2)
        data[name] = array[0] if array.shape[0] == 1 else array
    return data


def list_trials(data_dir=None, participants=None):
    """List all trials in the order used by the previous analysis.

    The order is: sensory condition (NI, SI) > participant (file order) >
    instructed angle (0, 45, 90, 135, 180) > push name (H, M, L; rep 1, 2).

    Parameters
    ----------
    data_dir : str or Path, optional
        Root of the open data. Defaults to :data:`config.DATA_DIR`.
    participants : list of str, optional
        Participant codes to keep. Defaults to all participants of
        ``Anatomical_Info.csv``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``participant`` (str), ``condition`` ("NI"/"SI"),
        ``sc_index`` (0/1), ``push_folder`` (folder name), ``angle``
        (instructed angle as str). Only trials whose folder exists are listed.
    """
    data_dir = Path(config.DATA_DIR if data_dir is None else data_dir)
    if participants is None:
        participants = list(load_participants().index)
    pushes = read_push_names()
    rows = []
    for sc_index, condition in enumerate(config.CONDITIONS):
        for participant in participants:
            for angle in config.INSTRUCTED_ANGLES:
                for push in pushes:
                    if instructed_angle(push) != angle:
                        continue
                    if trial_dir(participant, condition, push, data_dir).is_dir():
                        rows.append((participant, condition, sc_index, push, angle))
    return pd.DataFrame(rows, columns=["participant", "condition", "sc_index", "push_folder", "angle"])
