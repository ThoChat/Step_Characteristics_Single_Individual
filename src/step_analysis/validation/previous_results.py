"""Cross-validation against the results of the previous analysis.

The previous analysis (``DeCIMo-analysis``) saved its per-trial results in
``Saved_Data/AllQtyDF.csv`` (1257 trials) and ``Saved_Data/StepDataframe.csv``
(1260 trials) using the original participant pseudonyms. The open data uses
codes ``S0``...``S20`` in the same order. The mapping is rebuilt at run time from
the previous ``Anat_infos.csv`` (and checked on the anthropometric values); the
previous pseudonyms are dropped right after the mapping and never returned.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, data_io

# Suffix of the sensory condition in the previous folder names
PREVIOUS_CONDITION_SUFFIX = {"NI": "NA", "SI": "SI"}

# Trials whose open-data CSV files differ from the raw exports used by the
# previous analysis (found with :func:`compare_raw_data`). In all of them the
# previous version is shorter (re-processed and trimmed in August 2023); in
# S16 NI PL45_2 and S20 NI PL0_2 the force signal was also re-aligned.
# Keys: (participant, condition, push_folder).
UPDATED_TRIALS = [
    ("S2", "NI", "PH45_1"), ("S2", "NI", "PM90_1"), ("S3", "NI", "PH135_1"),
    ("S5", "NI", "PM0_1"), ("S5", "NI", "PH180_2"), ("S9", "NI", "PM0_2"),
    ("S10", "NI", "PH45_1"), ("S11", "NI", "PH135_1"), ("S12", "NI", "PH90_1"),
    ("S15", "NI", "PH180_1"), ("S16", "NI", "PL45_2"), ("S20", "NI", "PH0_2"),
    ("S20", "NI", "PL0_2"), ("S20", "NI", "PH90_1"), ("S20", "NI", "PM90_2"),
    ("S2", "SI", "PM90_1"), ("S5", "SI", "PH0_1"), ("S5", "SI", "PH0_2"),
    ("S8", "SI", "PL180_2"), ("S11", "SI", "PH135_2"), ("S13", "SI", "PM0_1"),
    ("S14", "SI", "PH0_2"), ("S14", "SI", "PH45_1"), ("S15", "SI", "PM0_2"),
    ("S15", "SI", "PH180_1"), ("S15", "SI", "PM180_1"),
]

ANTHROPOMETRIC_COLUMNS = ["Gender", "Age", "Mass", "Height", "Laterality"]

# (new column, previous column, kind). kind: "exact" or "float"
COMPARED_QUANTITIES = [
    ("step_index", "StepIndex", "exact"),
    ("stepping_foot", "SteppingFoot", "exact"),
    ("step_begin", "FirstStepBegIndex", "exact"),
    ("strategy", "UnifiedClassifStepName", "exact"),
    ("step_duration", "FirstStepDuration", "float"),
    ("step_dist", "FirstStepDist", "float"),
    ("step_length_x_vcom_ref", "StepLength_xVcomRef", "float"),
    ("step_length_y_vcom_ref", "StepLength_yVcomRef", "float"),
    ("impulse", "Impulse", "float"),
    ("perturbation_duration", "PerturbationDuration", "float"),
    ("final_com_x", "Final_CoM_x", "float"),
    ("final_com_y", "Final_CoM_y", "float"),
]
STEP_ONLY = {"step_begin", "strategy", "step_duration", "step_dist",
             "step_length_x_vcom_ref", "step_length_y_vcom_ref"}


def previous_available(previous_dir=config.PREVIOUS_DIR):
    """Check that the previous results needed for validation are reachable.

    Parameters
    ----------
    previous_dir : str or Path, optional
        Folder of the previous analysis.

    Returns
    -------
    bool
        True if ``Anat_infos.csv``, ``Saved_Data/AllQtyDF.csv`` and
        ``Saved_Data/StepDataframe.csv`` all exist.
    """
    d = Path(previous_dir)
    return all((d / f).is_file() for f in ("Anat_infos.csv", "Saved_Data/AllQtyDF.csv", "Saved_Data/StepDataframe.csv"))


def pseudonym_mapping(previous_dir=config.PREVIOUS_DIR):
    """Map the previous pseudonyms to the open-data participant codes.

    Parameters
    ----------
    previous_dir : str or Path, optional
        Folder of the previous analysis (contains ``Anat_infos.csv``).

    Returns
    -------
    dict of str -> str
        ``{previous_pseudonym: "S<i>"}``. Kept in memory only.

    Raises
    ------
    ValueError
        If the two anthropometric tables do not match row by row.
    """
    previous = pd.read_csv(Path(previous_dir) / "Anat_infos.csv", sep=";")
    current = data_io.load_participants().reset_index()
    same = len(previous) == len(current) and np.allclose(
        previous[["Age", "Mass", "Height", "Laterality"]].to_numpy(float),
        current[["Age", "Mass", "Height", "Laterality"]].to_numpy(float),
    ) and (previous["Gender"].to_numpy() == current["Gender"].to_numpy()).all()
    if not same:
        raise ValueError("Previous and open-data anthropometric tables do not match row by row.")
    return dict(zip(previous["Pseudo"], current["Pseudo"]))


@lru_cache(maxsize=4)
def _code_to_pseudonym(previous_dir):
    """Inverse of :func:`pseudonym_mapping` (cached, kept in memory only).

    Parameters
    ----------
    previous_dir : str
        Folder of the previous analysis.

    Returns
    -------
    dict of str -> str
        ``{"S<i>": previous_pseudonym}``.
    """
    return {code: pseudo for pseudo, code in pseudonym_mapping(previous_dir).items()}


def previous_raw_available(raw_dir=config.PREVIOUS_RAW_DIR, previous_dir=config.PREVIOUS_DIR):
    """Check that the previous raw CSV exports can be read.

    Parameters
    ----------
    raw_dir : str or Path, optional
        Root folder of the previous raw data (``Data_DeCIMo``).
    previous_dir : str or Path, optional
        Folder of the previous analysis (for the pseudonym mapping).

    Returns
    -------
    bool
    """
    return previous_available(previous_dir) and Path(raw_dir).is_dir()


def previous_trial_dir(participant, condition, push_folder, raw_dir=config.PREVIOUS_RAW_DIR,
                       previous_dir=config.PREVIOUS_DIR):
    """``CSV_export`` folder of a trial in the previous raw data layout.

    Parameters
    ----------
    participant : str
        Open-data participant code, e.g. ``"S4"``.
    condition : str
        ``"NI"`` or ``"SI"``.
    push_folder : str
        Trial folder name, e.g. ``"PH45_1"``.
    raw_dir, previous_dir : str or Path, optional
        See :func:`previous_raw_available`.

    Returns
    -------
    pathlib.Path
        ``<raw_dir>/<pseudonym><NA|SI>/Mocap/<push_folder>/CSV_export``. The path
        contains the previous pseudonym: do not write it to saved outputs.
    """
    pseudo = _code_to_pseudonym(str(previous_dir))[participant]
    return Path(raw_dir) / f"{pseudo}{PREVIOUS_CONDITION_SUFFIX[condition]}" / "Mocap" / push_folder / "CSV_export"


def load_previous_raw_trial(participant, condition, push_folder, markers=config.TRIAL_MARKERS):
    """Load one trial from the raw CSV exports used by the previous analysis.

    Parameters
    ----------
    participant : str
        Open-data participant code, e.g. ``"S4"``.
    condition : str
        ``"NI"`` or ``"SI"``.
    push_folder : str
        Trial folder name.
    markers : iterable of str, optional
        CSV files to load.

    Returns
    -------
    dict of str -> numpy.ndarray
        Same format as :func:`step_analysis.data_io.load_trial`. Can be passed as
        ``loader`` to :func:`step_analysis.pipeline.build_first_step_table`.
    """
    return data_io.load_csv_export(previous_trial_dir(participant, condition, push_folder), markers)


def compare_raw_data(trials=None, files=("COM", "RTOE", "LTOE", "ForceSensor")):
    """Compare the open-data CSV files with the previous raw CSV exports.

    Parameters
    ----------
    trials : pandas.DataFrame, optional
        Trials to compare (output of :func:`step_analysis.data_io.list_trials`). Default: all.
    files : iterable of str, optional
        CSV files compared.

    Returns
    -------
    pandas.DataFrame
        One row per trial: ``participant, condition, push_folder``,
        ``n_frames_open``, ``n_frames_previous`` and, for each file,
        ``<file>_max_diff``: maximal absolute difference over the frames
        present in both versions (unit of the file). ``identical`` is True when
        both versions have the same length and all differences are 0.
    """
    if trials is None:
        trials = data_io.list_trials()
    rows = []
    for t in trials.itertuples():
        open_trial = data_io.load_trial(t.participant, t.condition, t.push_folder, markers=files)
        prev_trial = load_previous_raw_trial(t.participant, t.condition, t.push_folder, markers=files)
        row = dict(participant=t.participant, condition=t.condition, push_folder=t.push_folder,
                   n_frames_open=open_trial["COM"].shape[-1], n_frames_previous=prev_trial["COM"].shape[-1])
        for f in files:
            a, b = np.atleast_2d(open_trial[f]), np.atleast_2d(prev_trial[f])
            n = min(a.shape[1], b.shape[1])
            row[f"{f}_max_diff"] = float(np.max(np.abs(a[:, :n] - b[:, :n])))
        rows.append(row)
    out = pd.DataFrame(rows)
    diff_cols = [f"{f}_max_diff" for f in files]
    out["identical"] = (out["n_frames_open"] == out["n_frames_previous"]) & (out[diff_cols] == 0).all(axis=1)
    return out


def load_previous_results(previous_dir=config.PREVIOUS_DIR):
    """Load the previous per-trial results with open-data participant codes.

    Parameters
    ----------
    previous_dir : str or Path, optional
        Folder of the previous analysis.

    Returns
    -------
    all_qty : pandas.DataFrame
        ``AllQtyDF.csv`` with columns ``participant``, ``condition`` and ``push``
        (corrected push name) added; previous pseudonyms removed.
    step_df : pandas.DataFrame
        ``StepDataframe.csv`` with ``participant``, ``condition`` and
        ``push_folder`` added; previous pseudonyms removed.
    """
    d = Path(previous_dir)
    mapping = pseudonym_mapping(d)
    all_qty = pd.read_csv(d / "Saved_Data" / "AllQtyDF.csv", sep=";")
    step_df = pd.read_csv(d / "Saved_Data" / "StepDataframe.csv", sep=";")
    for df in (all_qty, step_df):
        df["participant"] = df.pop("Pseudo").map(mapping)
        df["condition"] = df["SCIndex"].map(dict(enumerate(config.CONDITIONS)))
    all_qty = all_qty.rename(columns={"PushName": "push"})
    step_df = step_df.rename(columns={"PushName": "push_folder"})
    return all_qty, step_df


def compare_with_previous(table, previous_dir=config.PREVIOUS_DIR, atol=1e-9):
    """Compare the new first-step table with the previous results.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`step_analysis.pipeline.build_first_step_table`.
    previous_dir : str or Path, optional
        Folder of the previous analysis.
    atol : float, optional
        Absolute tolerance for float quantities (same units as the columns).

    Returns
    -------
    merged : pandas.DataFrame
        New table joined with the previous values (columns ``prev_<name>``).
    summary : pandas.DataFrame
        One row per compared quantity: ``quantity``, ``kind``, ``n_compared``,
        ``n_match``, ``match_pct``, ``max_abs_diff`` (float quantities).
    mismatches : dict of str -> pandas.DataFrame
        Rows that do not match, per quantity (empty frames if all match).
    trial_sets : dict
        ``missing_in_previous`` / ``missing_in_new`` trial keys of AllQtyDF and
        ``no_push`` trials, to check that the 3 discarded trials are the same.
    """
    all_qty, step_df = load_previous_results(previous_dir)
    keys = ["participant", "condition", "push"]

    prev = all_qty.set_index(keys)
    prev = prev.rename(columns={c: f"prev_{c}" for c in prev.columns})
    merged = table.set_index(keys).join(prev, how="left")
    step_prev = step_df.set_index(["participant", "condition", "push_folder"])["StepIndex"]
    merged["prev_StepIndex"] = [step_prev.get((p, c, f), np.nan) for (p, c, _), f in zip(merged.index, merged["push_folder"])]
    merged = merged.reset_index()

    rows, mismatches = [], {}
    for new, old, kind in COMPARED_QUANTITIES:
        prev_col = f"prev_{old}"
        mask = merged[prev_col].notna()
        if new in STEP_ONLY:
            mask &= (merged["step_index"] == 2) & (merged["prev_StepIndex"] == 2)
        sub = merged.loc[mask]
        if kind == "exact":
            a, b = sub[new], sub[prev_col]
            if new in ("step_index", "step_begin"):
                a, b = a.astype(float), b.astype(float)
            ok = (a.to_numpy() == b.to_numpy())
            max_diff = np.nan
        else:
            diff = np.abs(sub[new].astype(float).to_numpy() - sub[prev_col].astype(float).to_numpy())
            ok = diff <= atol
            max_diff = float(diff.max()) if len(diff) else np.nan
        rows.append(dict(quantity=new, previous=old, kind=kind, n_compared=int(mask.sum()),
                         n_match=int(ok.sum()), match_pct=100 * ok.mean() if len(ok) else np.nan,
                         max_abs_diff=max_diff))
        mismatches[new] = sub.loc[~ok, keys + ["push_folder", new, prev_col]]

    new_keys = set(map(tuple, table[keys].to_numpy()))
    old_keys = set(map(tuple, all_qty[keys].to_numpy()))
    trial_sets = dict(
        missing_in_previous=sorted(new_keys - old_keys),
        missing_in_new=sorted(old_keys - new_keys),
        no_push=sorted(map(tuple, table.loc[table["no_push"].astype(bool), keys].to_numpy())),
    )
    return merged, pd.DataFrame(rows), mismatches, trial_sets
