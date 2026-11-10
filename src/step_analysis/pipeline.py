"""Per-trial processing: from raw time series to one row of first-step features.

Main entry point: :func:`build_first_step_table`, which processes every trial of
the open dataset (~10 ms per trial) and caches the result in
``outputs/tables/first_step_table.csv``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data_io
from .methods import geometry, kinematics
from .methods.classification import unified_classification

# Experimental mistake documented in the previous analysis: this push was
# delivered at 45 deg instead of 135 deg. (participant, condition, folder) ->
# (corrected push name, corrected instructed angle)
KNOWN_CORRECTIONS = {("S5", "NI", "PH135_2"): ("PH45_3", "45")}


def process_trial(participant, condition, push_folder, height, data_dir=None, loader=None):
    """Compute the first recovery step features of one trial.

    Parameters
    ----------
    participant : str
        Participant code, e.g. ``"S4"``.
    condition : str
        Sensory condition, ``"NI"`` or ``"SI"``.
    push_folder : str
        Trial folder name, e.g. ``"PH45_1"``.
    height : float
        Participant standing height (m), used for the % body height values.
    data_dir : str or Path, optional
        Root of the open data. Defaults to :data:`config.DATA_DIR`.
    loader : callable, optional
        ``loader(participant, condition, push_folder) -> dict`` returning the
        trial time series in the format of :func:`step_analysis.data_io.load_trial`. Default:
        read the open data. Used to run the same pipeline on the previous raw
        exports (:func:`step_analysis.validation.previous_results.load_previous_raw_trial`).

    Returns
    -------
    dict
        One row of the first-step table. Main keys (NaN when not applicable):

        - identification: ``participant, condition, sc_index, push_folder,
          push, angle, height``;
        - perturbation: ``perturbation_start, perturbation_end`` (frames),
          ``perturbation_duration`` (s), ``impulse`` (N.s), ``no_push`` (bool);
        - detection: ``step_index`` (0 / 1 / 1.1 / 2), ``n_steps``,
          ``stepping_foot`` ("R"/"L"/"X"), ``step_begin, step_end`` (frames),
          ``step_duration`` (s);
        - classification: ``strategy`` (NS, LA, UCS, FS, UMS, LSS, SCS),
          ``crossover``, ``front_crossing_frame``, ``polygon_hit_frame``;
        - step vector (lab axes, m): ``step_dx, step_dy, step_dist`` and the
          projections on the CoM velocity at t_begin
          ``step_length_x_vcom_ref, step_length_y_vcom_ref`` (validation only);
        - decomposition (% body height): ``ml_pct`` (+ right), ``ap_pct``
          (+ forward), ``ml_lateral_pct`` (+ lateral side of the stepping leg),
          and in the participant frames ``ap_feet_pct, ml_feet_pct,
          ap_shoulders_pct, ml_shoulders_pct``;
        - participant heading at perturbation onset (deg from lab +y):
          ``heading_feet_deg, heading_shoulders_deg``;
        - initial feet polygons at t_begin relative to the stepping toe (m, lab
          axes): ``sf_<TOE|TARI|HEE|TAR>_<x|y>`` (stepping foot) and
          ``nsf_<...>_<x|y>`` (non-stepping foot);
        - ``final_com_x, final_com_y`` (m, last frame, validation only).
    """
    if loader is None:
        trial = data_io.load_trial(participant, condition, push_folder, data_dir=data_dir)
    else:
        trial = loader(participant, condition, push_folder)
    push, angle = KNOWN_CORRECTIONS.get(
        (participant, condition, push_folder), (push_folder, data_io.instructed_angle(push_folder))
    )
    detection_angle = data_io.instructed_angle(push_folder)  # angle used at detection time
    row = dict(
        participant=participant,
        condition=condition,
        sc_index=config.CONDITIONS.index(condition),
        push_folder=push_folder,
        push=push,
        angle=angle,
        height=height,
        final_com_x=trial["COM"][0, -1],
        final_com_y=trial["COM"][1, -1],
    )

    # --- Perturbation timing ------------------------------------------------
    force = trial["ForceSensor"]
    start, end = kinematics.perturbation_duration(force)
    imp = kinematics.impulse(force, start, end)
    row.update(
        perturbation_start=start,
        perturbation_end=end,
        perturbation_duration=(end - start) * config.DT,
        impulse=imp,
        no_push=imp < config.NO_PUSH_IMPULSE,
    )

    # --- Participant heading at perturbation initiation --------------------
    ap_feet = geometry.heading_from_feet(trial, start)
    ap_shoulders = geometry.heading_from_shoulders(trial, start)
    row.update(
        heading_feet_deg=float(geometry.heading_angle(ap_feet)),
        heading_shoulders_deg=float(geometry.heading_angle(ap_shoulders)),
    )

    # --- Step detection (exception mode for 90 deg pushes, as before) -------
    feet, steps = kinematics.step_detector(trial["RTOE"], trial["LTOE"], exception=(detection_angle == "90"))
    step_index = kinematics.step_index_code(feet)
    row["step_index"] = step_index
    if step_index != 2:
        row.update(stepping_foot="X", n_steps=0, strategy="NS" if step_index == 0 else "LA")
        return row

    foot = feet[0]
    begin, end_step = steps[0]
    row.update(
        stepping_foot=foot,
        n_steps=len(steps),
        step_begin=begin,
        step_end=end_step,
        step_duration=(end_step - begin) * config.DT,
    )

    # --- Unified classification --------------------------------------------
    result = unified_classification(trial, foot, begin, end_step)
    row.update(
        strategy=result.label,
        crossover=result.crossover,
        front_crossing_frame=result.front_crossing_frame,
        polygon_hit_frame=result.polygon_hit_frame,
    )

    # --- Step vector and decompositions --------------------------------------
    vec = geometry.step_vector(trial[foot + "TOE"], begin, end_step)
    v_begin = kinematics.temporal_deriv(trial["COM"][:2])[:, begin]  # CoM velocity at t_begin
    x_ref = v_begin / np.linalg.norm(v_begin)
    y_ref = np.array([x_ref[1], -x_ref[0]])
    ap_lab, ml_lab = geometry.decompose(vec)
    ap_f, ml_f = geometry.decompose(vec, ap_feet)
    ap_s, ml_s = geometry.decompose(vec, ap_shoulders)
    pct = lambda v: float(geometry.to_percent_height(v, height))
    row.update(
        step_dx=vec[0],
        step_dy=vec[1],
        step_dist=float(np.hypot(*vec)),
        step_length_x_vcom_ref=float(vec @ x_ref),
        step_length_y_vcom_ref=float(vec @ y_ref),
        ml_pct=pct(ml_lab),
        ap_pct=pct(ap_lab),
        ml_lateral_pct=pct(ml_lab * geometry.lateral_sign(foot)),
        ap_feet_pct=pct(ap_f),
        ml_feet_pct=pct(ml_f),
        ap_shoulders_pct=pct(ap_s),
        ml_shoulders_pct=pct(ml_s),
    )

    # --- Initial feet polygons relative to the stepping toe at t_begin -----
    origin = trial[foot + "TOE"][:2, begin]
    non_stepping = "L" if foot == "R" else "R"
    for prefix, side in (("sf", foot), ("nsf", non_stepping)):
        poly = geometry.foot_polygon(trial, side, begin) - origin
        for marker, (x, y) in zip(config.FOOT_MARKERS, poly):
            row[f"{prefix}_{marker}_x"] = x
            row[f"{prefix}_{marker}_y"] = y
    return row


def build_first_step_table(trials=None, cache_file=config.FIRST_STEP_TABLE_FILE, use_cache=True,
                           data_dir=None, loader=None):
    """Process all trials and return the first-step table (one row per trial).

    Parameters
    ----------
    trials : pandas.DataFrame, optional
        Trials to process, as returned by :func:`step_analysis.data_io.list_trials`.
        Defaults to all trials of the open dataset.
    cache_file : str or Path or None, optional
        CSV file where the table is cached. If ``use_cache`` and the file exists,
        it is read instead of recomputing. Set to None to disable caching.
    use_cache : bool, optional
        Read the cache if available (default True). Use False to force a
        recomputation (the cache is then overwritten).
    data_dir : str or Path, optional
        Root of the open data. Defaults to :data:`config.DATA_DIR`.
    loader : callable, optional
        Alternative trial loader, see :func:`process_trial`.

    Returns
    -------
    pandas.DataFrame
        One row per trial, columns described in :func:`process_trial`, trials
        in the order of :func:`step_analysis.data_io.list_trials`.
    """
    if cache_file is not None and use_cache and Path(cache_file).is_file():
        return pd.read_csv(cache_file, dtype={"angle": str})

    if trials is None:
        trials = data_io.list_trials(data_dir=data_dir)
    heights = data_io.load_participants()["Height"]
    rows = [
        process_trial(t.participant, t.condition, t.push_folder, float(heights[t.participant]),
                      data_dir=data_dir, loader=loader)
        for t in trials.itertuples()
    ]
    table = pd.DataFrame(rows)
    for col in ("step_begin", "step_end", "front_crossing_frame", "polygon_hit_frame", "crossover", "n_steps"):
        table[col] = table[col].astype("Int64")
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache_file, index=False)
    return table


def select_first_steps(table, strategies=config.LATERAL_STRATEGIES):
    """Keep valid trials whose first recovery step has one of the given labels.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`build_first_step_table`.
    strategies : iterable of str, optional
        Unified labels to keep. Default ``("UMS", "LSS", "UCS")``.

    Returns
    -------
    pandas.DataFrame
        Copy of the selected rows (``no_push`` trials excluded).
    """
    keep = table["strategy"].isin(list(strategies)) & ~table["no_push"].astype(bool)
    return table.loc[keep].copy()
