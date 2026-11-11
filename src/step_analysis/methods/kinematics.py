"""Kinematic tools: time derivative, perturbation timing and step detection.

Functions in this module reproduce the methods of the previous analysis
(``DeCIMo-analysis/AnalysisBasics.py``) and of the manuscript (Section 2.1.2.2).
Each function names its previous counterpart in the ``Notes`` section; the
equivalence is checked in ``tests/`` and in ``notebooks/02_Analysis_Test.ipynb``.
"""

import numpy as np
from scipy.signal import savgol_filter

from .. import config


def temporal_deriv(x, dt=config.DT):
    """Second-order accurate time derivative along the last axis.

    Parameters
    ----------
    x : array_like, shape (n_frames,) or (n_dims, n_frames)
        Signal(s) sampled at constant period ``dt``. At least 3 frames.
    dt : float, optional
        Sampling period (s). Defaults to 1/200 s.

    Returns
    -------
    numpy.ndarray
        Derivative with the same shape as ``x`` (unit of ``x`` per second).
        Central differences inside, 2nd-order one-sided differences at both ends.

    Notes
    -----
    Vectorised replacement of ``TemporalDeriv``: ``np.gradient`` with
    ``edge_order=2`` uses exactly the same three finite-difference formulas,
    so results are identical up to floating point rounding (~1e-15).
    """
    return np.gradient(np.asarray(x, dtype=float), dt, axis=-1, edge_order=2)


def find_speed_peaks(curve, threshold, threshold_inflexion, epsilon=0.0, start_index=0):
    """Find the inflexion points around every peak of a speed curve.

    The highest peak is located first; the closest frames before and after it
    where the curve is both under ``threshold_inflexion`` and flat (gradient
    under ``epsilon``) are its begin/end inflexion points. The search is then
    repeated recursively on the parts of the curve on the left and on the right.

    Parameters
    ----------
    curve : array_like, shape (n_frames,)
        Speed curve (e.g. transversal toe speed, m/s).
    threshold : float
        Peaks whose maximum is below this value are ignored (same unit as ``curve``).
    threshold_inflexion : float
        Inflexion points must have a value below this threshold.
    epsilon : float, optional
        Maximal gradient between two consecutive frames for a frame to be an
        inflexion point (same unit as ``curve``). Default 0.
    start_index : int, optional
        Index offset of ``curve[0]`` in the full signal (used by the recursion).

    Returns
    -------
    list of [int, int]
        ``[[begin_1, end_1], [begin_2, end_2], ...]`` frame indices of the
        inflexion points of each peak, sorted in time. Empty list if no peak.

    Notes
    -----
    Faithful port of the recursive ``FindPeaks`` (same loop conditions,
    same tie handling with ``argmax``).
    """
    curve = np.asarray(curve, dtype=float)
    n = len(curve)
    if n < 3 or np.max(curve) < threshold:
        return []

    max_index = int(np.argmax(curve))

    # Begin inflexion point (walk backwards from the peak)
    i = max_index
    while i > 0 and (curve[i] - curve[i - 1] > epsilon or curve[i] >= threshold_inflexion):
        i -= 1
    begin = i if max_index > 0 else 0

    # End inflexion point (walk forwards from the peak)
    if max_index < n - 1:
        i = max_index
        while i < n - 1 and (curve[i] - curve[i + 1] > epsilon or curve[i] >= threshold_inflexion):
            i += 1
        end = i
    else:
        end = n - 1

    left = find_speed_peaks(curve[:begin], threshold, threshold_inflexion, epsilon, start_index)
    right = find_speed_peaks(curve[end:], threshold, threshold_inflexion, epsilon, start_index + end)
    return left + [[start_index + begin, start_index + end]] + right


def _foot_steps(toe, dt, exception, dist_threshold, abduction_threshold,
                successful_abduction_threshold, local_max_threshold, trans_vel_threshold,
                epsilon, frames_ignored):
    """Detect the steps of one foot (helper of :func:`step_detector`).

    Parameters
    ----------
    toe : numpy.ndarray, shape (3, n_frames)
        Toe marker trajectory (m).
    dt, exception, dist_threshold, ... : see :func:`step_detector`.

    Returns
    -------
    steps : list
        List of ``[begin, end]`` frame indices, or ``["Abduction"]`` if a leg
        abduction was detected for this foot. Empty if the foot did not move.
    diagnostics : dict
        ``dist`` (toe distance from its initial position, m), ``vtrans``
        (transversal toe speed, m/s) and ``peaks`` (all candidate peaks),
        all aligned with the original frame indices.
    """
    # Ignore the first/last frames and pad the start so indices match the raw frames
    ftoe = np.pad(toe[:, frames_ignored:-frames_ignored], ((0, 0), (frames_ignored, 0)), mode="edge")
    dist = np.sqrt((ftoe[0] - ftoe[0, 0]) ** 2 + (ftoe[1] - ftoe[1, 0]) ** 2)
    diagnostics = dict(dist=dist, vtrans=None, peaks=[])

    if np.max(dist) < dist_threshold:
        return [], diagnostics

    # Leg abduction: the foot moved and came back close to its initial position
    if abs(np.max(dist) - dist[-1]) > abduction_threshold and not (
        exception or dist[-1] > successful_abduction_threshold
    ):
        return ["Abduction"], diagnostics

    velocity = temporal_deriv(ftoe[:2], dt)
    vtrans = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    vmax = np.max(vtrans)
    peaks = find_speed_peaks(vtrans, local_max_threshold * vmax, trans_vel_threshold * vmax, epsilon * vmax)
    diagnostics.update(vtrans=vtrans, peaks=peaks)

    steps = []
    for begin, end in peaks:
        displacement = np.sqrt((ftoe[0, end] - ftoe[0, begin]) ** 2 + (ftoe[1, end] - ftoe[1, begin]) ** 2)
        if displacement > dist_threshold:
            steps.append([begin, end])
    return steps, diagnostics


def step_detector(rtoe, ltoe, dt=config.DT, exception=False, return_diagnostics=False,
                  **params):
    """Detect and sort the recovery steps of both feet from toe trajectories.

    Three-stage method of the manuscript (Section 2.1.2.2): (1) the toe must
    travel more than ``dist_threshold``; (2) peaks of the transversal toe speed
    are candidate steps; (3) the closest inflexion points under 10 % of the max
    speed give the step beginning (t_begin) and end (t_end). Leg abductions are
    detected when the foot comes back near its initial position.

    Parameters
    ----------
    rtoe, ltoe : numpy.ndarray, shape (3, n_frames)
        Right and left toe marker trajectories (m).
    dt : float, optional
        Sampling period (s).
    exception : bool, optional
        If True, the leg-abduction search is disabled and every motion is
        handled as a step. The previous analysis used ``exception=True`` for
        the 90 deg (posterior) perturbations.
    return_diagnostics : bool, optional
        If True, also return per-foot diagnostic signals (for plots).
    **params
        Overrides of :data:`config.STEP_DETECTION_PARAMS`
        (``dist_threshold``, ``abduction_threshold``,
        ``successful_abduction_threshold``, ``local_max_threshold``,
        ``trans_vel_threshold``, ``epsilon``, ``frames_ignored``).

    Returns
    -------
    stepping_feet : list of str
        Stepping foot of each step sorted by beginning time (``"R"``/``"L"``),
        or a single code: ``["X"]`` no step, ``["lAbd"]`` leg abduction,
        ``["lAbd+Ul"]`` leg abduction + unstable loaded leg.
    steps : list of [int, int]
        ``[begin, end]`` frame indices of each step (same order as
        ``stepping_feet``). ``[[1, -1]]`` when no step was detected.
    diagnostics : dict, optional
        Only if ``return_diagnostics``: ``{"R": {...}, "L": {...}}`` as
        returned by :func:`_foot_steps`.

    Notes
    -----
    Port of the previous ``StepDetector`` with vectorised distance and speed
    computations; detection logic, thresholds and the merge order of left and
    right steps (ties go to the left foot) are unchanged.
    """
    p = dict(config.STEP_DETECTION_PARAMS)
    p.update(params)

    right, diag_r = _foot_steps(rtoe, dt, exception, **p)
    left, diag_l = _foot_steps(ltoe, dt, exception, **p)
    diagnostics = {"R": diag_r, "L": diag_l}

    def _out(feet, steps):
        return (feet, steps, diagnostics) if return_diagnostics else (feet, steps)

    if "Abduction" in right + left:
        if len(right) == 0 or len(left) == 0:
            return _out(["lAbd"], [[1, -1]])
        return _out(["lAbd+Ul"], [[1, -1]])

    if len(right) + len(left) == 0:
        return _out(["X"], [[1, -1]])

    # Merge both feet by beginning frame (same merge as the previous code)
    feet, steps = [], []
    i_r = i_l = 0
    while i_r < len(right) and i_l < len(left):
        if right[i_r][0] < left[i_l][0]:
            feet.append("R"); steps.append(right[i_r]); i_r += 1
        else:
            feet.append("L"); steps.append(left[i_l]); i_l += 1
    for step in right[i_r:]:
        feet.append("R"); steps.append(step)
    for step in left[i_l:]:
        feet.append("L"); steps.append(step)
    return _out(feet, steps)


def step_index_code(stepping_feet):
    """Convert the step detector output into the previous ``StepIndex`` code.

    Parameters
    ----------
    stepping_feet : list of str
        First output of :func:`step_detector`.

    Returns
    -------
    float
        ``0`` no step, ``1`` leg abduction, ``1.1`` leg abduction + unstable
        loaded leg, ``2`` step (as in the previous ``GenerateStepDataframe``).
    """
    return {"X": 0.0, "lAbd": 1.0, "lAbd+Ul": 1.1}.get(stepping_feet[0], 2.0)


def perturbation_duration(force, dt=config.DT, threshold=config.PERTURBATION_THRESHOLD):
    """Find the beginning and end frames of the push from the force signal.

    Parameters
    ----------
    force : array_like, shape (n_frames,)
        Force sensor signal (N).
    dt : float, optional
        Sampling period (s).
    threshold : float, optional
        Fraction of the maximal force under which the push is considered to
        start / end. Default 0.25.

    Returns
    -------
    start, end : int
        Frame indices of the push onset (last zero of the filtered force
        derivative before the rising edge) and offset (first zero after the
        falling edge). ``(0, 1)`` if no push is found (force < 5 N).

    Notes
    -----
    Port of the previous ``ComputePerturbationDuration`` (Savitzky-Golay
    filter of window 55 and order 5 before differentiation).
    """
    force = np.asarray(force, dtype=float)
    n = len(force)
    start, end = 0, n - 1
    fmax = np.max(force)

    filtered = savgol_filter(force, 55, 5)
    deriv = temporal_deriv(filtered, dt)
    deriv = deriv / np.max(np.abs(deriv))

    i = int(np.argmax(deriv))  # rising edge
    while start == 0 and i > 0:
        if deriv[i] <= 0 and force[i] / fmax < threshold:
            start = i
        i -= 1
    i = int(np.argmin(deriv))  # falling edge
    while end == n - 1 and i < n - 1:
        if deriv[i] >= 0 and force[i] / fmax < threshold:
            end = i
        i += 1
    if np.max(force - force[start]) < 5:
        start, end = 0, 1
    return start, end


def impulse(force, start, end, dt=config.DT):
    """Perturbation impulse by trapezoidal integration of the force.

    Parameters
    ----------
    force : array_like, shape (n_frames,)
        Force sensor signal (N). The last value is used as zero offset.
    start, end : int
        Integration bounds (frame indices, ``end`` included).
    dt : float, optional
        Sampling period (s).

    Returns
    -------
    float
        Impulse (N.s).

    Notes
    -----
    Vectorised port of the previous ``ComputeImpulse``.
    """
    f = np.asarray(force, dtype=float) - force[-1]
    return float(np.sum((f[start:end] + f[start + 1:end + 1]) * 0.5) * dt)


def peak_force(force):
    """Peak perturbation force (perturbation intensity).

    Parameters
    ----------
    force : array_like, shape (n_frames,)
        Force sensor signal (N).

    Returns
    -------
    float
        ``max(force - force[-1])`` (N): maximum of the signal with the last
        value as zero offset, as ``MaxForce`` in the previous ``AllQtyDF.csv``.
        The manuscript describes the Low / Medium / High intensity levels by this
        maximal force (54 +- 12, 68 +- 13 and 88 +- 20 N, §2.1.1).
    """
    f = np.asarray(force, dtype=float)
    return float(np.max(f - f[-1]))

