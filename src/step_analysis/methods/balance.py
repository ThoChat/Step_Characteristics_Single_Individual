"""Time to boundary (Ttb) and estimated perturbation angle.

Manuscript §2.1.2.1 (Ttb, Eq. 2.3) and §2.3.1.2 (perturbation angle, Fig. 2.13). The
*estimated* angle of perturbation is the direction of the CoM velocity at the
moment of minimal Ttb, between perturbation onset and the beginning of the first
step. It only uses kinematic information.

Vectorised versions of the previous ``BoSSegmentCrossingPoint`` loop and of the
Min Ttb / perturbation angle blocks of ``CreateAllQtyDF`` (reference copies in
:mod:`step_analysis.validation.previous_code`). All functions work in the lab
horizontal plane: +x = participant's right, +y = forward.
"""

import numpy as np

from . import geometry, kinematics

# Base of support outline of the previous code (closed polygon, both feet)
BOS_MARKERS = ("RTOE", "LTOE", "LTAR", "LHEE", "RHEE", "RTAR", "RTOE")
# Window end for trials without step: 1 s after the end of the push (frames)
NO_STEP_WINDOW_FRAMES = 200


def bos_outline(trial, frames):
    """Base of support outline at the given frames.

    Parameters
    ----------
    trial : dict
        Output of :func:`step_analysis.data_io.load_trial`.
    frames : array_like of int
        Frame indices.

    Returns
    -------
    numpy.ndarray, shape (n_frames, 7, 2)
        (x, y) of RTOE, LTOE, LTAR, LHEE, RHEE, RTAR, RTOE (m): the closed BoS
        polygon of the previous code, one per frame.
    """
    frames = np.asarray(frames, dtype=int)
    return np.stack([trial[m][:2, frames].T for m in BOS_MARKERS], axis=1)


def _cross(a, b):
    """z component of the cross product of 2D vectors (broadcast on the last axis)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _closest_points_on_segments(a, b, c):
    """Closest point to c on each segment [a, b] (previous ``closest_point_seg``).

    Parameters
    ----------
    a, b : numpy.ndarray, shape (..., 2)
        Segment end points.
    c : numpy.ndarray, shape (..., 2)
        Point (broadcastable to ``a``).

    Returns
    -------
    numpy.ndarray, shape (..., 2)
    """
    u = a - c
    v = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -np.sum(v * u, axis=-1) / np.sum(v * v, axis=-1)
    inside = (t >= 0) & (t <= 1)
    end_b_further = np.sum((u + v) ** 2, axis=-1) > np.sum(u * u, axis=-1)
    endpoint = np.where(end_b_further[..., None], a, b)
    return np.where(inside[..., None], (1 - t[..., None]) * a + t[..., None] * b, endpoint)


def boundary_crossing_points(bos, com, vcom):
    """Point of the BoS boundary the CoM is heading to, at each frame.

    Same rules as the previous ``BoSSegmentCrossingPoint`` (method "Both"):

    1. Forward: the first BoS segment, in outline order, crossed by the 1 m
       segment from the CoM along its velocity. The crossing point is returned.
       Inside a convex BoS only one segment is crossed. When the CoM is outside
       the BoS and moving towards it, two segments can be crossed and the first
       in outline order is kept, not necessarily the nearest (previous rule).
    2. Otherwise (CoM outside the BoS and moving away): among the segments
       crossed by the 1 m segment *against* the velocity, the crossing point
       closest to the CoM.
    3. Otherwise (velocity parallel to the boundary): the closest point of the
       boundary.

    Parameters
    ----------
    bos : numpy.ndarray, shape (n, k + 1, 2)
        Closed BoS outline per frame (first point repeated at the end), e.g.
        from :func:`bos_outline`.
    com : numpy.ndarray, shape (n, 2)
        CoM ground projection (m).
    vcom : numpy.ndarray, shape (n, 2)
        Horizontal CoM velocity (m/s).

    Returns
    -------
    numpy.ndarray, shape (n, 2)
        Crossing point C per frame (m). NaN where the velocity is zero.
    """
    bos, com, vcom = (np.asarray(a, dtype=float) for a in (bos, com, vcom))
    a, b = bos[:, :-1], bos[:, 1:]  # (n, k, 2) segment end points
    p = com[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (vcom / np.linalg.norm(vcom, axis=1, keepdims=True))[:, None, :]  # unit direction
        # Intersection of the segment lines with the CoM velocity line
        t = _cross(p - a, d) / _cross(b - a, d)
        on_line = a + t[..., None] * (b - a)

    forward = geometry.segments_intersect(a, b, p, p + d)
    backward = geometry.segments_intersect(a, b, p, p - d)

    n = len(com)
    rows = np.arange(n)
    out = on_line[rows, np.argmax(forward, axis=1)]

    no_forward = ~forward.any(axis=1)
    dist = np.where(backward, np.linalg.norm(on_line - p, axis=-1), np.inf)
    out = np.where(no_forward[:, None], on_line[rows, np.argmin(dist, axis=1)], out)

    parallel = no_forward & ~backward.any(axis=1)
    if parallel.any():
        closest = _closest_points_on_segments(a[parallel], b[parallel], p[parallel])
        idx = np.argmin(np.linalg.norm(closest - p[parallel], axis=-1), axis=1)
        out[parallel] = closest[np.arange(parallel.sum()), idx]
    return out


def time_to_boundary(bos, com, vcom):
    """Time to boundary of the CoM at each frame (manuscript Eq. 2.3).

    Parameters
    ----------
    bos : numpy.ndarray, shape (n, k + 1, 2)
        Closed BoS outline per frame.
    com : numpy.ndarray, shape (n, 2)
        CoM ground projection (m).
    vcom : numpy.ndarray, shape (n, 2)
        Horizontal CoM velocity (m/s).

    Returns
    -------
    ttb : numpy.ndarray, shape (n,)
        Distance to the boundary along the velocity divided by the speed (s).
        Negative when the CoM is outside the BoS and moving away from it.
        +inf where the CoM does not move (the previous code gave NaN there;
        this never happens in the real data).
    dtb : numpy.ndarray, shape (n,)
        Distance to the boundary along the velocity (m), NaN where the CoM
        does not move.
    """
    c = boundary_crossing_points(bos, com, vcom)
    speed = np.linalg.norm(vcom, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dtb = np.sum((c - com) * vcom, axis=1) / speed
        ttb = np.where(speed > 0, dtb / speed, np.inf)
    return ttb, dtb


def min_time_to_boundary(trial, start, stop, vcom=None):
    """Minimal Ttb between two frames.

    Parameters
    ----------
    trial : dict
        Output of :func:`step_analysis.data_io.load_trial`.
    start, stop : int
        Window [start, stop) in frames: perturbation onset and t_begin of the
        first step (or end of the push + 1 s for trials without step).
    vcom : numpy.ndarray, shape (2 or 3, n_frames), optional
        CoM velocity (m/s). Computed with :func:`kinematics.temporal_deriv` if
        not given.

    Returns
    -------
    min_ttb : float
        Minimal Ttb in the window (s), NaN if the window is empty or the CoM
        does not move in it.
    frame : int or None
        Frame of the minimal Ttb (first one if tied), None if ``min_ttb`` is NaN.
    """
    if stop <= start:
        return np.nan, None
    if vcom is None:
        vcom = kinematics.temporal_deriv(trial["COM"][:2])
    frames = np.arange(start, stop)
    ttb, _ = time_to_boundary(bos_outline(trial, frames), trial["COM"][:2, frames].T, vcom[:2, frames].T)
    i = int(np.argmin(ttb))
    if not np.isfinite(ttb[i]):
        return np.nan, None
    return float(ttb[i]), int(frames[i])


def velocity_angle(v):
    """Direction of a horizontal velocity, convention of the previous code.

    Parameters
    ----------
    v : array_like, shape (2,) or (n, 2)
        Horizontal velocity (lab axes).

    Returns
    -------
    float or numpy.ndarray
        Angle (deg) in (-180, 180] from the forward direction (+y): 0 forward,
        +90 towards the participant's left (-x), -90 towards the right (+x),
        180 backward. This is the convention of ``PerturbationAngle`` and
        ``GTangle`` in ``AllQtyDF.csv`` (corrected instructed angles: 90 for
        the instructed 0 deg push, -90 for the instructed 180 deg push).
    """
    v = np.asarray(v, dtype=float)
    return np.degrees(np.arctan2(-v[..., 0] + 0.0, v[..., 1]))  # + 0.0: no -0.0, so backward = +180


def lateral_angle(angle, stepping_foot):
    """Express an angle of :func:`velocity_angle` in the stepping-leg frame.

    Parameters
    ----------
    angle : float or array_like
        Angle (deg), convention of :func:`velocity_angle`.
    stepping_foot : {"R", "L"} or array_like of str
        Stepping foot of the first step.

    Returns
    -------
    float or numpy.ndarray
        Angle (deg) from the forward direction, + towards the lateral side of
        the stepping leg, - towards the other foot (medial). Same frame as
        ``ml_lateral_pct``: the sign is flipped for right-foot steps.
    """
    return -np.asarray(angle, dtype=float) * geometry.lateral_sign(stepping_foot)


def estimated_perturbation_angle(trial, start, stop, vcom=None):
    """Estimated angle of perturbation (direction of the CoM velocity at min Ttb).

    Parameters
    ----------
    trial : dict
        Output of :func:`step_analysis.data_io.load_trial`.
    start, stop : int
        Ttb window [start, stop) in frames, see :func:`min_time_to_boundary`.
    vcom : numpy.ndarray, shape (2 or 3, n_frames), optional
        CoM velocity (m/s), computed if not given.

    Returns
    -------
    dict
        ``min_ttb`` (s), ``min_ttb_frame`` (int or None) and
        ``estimated_angle_deg`` (deg, convention of :func:`velocity_angle`;
        NaN if the window is empty).
    """
    if vcom is None:
        vcom = kinematics.temporal_deriv(trial["COM"][:2])
    min_ttb, frame = min_time_to_boundary(trial, start, stop, vcom)
    angle = np.nan if frame is None else float(velocity_angle(vcom[:2, frame]))
    return dict(min_ttb=min_ttb, min_ttb_frame=frame, estimated_angle_deg=angle)
