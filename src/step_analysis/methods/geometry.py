"""Planar geometry: segment intersection, foot polygons, heading and step decomposition.

All functions work in the horizontal plane (x, y) of the lab: +x = participant's
right, +y = forward. Vectors are numpy arrays of shape (2,) or (n, 2).
"""

import numpy as np
from matplotlib.path import Path

from .. import config


# ---------------------------------------------------------------------------
# Segment intersection (replaces shapely.LineString.intersects)
# ---------------------------------------------------------------------------
def _orientation(p, q, r):
    """Orientation of the ordered triplets (p, q, r).

    Parameters
    ----------
    p, q, r : numpy.ndarray, shape (..., 2)
        Points (broadcastable).

    Returns
    -------
    numpy.ndarray of float
        ``0`` collinear, ``1`` clockwise, ``-1`` counter-clockwise (sign of the
        cross product, same convention as the previous ``orientation``).
    """
    val = (q[..., 1] - p[..., 1]) * (r[..., 0] - q[..., 0]) - (q[..., 0] - p[..., 0]) * (r[..., 1] - q[..., 1])
    return np.sign(val)


def _on_segment(p, q, r):
    """For collinear p, q, r: is q inside the bounding box of segment [p, r]?

    Parameters
    ----------
    p, q, r : numpy.ndarray, shape (..., 2)
        Points (broadcastable).

    Returns
    -------
    numpy.ndarray of bool
    """
    return (
        (q[..., 0] <= np.maximum(p[..., 0], r[..., 0]))
        & (q[..., 0] >= np.minimum(p[..., 0], r[..., 0]))
        & (q[..., 1] <= np.maximum(p[..., 1], r[..., 1]))
        & (q[..., 1] >= np.minimum(p[..., 1], r[..., 1]))
    )


def segments_intersect(p1, q1, p2, q2):
    """Test whether segments [p1, q1] and [p2, q2] intersect (touching included).

    Parameters
    ----------
    p1, q1 : array_like, shape (2,) or (n, 2)
        End points of the first segment(s).
    p2, q2 : array_like, shape (2,) or (n, 2)
        End points of the second segment(s). Shapes are broadcast, so one fixed
        segment can be tested against n segments at once.

    Returns
    -------
    numpy.ndarray of bool, shape () or (n,)
        True where the segments share at least one point.

    Notes
    -----
    Vectorised version of the previous ``doIntersect`` (orientation test with
    collinear special cases). It reproduces ``shapely`` ``LineString.intersects``
    used in the previous Unified classification (checked in the tests).
    """
    p1, q1, p2, q2 = (np.asarray(a, dtype=float) for a in (p1, q1, p2, q2))
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    general = (o1 != o2) & (o3 != o4)
    special = (
        ((o1 == 0) & _on_segment(p1, p2, q1))
        | ((o2 == 0) & _on_segment(p1, q2, q1))
        | ((o3 == 0) & _on_segment(p2, p1, q2))
        | ((o4 == 0) & _on_segment(p2, q1, q2))
    )
    return general | special


def first_crossing_frame(segment_start, segment_end, trajectory):
    """First frame at which a 2D trajectory crosses a fixed segment.

    Parameters
    ----------
    segment_start, segment_end : array_like, shape (2,)
        End points of the fixed segment (m).
    trajectory : array_like, shape (2, n_frames)
        Trajectory (x, y) in time (m), e.g. the CoM ground projection.

    Returns
    -------
    int or None
        Index ``t`` of the first trajectory piece [t, t+1] intersecting the
        segment, or None if it never does.
    """
    traj = np.asarray(trajectory, dtype=float)[:2].T
    hits = segments_intersect(segment_start, segment_end, traj[:-1], traj[1:])
    idx = np.flatnonzero(hits)
    return int(idx[0]) if idx.size else None


# ---------------------------------------------------------------------------
# Foot polygons
# ---------------------------------------------------------------------------
def foot_polygon(trial, side, frame):
    """Polygon of one foot at a given frame.

    Parameters
    ----------
    trial : dict
        Output of :func:`step_analysis.data_io.load_trial`.
    side : {"R", "L"}
        Foot side.
    frame : int
        Frame index.

    Returns
    -------
    numpy.ndarray, shape (4, 2)
        (x, y) of the TOE, TARI, HEE and TAR markers (m): the foot polygon of
        manuscript Fig. 2.7, vertex order of the previous code.
    """
    return np.array([trial[side + m][:2, frame] for m in config.FOOT_MARKERS])


def first_frame_inside(polygon, points):
    """First point of a sequence that lies inside a polygon.

    Parameters
    ----------
    polygon : array_like, shape (n_vertices, 2)
        Polygon vertices (implicitly closed).
    points : array_like, shape (n_points, 2)
        Points to test, in time order.

    Returns
    -------
    int or None
        Index of the first point inside the polygon, None if none is.

    Notes
    -----
    ``Path.contains_points`` is the vectorised form of the
    ``Path.contains_point`` loop of the previous Unified classification.
    """
    inside = Path(np.asarray(polygon, dtype=float)).contains_points(np.asarray(points, dtype=float))
    idx = np.flatnonzero(inside)
    return int(idx[0]) if idx.size else None


# ---------------------------------------------------------------------------
# Step vector, participant heading and decomposition
# ---------------------------------------------------------------------------
def step_vector(toe, begin, end):
    """Horizontal step vector of the stepping toe marker.

    Parameters
    ----------
    toe : numpy.ndarray, shape (3, n_frames)
        Stepping toe marker trajectory (m).
    begin, end : int
        Step beginning (t_begin) and end (t_end) frames.

    Returns
    -------
    numpy.ndarray, shape (2,)
        (dx, dy) displacement of the toe between t_begin and t_end (m), in lab
        axes (manuscript Section 2.3.1.2, Fig. 2.14).
    """
    return toe[:2, end] - toe[:2, begin]


def _window(frame, n_frames):
    """Frame slice ending at ``frame`` used to average a posture.

    Parameters
    ----------
    frame : int
        Last frame of the window (perturbation onset).
    n_frames : int
        Window length (frames).

    Returns
    -------
    slice
        ``slice(max(0, frame - n_frames), frame + 1)``.
    """
    return slice(max(0, frame - n_frames), frame + 1)


def heading_from_feet(trial, frame, n_frames=config.HEADING_WINDOW_FRAMES):
    """Forward direction of the participant estimated from the feet.

    Parameters
    ----------
    trial : dict
        Output of :func:`step_analysis.data_io.load_trial`.
    frame : int
        Reference frame (e.g. perturbation onset).
    n_frames : int, optional
        Number of frames before ``frame`` over which positions are averaged.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Unit vector from the mid-heels to the mid-toes point.
    """
    w = _window(frame, n_frames)
    toes = (trial["RTOE"][:2, w] + trial["LTOE"][:2, w]).mean(axis=1) / 2
    heels = (trial["RHEE"][:2, w] + trial["LHEE"][:2, w]).mean(axis=1) / 2
    v = toes - heels
    return v / np.linalg.norm(v)


def heading_from_shoulders(trial, frame, n_frames=config.HEADING_WINDOW_FRAMES):
    """Forward direction of the participant estimated from the shoulders.

    Parameters
    ----------
    trial : dict
        Output of :func:`step_analysis.data_io.load_trial`.
    frame : int
        Reference frame (e.g. perturbation onset).
    n_frames : int, optional
        Number of frames before ``frame`` over which positions are averaged.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Unit horizontal normal to the left-to-right shoulder vector, pointing in
        the direction of sight (the reference used for angles in the manuscript).
    """
    w = _window(frame, n_frames)
    v = (trial["RSHO"][:2, w] - trial["LSHO"][:2, w]).mean(axis=1)
    normal = np.array([-v[1], v[0]])  # rotate left->right vector by +90 deg
    return normal / np.linalg.norm(normal)


def heading_angle(ap_axis):
    """Angle between a forward direction and the lab +y axis.

    Parameters
    ----------
    ap_axis : array_like, shape (2,) or (n, 2)
        Forward (anteroposterior) direction(s), not necessarily unit.

    Returns
    -------
    float or numpy.ndarray
        Angle in degrees, in (-180, 180]; positive when the direction is turned
        towards +x (participant's right, clockwise seen from above).
    """
    ap_axis = np.asarray(ap_axis, dtype=float)
    return np.degrees(np.arctan2(ap_axis[..., 0], ap_axis[..., 1]))


def decompose(vector, ap_axis=(0.0, 1.0)):
    """Decompose horizontal vector(s) on the anteroposterior / mediolateral axes.

    Parameters
    ----------
    vector : array_like, shape (2,) or (n, 2)
        Horizontal vector(s) (x, y) in lab coordinates (m).
    ap_axis : array_like, shape (2,) or (n, 2), optional
        Forward (AP) unit direction(s). The ML axis is the AP axis rotated by
        -90 deg so that +ML points to the participant's right. Default
        ``(0, 1)``: lab axes (AP = y, ML = x).

    Returns
    -------
    ap, ml : float or numpy.ndarray
        Components along the AP (+ forward) and ML (+ right) axes (m). The
        decomposition is a rotation: ``ap**2 + ml**2 == |vector|**2``.
    """
    v = np.asarray(vector, dtype=float)
    u = np.asarray(ap_axis, dtype=float)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    ml_axis = np.stack([u[..., 1], -u[..., 0]], axis=-1)
    return np.sum(v * u, axis=-1), np.sum(v * ml_axis, axis=-1)


def lateral_sign(stepping_foot):
    """Sign that maps the anatomical ML axis to the stepping-leg ML axis.

    Parameters
    ----------
    stepping_foot : {"R", "L"} or array_like of str

    Returns
    -------
    int or numpy.ndarray of int
        ``+1`` for a right-foot step, ``-1`` for a left-foot step. Multiplying the
        anatomical ML component (+ right) by this sign gives a component that is
        positive towards the lateral side of the stepping leg and negative
        towards the medial side (towards the other foot).
    """
    return np.where(np.asarray(stepping_foot) == "R", 1, -1)


def to_percent_height(value, height):
    """Express a length as a percentage of body height.

    Parameters
    ----------
    value : float or array_like
        Length(s) in m.
    height : float or array_like
        Standing height(s) in m (``Anatomical_Info.csv``).

    Returns
    -------
    float or numpy.ndarray
        ``100 * value / height`` (% body height).
    """
    return 100.0 * np.asarray(value, dtype=float) / np.asarray(height, dtype=float)
