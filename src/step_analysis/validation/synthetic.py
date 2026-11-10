"""Synthetic trials with known answers, used by the tests and the test notebook.

A synthetic trial has the same format as :func:`step_analysis.data_io.load_trial`
(lab axes: +x right, +y forward). Both feet start side by side; one foot can
perform a minimum-jerk step, the CoM follows a minimum-jerk path and the force
sensor records a bell-shaped push.
"""

import numpy as np

from .. import config

# Initial (x, y) of the right-foot markers; the left foot is the mirror image (x -> -x)
RIGHT_FOOT = {"TOE": (0.10, 0.14), "TARI": (0.06, 0.06), "HEE": (0.10, -0.13), "TAR": (0.14, 0.04)}
FOOT_HEIGHT = {"TOE": 0.06, "TARI": 0.04, "HEE": 0.05, "TAR": 0.04}


def minimum_jerk(n_frames, begin, end):
    """Minimum-jerk transition profile from 0 to 1.

    Parameters
    ----------
    n_frames : int
        Number of frames.
    begin, end : int
        Frames where the motion starts and ends (begin < end).

    Returns
    -------
    numpy.ndarray, shape (n_frames,)
        0 before ``begin``, 1 after ``end`` and ``10 s^3 - 15 s^4 + 6 s^5``
        in between (s = normalised time). Its speed is a single bell.
    """
    s = np.clip((np.arange(n_frames) - begin) / (end - begin), 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def threshold_crossing(begin, end, fraction=0.1):
    """Frames where a minimum-jerk speed bell crosses ``fraction`` of its peak.

    The step detector stops at the first frame (walking away from the peak)
    that is both under 10 % of the peak speed and flat (gradient < epsilon).
    For a minimum-jerk motion its t_begin therefore lies between the true
    motion onset ``begin`` and the first output of this function, and its
    t_end between the second output and ``end``.

    Parameters
    ----------
    begin, end : int
        Motion start and end frames of :func:`minimum_jerk`.
    fraction : float, optional
        Fraction of the peak speed (0.1 = inflexion threshold of the detector).

    Returns
    -------
    (float, float)
        Frames of the rising and falling crossings of ``fraction * peak``.
    """
    # speed ~ 30 s^2 (1 - s)^2, peak 1.875 at s = 0.5
    s = (1 - np.sqrt(1 - 4 * np.sqrt(fraction * 1.875 / 30))) / 2
    return begin + s * (end - begin), end - s * (end - begin)


def synthetic_trial(stepping_foot=None, displacement=(0.0, 0.0), com_end=(0.0, 0.0), n_frames=600,
                    step_frames=(300, 360), com_frames=(200, 450), push_frames=(150, 260),
                    push_peak=80.0, heading_deg=0.0, abduction=False):
    """Build a synthetic trial.

    Parameters
    ----------
    stepping_foot : {"R", "L", None}, optional
        Foot that moves (None: no step).
    displacement : (float, float), optional
        Horizontal displacement of the stepping foot (m, lab axes).
    com_end : (float, float), optional
        Final CoM ground position (m); the CoM starts at (0, 0).
    n_frames : int, optional
        Trial length (frames, 200 Hz).
    step_frames : (int, int), optional
        Start and end of the foot motion (frames).
    com_frames : (int, int), optional
        Start and end of the CoM motion (frames).
    push_frames : (int, int), optional
        Start and end of the push (frames).
    push_peak : float, optional
        Maximal push force (N).
    heading_deg : float, optional
        Rotation of the whole body posture around z (deg, positive = turned
        towards +x), to test heading estimation.
    abduction : bool, optional
        If True, the stepping foot goes out and comes back to its initial
        position (leg abduction).

    Returns
    -------
    dict of str -> numpy.ndarray
        Same keys and shapes as :func:`step_analysis.data_io.load_trial`
        (``RTOE`` ... ``LTAR``, ``RSHO``, ``LSHO``, ``COM`` (3, n_frames) and
        ``ForceSensor`` (n_frames,)).
    """
    theta = np.radians(heading_deg)
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])  # clockwise
    profile = minimum_jerk(n_frames, *step_frames)
    if abduction:
        mid = (step_frames[0] + step_frames[1]) // 2
        profile = minimum_jerk(n_frames, step_frames[0], mid) - minimum_jerk(n_frames, mid, step_frames[1])
    trial = {}
    for side, sx in (("R", 1.0), ("L", -1.0)):
        for marker, (x, y) in RIGHT_FOOT.items():
            xy = rot @ np.array([sx * x, y])
            pos = np.tile(np.array([xy[0], xy[1], FOOT_HEIGHT[marker]])[:, None], (1, n_frames))
            if side == stepping_foot:
                pos[0] += displacement[0] * profile
                pos[1] += displacement[1] * profile
            trial[side + marker] = pos
        sho = rot @ np.array([sx * 0.18, 0.0])
        trial[side + "SHO"] = np.tile(np.array([sho[0], sho[1], 1.45])[:, None], (1, n_frames))
    com_profile = minimum_jerk(n_frames, *com_frames)
    trial["COM"] = np.vstack([com_end[0] * com_profile, com_end[1] * com_profile, np.full(n_frames, 1.0)])
    t = np.arange(n_frames)
    s = np.clip((t - push_frames[0]) / (push_frames[1] - push_frames[0]), 0, 1)
    trial["ForceSensor"] = push_peak * np.sin(np.pi * s) ** 2
    return trial


# Scenarios of the Unified classification with their expected label.
# The perturbation pushes the CoM to the left (-x) except for FS.
SCENARIOS = {
    "UMS": dict(stepping_foot="R", displacement=(-0.15, 0.0), com_end=(-0.10, 0.0)),
    "LSS": dict(stepping_foot="L", displacement=(-0.25, 0.0), com_end=(-0.10, 0.0)),
    "UCS": dict(stepping_foot="R", displacement=(-0.35, 0.10), com_end=(-0.10, 0.0)),
    "FS": dict(stepping_foot="R", displacement=(0.0, 0.30), com_end=(0.0, 0.30)),
    "SCS": dict(stepping_foot="R", displacement=(-0.04, 0.0), com_end=(-0.03, 0.0)),
}
