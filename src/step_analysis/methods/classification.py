"""Unified stepping strategy classification (manuscript Section 2.3.1.1, Fig. 2.12).

The classification only uses the CoM ground projection and the feet positions,
not the perturbation angle:

1. Feet crossed during the step              -> UCS (Unloaded Crossover Step)
2. CoM crossed the front BoS boundary        -> FS  (Forward Step)
   (segment between both toes at t_begin)
3. First foot polygon (at t_begin) entered by the CoM after t_begin:
   non-stepping foot -> UMS (Unloaded Medial Step)
   stepping foot     -> LSS (Loaded Side Step)
4. Otherwise                                 -> SCS (Sub-Critical Step)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import first_crossing_frame, first_frame_inside, foot_polygon


def crossover_index(trial, begin, end):
    """Detect a crossing of the feet during a step (manuscript Eq. 2.5-2.6).

    Parameters
    ----------
    trial : dict
        Output of :func:`step_analysis.data_io.load_trial` (needs RTOE, LTOE,
        RHEE, LHEE).
    begin, end : int
        Step beginning and end frames.

    Returns
    -------
    int
        ``1`` if the left-to-right toe vector or the left-to-right heel vector
        changed direction between ``begin`` and ``end`` (negative dot product),
        ``-1`` otherwise (``0`` in the degenerate case of a null dot product).

    Notes
    -----
    Same formula as ``CrossoverIndex`` in the previous ``CreateAllQtyDF``.
    """
    def _flip(right, left):
        v_begin = trial[right][:2, begin] - trial[left][:2, begin]
        v_end = trial[right][:2, end] - trial[left][:2, end]
        return -np.sign(v_begin @ v_end)

    return int(max(_flip("RTOE", "LTOE"), _flip("RHEE", "LHEE")))


@dataclass
class UnifiedResult:
    """Output of :func:`unified_classification`.

    Attributes
    ----------
    label : str
        ``"UCS"``, ``"FS"``, ``"UMS"``, ``"LSS"`` or ``"SCS"``.
    crossover : int
        Output of :func:`crossover_index`.
    front_crossing_frame : int or None
        First frame at which the CoM crosses the toes segment (FS criterion).
    polygon_hit_frame : int or None
        First frame (>= t_begin) at which the CoM enters a foot polygon.
    """

    label: str
    crossover: int
    front_crossing_frame: Optional[int] = None
    polygon_hit_frame: Optional[int] = None


def unified_classification(trial, stepping_foot, begin, end):
    """Label the first recovery step with the Unified classification.

    Parameters
    ----------
    trial : dict
        Output of :func:`step_analysis.data_io.load_trial` (feet markers + COM).
    stepping_foot : {"R", "L"}
        Foot that performs the first step.
    begin, end : int
        First step beginning (t_begin) and end (t_end) frames.

    Returns
    -------
    UnifiedResult
        Label and the frames that triggered it (see :class:`UnifiedResult`).

    Notes
    -----
    Reproduces ``UnifiedClassifStepName`` of the previous ``CreateAllQtyDF``:

    - the FS test scans the whole CoM trajectory of the trial (all frames);
    - the UMS / LSS test scans the CoM from t_begin to the end of the trial,
      the non-stepping foot polygon being tested first at each frame;
    - polygons are taken at t_begin.

    The frame-by-frame loops (``shapely`` intersection and
    ``Path.contains_point``) are replaced by vectorised equivalents.
    """
    non_stepping = "L" if stepping_foot == "R" else "R"
    crossover = crossover_index(trial, begin, end)
    if crossover == 1:
        return UnifiedResult("UCS", crossover)

    com = trial["COM"][:2]
    toes_start = trial[stepping_foot + "TOE"][:2, begin]
    toes_end = trial[non_stepping + "TOE"][:2, begin]
    front = first_crossing_frame(toes_start, toes_end, com)
    if front is not None:
        return UnifiedResult("FS", crossover, front_crossing_frame=front)

    com_after = com[:, begin:].T
    hit_nsf = first_frame_inside(foot_polygon(trial, non_stepping, begin), com_after)
    hit_sf = first_frame_inside(foot_polygon(trial, stepping_foot, begin), com_after)
    if hit_nsf is None and hit_sf is None:
        return UnifiedResult("SCS", crossover)
    # Non-stepping foot checked first at a given frame (same as the previous loop)
    if hit_sf is None or (hit_nsf is not None and hit_nsf <= hit_sf):
        return UnifiedResult("UMS", crossover, polygon_hit_frame=begin + hit_nsf)
    return UnifiedResult("LSS", crossover, polygon_hit_frame=begin + hit_sf)
