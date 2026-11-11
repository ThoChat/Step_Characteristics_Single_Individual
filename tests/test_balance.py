"""Unit tests of step_analysis.methods.balance (time to boundary, estimated perturbation angle)."""

import numpy as np
import pytest

from step_analysis import pipeline
from step_analysis.methods import balance, kinematics
from step_analysis.validation import previous_code as ref
from step_analysis.validation import synthetic

RNG = np.random.default_rng(2)

# Rectangle BoS x in [-0.15, 0.15], y in [-0.1, 0.2], closed outline
RECTANGLE = np.array([[0.15, 0.2], [-0.15, 0.2], [-0.15, -0.1], [0.15, -0.1], [0.15, 0.2]])


@pytest.mark.parametrize(
    "com, v, expected",
    [
        ((0.0, 0.0), (0.3, 0.0), 0.15 / 0.3),  # inside, to the right
        ((0.0, 0.0), (0.0, 0.5), 0.2 / 0.5),  # inside, forward
        ((0.0, 0.0), (-0.2, -0.2), 0.1 / 0.2),  # inside, diagonal: hits the back edge first
        ((0.25, 0.0), (0.5, 0.0), -0.1 / 0.5),  # outside, moving away: closest crossing behind
        # outside, moving towards the BoS: two edges crossed, the first in outline
        # order (far left edge) is kept, as in the previous code
        ((0.25, 0.0), (-0.5, 0.0), 0.4 / 0.5),
    ],
)
def test_time_to_boundary_known_values(com, v, expected):
    ttb, dtb = balance.time_to_boundary(RECTANGLE[None], np.array([com]), np.array([v]))
    assert ttb[0] == pytest.approx(expected, abs=1e-12)
    assert dtb[0] == pytest.approx(expected * np.hypot(*v), abs=1e-12)


def test_crossing_point_parallel_fallback():
    """Outside the BoS with the velocity parallel to the boundary: closest boundary point."""
    c = balance.boundary_crossing_points(RECTANGLE[None], np.array([[0.3, 0.05]]), np.array([[0.0, 1.0]]))
    assert np.allclose(c[0], [0.15, 0.05])


def test_crossing_points_match_previous_function():
    base = balance.bos_outline(synthetic.synthetic_trial(), [0])[0]
    for _ in range(2000):
        bos = base + np.r_[RNG.normal(0, 0.02, (6, 2)), np.zeros((1, 2))]
        bos[-1] = bos[0]
        com, v = RNG.uniform([-0.35, -0.3], [0.35, 0.4]), RNG.normal(0, 0.5, 2)
        old = ref.BoSSegmentCrossingPoint(bos[:, 0], bos[:, 1], com, v)
        new = balance.boundary_crossing_points(bos[None], com[None], v[None])[0]
        assert np.allclose(old, new, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "v, expected",
    [((0, 1), 0), ((-1, 0), 90), ((1, 0), -90), ((-1, 1), 45), ((1, -1), -135), ((0, -1), 180)],
)
def test_velocity_angle_convention(v, expected):
    assert balance.velocity_angle(v) == pytest.approx(expected)


def test_velocity_angle_matches_previous_formula():
    c90, s90 = np.cos(np.pi / 2), np.sin(np.pi / 2)
    rot = np.array([[c90, s90, 0], [-s90, c90, 0], [0, 0, 1]])
    vel = RNG.normal(0, 0.5, (3000, 3))
    nr = np.array([ref.ChangeOfBasis(x, rot) for x in vel])
    old = np.array([np.sign(r[1]) * np.arccos(r[0] / ref.norm(r[:2])) * 180 / np.pi for r in nr])
    assert np.allclose(balance.velocity_angle(vel[:, :2]), old, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "angle, foot, expected",
    [(90, "R", -90), (90, "L", 90), (-45, "R", 45), (-45, "L", -45), (0, "R", 0)],
)
def test_lateral_angle(angle, foot, expected):
    """+ = lateral side of the stepping leg: a push to the left is medial for a right-foot step."""
    assert balance.lateral_angle(angle, foot) == pytest.approx(expected)


@pytest.mark.parametrize("label", list(synthetic.SCENARIOS))
def test_estimated_angle_on_synthetic_trials(label):
    """The CoM moves in a straight line: the estimated angle is its direction, and the
    pipeline matches the previous Ttb block."""
    kw = synthetic.SCENARIOS[label]
    trial = synthetic.synthetic_trial(**kw, com_frames=(140, 450))  # CoM moving during the whole window
    row = pipeline.process_trial("S0", "NI", "PH45_1", 1.75, loader=lambda *_: trial)
    assert row["estimated_angle_deg"] == pytest.approx(float(balance.velocity_angle(kw["com_end"])), abs=1e-9)
    old = ref.PerturbationAngleBlock(*(trial[k] for k in ("RTOE", "LTOE", "LTAR", "LHEE", "RHEE", "RTAR", "COM",
                                                          "ForceSensor")), FirstStepBegIndex=row["step_begin"])
    assert old["MinTtbIndex"] == row["min_ttb_frame"]
    assert old["MinTtb"] == pytest.approx(row["min_ttb"], abs=1e-9)
    assert old["PerturbationAngle"] == pytest.approx(row["estimated_angle_deg"], abs=1e-9)
    assert old["MaxForce"] == pytest.approx(row["peak_force"], abs=1e-12)


def test_static_com_frames_are_ignored():
    """Frames where the CoM does not move have an infinite Ttb (NaN in the previous code)."""
    trial = synthetic.synthetic_trial(**synthetic.SCENARIOS["UMS"])  # CoM still until frame 200
    out = balance.estimated_perturbation_angle(trial, 150, 300)
    assert out["min_ttb_frame"] > 200 and out["estimated_angle_deg"] == pytest.approx(90)
    still = balance.estimated_perturbation_angle(trial, 150, 190)
    assert still["min_ttb_frame"] is None and np.isnan(still["estimated_angle_deg"])


def test_peak_force_uses_last_value_as_zero():
    force = np.r_[np.zeros(10), np.linspace(0, 50, 20), np.full(10, 3.0)]
    assert kinematics.peak_force(force) == pytest.approx(47.0)


def test_empty_window_gives_nan():
    trial = synthetic.synthetic_trial()
    out = balance.estimated_perturbation_angle(trial, 200, 200)
    assert np.isnan(out["min_ttb"]) and out["min_ttb_frame"] is None and np.isnan(out["estimated_angle_deg"])
