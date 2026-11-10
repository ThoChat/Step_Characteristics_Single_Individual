"""Unit tests of step_analysis.methods.geometry."""

import numpy as np
import pytest
from matplotlib.path import Path
from shapely.geometry import LineString

from step_analysis.methods import geometry
from step_analysis.validation import synthetic

RNG = np.random.default_rng(1)


def _random_segments(n, grid=None):
    """n random non-degenerate segments; on an integer grid if ``grid`` (many collinear/touching cases)."""
    pts = RNG.integers(0, grid, size=(n, 4)).astype(float) if grid else RNG.uniform(0, 1, size=(n, 4))
    keep = np.hypot(pts[:, 2] - pts[:, 0], pts[:, 3] - pts[:, 1]) > 0
    return pts[keep]


@pytest.mark.parametrize("grid", [None, 4])
def test_segments_intersect_matches_shapely(grid):
    a, b = _random_segments(6000, grid), _random_segments(6000, grid)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    new = geometry.segments_intersect(a[:, :2], a[:, 2:], b[:, :2], b[:, 2:])
    old = np.array([LineString([p[:2], p[2:]]).intersects(LineString([q[:2], q[2:]])) for p, q in zip(a, b)])
    assert (new == old).all()


def test_segments_intersect_special_cases():
    s = lambda *c: np.array(c, dtype=float)
    assert geometry.segments_intersect(s(0, 0), s(1, 0), s(1, 0), s(2, 1))  # touching end point
    assert geometry.segments_intersect(s(0, 0), s(2, 0), s(1, 0), s(3, 0))  # collinear overlap
    assert not geometry.segments_intersect(s(0, 0), s(1, 0), s(2, 0), s(3, 0))  # collinear disjoint
    assert not geometry.segments_intersect(s(0, 0), s(1, 0), s(0, 1), s(1, 1))  # parallel


def test_first_crossing_frame():
    traj = np.vstack([np.zeros(100), np.linspace(-1, 1, 100)])  # moves forward along y
    frame = geometry.first_crossing_frame((-0.5, 0.0), (0.5, 0.0), traj)
    assert traj[1, frame] <= 0 <= traj[1, frame + 1]
    assert geometry.first_crossing_frame((-0.5, 2.0), (0.5, 2.0), traj) is None


def test_first_frame_inside_matches_contains_point_loop():
    for _ in range(50):
        poly = np.array(list(synthetic.RIGHT_FOOT.values())) + RNG.normal(0, 0.01, (4, 2))
        pts = RNG.uniform(-0.2, 0.3, size=(400, 2))
        loop = next((i for i, p in enumerate(pts) if Path(poly).contains_point(p)), None)
        assert geometry.first_frame_inside(poly, pts) == loop


def test_decompose_lab_axes_and_rotation():
    v = RNG.normal(size=(200, 2))
    ap, ml = geometry.decompose(v)
    assert np.allclose(ap, v[:, 1]) and np.allclose(ml, v[:, 0])
    theta = np.radians(RNG.uniform(-30, 30, 200))
    u = np.stack([np.sin(theta), np.cos(theta)], axis=1)  # AP axis turned by theta towards +x
    ap, ml = geometry.decompose(v, u)
    assert np.allclose(ap**2 + ml**2, (v**2).sum(axis=1))  # rotation preserves length
    ap_u, ml_u = geometry.decompose(u, u)
    assert np.allclose(ap_u, 1) and np.allclose(ml_u, 0)
    ml_axis = np.stack([u[:, 1], -u[:, 0]], axis=1)
    ap_m, ml_m = geometry.decompose(ml_axis, u)
    assert np.allclose(ap_m, 0) and np.allclose(ml_m, 1)
    assert np.allclose(geometry.heading_angle(u), np.degrees(theta))


@pytest.mark.parametrize("heading", [-12.0, 0.0, 7.0])
def test_heading_estimators_on_synthetic_posture(heading):
    trial = synthetic.synthetic_trial(heading_deg=heading)
    assert geometry.heading_angle(geometry.heading_from_feet(trial, 100)) == pytest.approx(heading, abs=1e-9)
    assert geometry.heading_angle(geometry.heading_from_shoulders(trial, 100)) == pytest.approx(heading, abs=1e-9)


def test_small_helpers():
    assert geometry.heading_angle([1, 1]) == pytest.approx(45)
    assert geometry.heading_angle([-1, 1]) == pytest.approx(-45)
    assert list(geometry.lateral_sign(["R", "L"])) == [1, -1]
    assert geometry.to_percent_height(0.17, 1.70) == pytest.approx(10.0)
    toe = np.array([[0.0, 0.1, 0.3], [0.0, 0.05, 0.2], [0, 0, 0]])
    assert np.allclose(geometry.step_vector(toe, 0, 2), [0.3, 0.2])
