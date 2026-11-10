"""Unit tests of step_analysis.methods.kinematics against the previous functions and synthetic data."""

import numpy as np
import pytest

from step_analysis import config
from step_analysis.methods import kinematics
from step_analysis.validation import synthetic
from step_analysis.validation import previous_code as ref

RNG = np.random.default_rng(0)


def _bumpy_curve(n=800, n_bumps=4, noise=0.01):
    """Random positive curve made of Gaussian bumps (speed-like signal)."""
    t = np.arange(n)
    curve = np.zeros(n)
    for _ in range(n_bumps):
        curve += RNG.uniform(0.2, 1.5) * np.exp(-0.5 * ((t - RNG.uniform(0, n)) / RNG.uniform(5, 40)) ** 2)
    return np.abs(curve + noise * RNG.standard_normal(n))


@pytest.mark.parametrize("shape", [(500,), (3, 500)])
def test_temporal_deriv_matches_previous(shape):
    x = RNG.standard_normal(shape).cumsum(axis=-1)
    assert np.allclose(kinematics.temporal_deriv(x), ref.TemporalDeriv(x, config.DT), rtol=1e-12, atol=1e-9)


def test_temporal_deriv_accuracy():
    t = np.arange(400) * config.DT
    x = np.sin(2 * np.pi * 2 * t)
    exact = 2 * np.pi * 2 * np.cos(2 * np.pi * 2 * t)
    assert np.max(np.abs(kinematics.temporal_deriv(x) - exact)) < 1e-2 * np.max(np.abs(exact))


def test_find_speed_peaks_matches_previous_on_random_curves():
    for _ in range(300):
        curve = _bumpy_curve()
        vmax = curve.max()
        args = (0.3 * vmax, 0.1 * vmax, 0.01 * vmax)
        assert kinematics.find_speed_peaks(curve, *args) == ref.FindPeaks(list(curve), *args)


def test_find_speed_peaks_two_bells():
    t = np.arange(600)
    curve = np.exp(-0.5 * ((t - 150) / 15) ** 2) + 0.8 * np.exp(-0.5 * ((t - 400) / 20) ** 2)
    peaks = kinematics.find_speed_peaks(curve, 0.3, 0.1, 0.01)
    assert len(peaks) == 2
    (b1, e1), (b2, e2) = peaks
    assert b1 < 150 < e1 < b2 < 400 < e2


def _all_detector_cases():
    cases = {name: synthetic.synthetic_trial(**kw) for name, kw in synthetic.SCENARIOS.items()}
    cases["no_step"] = synthetic.synthetic_trial()
    cases["abduction"] = synthetic.synthetic_trial("R", (0.15, 0.0), abduction=True)
    return cases


@pytest.mark.parametrize("exception", [False, True])
@pytest.mark.parametrize("case", list(_all_detector_cases()))
def test_step_detector_matches_previous(case, exception):
    trial = _all_detector_cases()[case]
    new = kinematics.step_detector(trial["RTOE"], trial["LTOE"], exception=exception)
    old = ref.StepDetector(trial["RTOE"], trial["LTOE"], config.DT, exception=exception)
    assert new[0] == old[0]
    assert [list(map(int, s)) for s in new[1]] == [list(map(int, s)) for s in old[1]]


def test_step_detector_timing_on_minimum_jerk_step():
    begin, end = 300, 360
    trial = synthetic.synthetic_trial("R", (0.0, 0.3), step_frames=(begin, end))
    feet, steps = kinematics.step_detector(trial["RTOE"], trial["LTOE"])
    assert feet == ["R"] and len(steps) == 1
    up, down = synthetic.threshold_crossing(begin, end)
    t_begin, t_end = steps[0]
    assert begin - 1 <= t_begin <= up + 1
    assert down - 1 <= t_end <= end + 1


def test_step_detector_special_codes():
    cases = _all_detector_cases()
    assert kinematics.step_detector(cases["no_step"]["RTOE"], cases["no_step"]["LTOE"])[0] == ["X"]
    abd = cases["abduction"]
    assert kinematics.step_detector(abd["RTOE"], abd["LTOE"])[0] == ["lAbd"]
    # exception mode (used for 90 deg pushes) handles the abduction as steps
    assert kinematics.step_detector(abd["RTOE"], abd["LTOE"], exception=True)[0][0] == "R"
    assert [kinematics.step_index_code(c) for c in (["X"], ["lAbd"], ["lAbd+Ul"], ["R", "L"])] == [0, 1, 1.1, 2]


def test_perturbation_duration_and_impulse():
    trial = synthetic.synthetic_trial(push_frames=(150, 260), push_peak=80.0)
    force = trial["ForceSensor"] + 0.01 * RNG.standard_normal(600)
    start, end = kinematics.perturbation_duration(force)
    assert (start, end) == ref.ComputePerturbationDuration(force)
    assert abs(start - 150) <= 15 and abs(end - 260) <= 15
    # Impulse of A sin^2 over T seconds = A T / 2
    clean = trial["ForceSensor"]
    assert kinematics.impulse(clean, 150, 260) == pytest.approx(80.0 * 110 * config.DT / 2, rel=1e-6)
    assert kinematics.impulse(force, start, end) == pytest.approx(ref.ComputeImpulse(force, start, end, config.DT), abs=1e-10)


def test_no_push_detection():
    assert kinematics.perturbation_duration(np.full(600, 0.5) + 0.1 * RNG.standard_normal(600)) == (0, 1)
