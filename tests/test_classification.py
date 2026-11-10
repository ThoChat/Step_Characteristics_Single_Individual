"""Unit tests of the Unified classification on synthetic trials with known labels."""

import numpy as np
import pytest

from step_analysis.methods import classification, kinematics
from step_analysis.validation import synthetic
from step_analysis.validation import previous_code as ref


def _first_step(trial):
    feet, steps = kinematics.step_detector(trial["RTOE"], trial["LTOE"])
    return feet[0], steps[0][0], steps[0][1]


@pytest.mark.parametrize("label", list(synthetic.SCENARIOS))
def test_unified_classification_synthetic(label):
    trial = synthetic.synthetic_trial(**synthetic.SCENARIOS[label])
    foot, begin, end = _first_step(trial)
    assert foot == synthetic.SCENARIOS[label]["stepping_foot"]
    assert classification.unified_classification(trial, foot, begin, end).label == label


@pytest.mark.parametrize("label", list(synthetic.SCENARIOS))
def test_unified_classification_matches_previous(label):
    trial = synthetic.synthetic_trial(**synthetic.SCENARIOS[label])
    foot, begin, end = _first_step(trial)
    old = ref.FirstStepBlock(*(trial[k] for k in ("RTOE", "RTARI", "RHEE", "RTAR", "LTOE", "LTARI", "LHEE", "LTAR", "COM")))
    new = classification.unified_classification(trial, foot, begin, end)
    assert new.label == old["UnifiedClassifStepName"]
    assert new.crossover == old["CrossoverIndex"]


def test_crossover_index():
    ucs = synthetic.synthetic_trial(**synthetic.SCENARIOS["UCS"])
    ums = synthetic.synthetic_trial(**synthetic.SCENARIOS["UMS"])
    assert classification.crossover_index(ucs, 0, 599) == 1
    assert classification.crossover_index(ums, 0, 599) == -1


def test_classification_diagnostic_frames():
    trial = synthetic.synthetic_trial(**synthetic.SCENARIOS["FS"])
    foot, begin, end = _first_step(trial)
    res = classification.unified_classification(trial, foot, begin, end)
    assert res.front_crossing_frame is not None
    y = trial["COM"][1]
    assert y[res.front_crossing_frame] <= 0.14 <= y[res.front_crossing_frame + 1]  # toes at y = 0.14
    trial = synthetic.synthetic_trial(**synthetic.SCENARIOS["LSS"])
    foot, begin, end = _first_step(trial)
    res = classification.unified_classification(trial, foot, begin, end)
    assert res.polygon_hit_frame >= begin
