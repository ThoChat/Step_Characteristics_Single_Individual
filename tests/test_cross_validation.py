"""Cross-validation on the real data against the previous analysis.

1. New pipeline on the previous raw exports  == previous saved results (all trials).
2. New pipeline on the open data             == previous saved results, except
   for the trials whose open-data files differ (previous_results.UPDATED_TRIALS).
3. Open data vs previous raw exports differ exactly on UPDATED_TRIALS, and the
   previous version is always the shorter one.
4. New functions == previous functions (reference copies) on every open-data
   step trial (step detection, Unified label, step vector, minimal time to
   boundary, estimated perturbation angle, peak force).
"""

import numpy as np
import pytest

from step_analysis import config, data_io, pipeline
from step_analysis.validation import previous_results
from step_analysis.validation import previous_code as ref

pytestmark = [
    pytest.mark.previous,
    pytest.mark.skipif(not previous_results.previous_raw_available(), reason="previous analysis data not available"),
]
UPDATED = set(previous_results.UPDATED_TRIALS)


@pytest.fixture(scope="module")
def open_table():
    return pipeline.build_first_step_table(cache_file=None)


@pytest.fixture(scope="module")
def previous_raw_table():
    return pipeline.build_first_step_table(cache_file=None, loader=previous_results.load_previous_raw_trial)


def test_pipeline_reproduces_previous_results(previous_raw_table):
    _, summary, mismatches, sets = previous_results.compare_with_previous(previous_raw_table)
    assert (summary["n_match"] == summary["n_compared"]).all(), summary
    assert sets["missing_in_new"] == [] and sets["missing_in_previous"] == sets["no_push"]
    assert len(sets["no_push"]) == 3


def test_open_data_matches_previous_except_updated_trials(open_table):
    merged, summary, mismatches, _ = previous_results.compare_with_previous(open_table)
    for quantity, rows in mismatches.items():
        keys = set(zip(rows["participant"], rows["condition"], rows["push_folder"]))
        assert keys <= UPDATED, (quantity, keys - UPDATED)


def test_open_data_differs_only_in_updated_trials():
    diff = previous_results.compare_raw_data()
    changed = diff[~diff["identical"]]
    assert set(zip(changed["participant"], changed["condition"], changed["push_folder"])) == UPDATED
    assert (changed["n_frames_open"] > changed["n_frames_previous"]).all()


def test_new_functions_match_previous_functions_on_open_data(open_table):
    steps = open_table[open_table["step_index"] == 2]
    assert len(steps) > 600
    for row in steps.itertuples():
        trial = data_io.load_trial(row.participant, row.condition, row.push_folder)
        exception = data_io.instructed_angle(row.push_folder) == "90"  # as in GenerateStepDataframe
        old = ref.FirstStepBlock(*(trial[k] for k in ("RTOE", "RTARI", "RHEE", "RTAR", "LTOE", "LTARI", "LHEE", "LTAR", "COM")),
                                 exception=exception)
        assert old["SteppingFoot"] == row.stepping_foot
        assert old["FirstStepBegIndex"] == row.step_begin
        assert old["FirstStepEndIndex"] == row.step_end
        assert old["UnifiedClassifStepName"] == row.strategy
        assert np.allclose(old["StepVector"], [row.step_dx, row.step_dy], atol=1e-12)
        assert old["StepLength_xVcomRef"] == pytest.approx(row.step_length_x_vcom_ref, abs=1e-9)
        assert old["StepLength_yVcomRef"] == pytest.approx(row.step_length_y_vcom_ref, abs=1e-9)
        if row.step_begin <= row.perturbation_start:  # empty Ttb window (S16 NI PL45_2, step before the push)
            assert np.isnan(row.min_ttb)
            continue
        old = ref.PerturbationAngleBlock(*(trial[k] for k in ("RTOE", "LTOE", "LTAR", "LHEE", "RHEE", "RTAR", "COM",
                                                              "ForceSensor")),
                                         FirstStepBegIndex=old["FirstStepBegIndex"])
        assert old["MinTtbIndex"] == row.min_ttb_frame
        assert old["MinTtb"] == pytest.approx(row.min_ttb, abs=1e-9)
        assert old["PerturbationAngle"] == pytest.approx(row.estimated_angle_deg, abs=1e-9)
        assert old["MaxForce"] == pytest.approx(row.peak_force, abs=1e-9)
