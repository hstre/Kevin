"""real_trial_protocol_v1: a method trial that MEASURES (baseline vs intervention on a frozen
task set, a predefined metric, a negative control) and DECIDES on the metric alone - never on a
model's opinion. The pass/fail must move only with the measured effect."""

from kevin import real_trial


def test_frozen_task_set_is_content_addressed():
    ts = real_trial.example_task_set()
    sha = ts.sha()
    assert sha and ts.sha() == sha                 # stable
    # a changed case changes the sha -> drift is detectable
    from dataclasses import replace
    ts2 = replace(ts, cases=ts.cases[:-1])
    assert ts2.sha() != sha


def test_example_method_passes_because_it_measurably_helps():
    r = real_trial.run_example()
    # the contradiction-first method removes the false contradictions the baseline makes
    assert r.baseline > r.intervention              # lower_is_better: intervention is better
    assert r.intervention == 0.0                     # no duplicate mislabelled as contradiction
    assert r.passed is True and r.direction == "positive" and r.uncertainty == "low"
    assert r.evaluation_mode == "real_trial_protocol_v1" and r.epistemic_weight == "provisional"
    assert r.task_set_sha and r.repetitions == 5     # provenance present


def test_decision_rests_on_the_metric_not_a_model():
    # a method identical to the baseline shows NO effect -> must NOT pass, whatever it is called
    r = real_trial.run_real_trial(
        method_id="noop", task_set=real_trial.example_task_set(),
        metric_name="false_contradiction_rate", metric=real_trial.false_contradiction_rate,
        baseline=real_trial.baseline_solver, intervention=real_trial.baseline_solver,
        negative_control=real_trial.negative_control_solver, lower_is_better=True, min_effect=0.34)
    assert r.delta == 0.0 and r.passed is False and r.direction == "none"


def test_negative_control_guards_against_measuring_noise():
    # if the "method" is the sham negative control itself, even a delta is inconclusive: the
    # control shows the same effect, so passed must be False (high uncertainty).
    r = real_trial.run_real_trial(
        method_id="sham", task_set=real_trial.example_task_set(),
        metric_name="false_contradiction_rate", metric=real_trial.false_contradiction_rate,
        baseline=real_trial.baseline_solver, intervention=real_trial.negative_control_solver,
        negative_control=real_trial.negative_control_solver, lower_is_better=True, min_effect=0.34)
    assert r.passed is False                          # control not clean -> never a pass
