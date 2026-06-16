"""real_trial_protocol_v1: a method trial that MEASURES (baseline vs intervention on a frozen
task set, a predefined two-sided metric, a structureless negative control) and DECIDES on the
metric alone - never on a model's opinion. The pass must be EARNED: the method beats a plausible
baseline but is not preordained-perfect, and a structureless control cannot fake the gain."""

from kevin import real_trial


def test_frozen_task_set_is_content_addressed():
    ts = real_trial.example_task_set()
    sha = ts.sha()
    assert sha and ts.sha() == sha                 # stable
    # a changed case changes the sha -> drift is detectable
    from dataclasses import replace
    ts2 = replace(ts, cases=ts.cases[:-1])
    assert ts2.sha() != sha


def test_example_pass_is_earned_not_preordained():
    r = real_trial.run_example()
    # the method beats the (plausible, non-strawman) lexical baseline ...
    assert r.baseline > r.intervention
    # ... but is NOT perfect: it still mislabels the double-negation case, so the rate is > 0.
    # (this is what makes the pass earned rather than a rigged constant)
    assert r.intervention > 0.0
    assert r.passed is True and r.direction == "positive" and r.uncertainty == "low"
    assert r.evaluation_mode == "real_trial_protocol_v1" and r.epistemic_weight == "provisional"
    assert r.task_set_sha and r.repetitions == 5


def test_two_sided_metric_cannot_be_gamed_by_a_constant_label():
    # a one-sided metric let an "always duplicate" solver score a perfect 0. The two-sided
    # misclassification_rate must punish collapsing to a single label.
    ts = real_trial.example_task_set()
    always_dup = [{"label": "duplicate"} for _ in ts.cases]
    always_con = [{"label": "contradiction"} for _ in ts.cases]
    assert real_trial.misclassification_rate(always_dup, ts.cases) > 0.0
    assert real_trial.misclassification_rate(always_con, ts.cases) > 0.0


def test_decision_rests_on_the_metric_not_a_model():
    # a method identical to the baseline shows NO effect -> must NOT pass, whatever it is called
    r = real_trial.run_real_trial(
        method_id="noop", task_set=real_trial.example_task_set(),
        metric_name="misclassification_rate", metric=real_trial.misclassification_rate,
        baseline=real_trial.baseline_solver, intervention=real_trial.baseline_solver,
        negative_control=real_trial.negative_control_solver, lower_is_better=True, min_effect=0.2)
    assert r.delta == 0.0 and r.passed is False and r.direction == "none"


def test_negative_control_can_score_but_does_not_fake_the_gain():
    # the structureless hash-parity control is not a constant - it CAN get cases right by luck -
    # but on the frozen set it does not beat the baseline by the threshold, so the rig is not just
    # measuring noise. (If a sham is used AS the intervention, it must not pass.)
    r = real_trial.run_real_trial(
        method_id="sham", task_set=real_trial.example_task_set(),
        metric_name="misclassification_rate", metric=real_trial.misclassification_rate,
        baseline=real_trial.baseline_solver, intervention=real_trial.negative_control_solver,
        negative_control=real_trial.negative_control_solver, lower_is_better=True, min_effect=0.2)
    assert r.passed is False


def test_frozen_joni_conflict_cases_v1_earns_its_pass_on_real_material():
    r = real_trial.run_joni_conflict_trial()
    assert r.task_set.startswith("frozen_joni_conflict_cases@v1") and r.task_set_sha
    assert r.baseline > r.intervention > 0.0       # method helps, but is not preordained-perfect
    assert r.negative_control >= r.intervention     # the sham does not beat the real method
    assert r.passed is True and r.direction == "positive"
    assert r.epistemic_weight == "provisional"
