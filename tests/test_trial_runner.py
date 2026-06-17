"""Kevin's trials are transfer experiments on foreign tasks - real pass/fail, never promote."""

from dataclasses import dataclass, field

import pytest

from kevin import layer9_link, trial_runner

pytest.importorskip("desi_layer9")
import desi_layer9 as l9  # noqa: E402
from desi_layer9 import Operator, ProposalType, make_proposal  # noqa: E402
from desi_layer9.provenance import Provenance  # noqa: E402


@dataclass
class _M:
    id: str
    name: str
    summary: str
    origin: str = "unknown"
    applicable_to: tuple = ()
    supporting_runs: tuple = field(default_factory=tuple)
    failed_runs: tuple = field(default_factory=tuple)


# a method whose shape covers something in every task -> it should help on foreign tasks
_RICH = _M("M-1", "premortem",
           "assume failure and work backward; enumerate every risk and hazard; attack and "
           "challenge each cause; trace the causal root; check the invariant that must hold")
_THIN = _M("M-2", "thing", "thing")


def _propose(core, name, summary, *, origin="unknown", applicable_to=()):
    return layer9_link.propose_method(core, name=name, summary=summary, steps=[],
                                      affinities=applicable_to, origin=origin)


def _operator_promote(core, mid):
    core.submit(make_proposal(ProposalType.METHOD_PROPOSAL, Operator.METHOD_PROMOTE,
                payload={}, proposer="human", provenance=Provenance.from_human(),
                target_objects=(mid,)), actor="human")


def _human_validate(core, mid):
    # a Kevin-proposed method is model-origin (tainted: unverified_model_output): before it can be
    # made authoritative a HUMAN must sign off on it (clears the taint ENFORCEMENT, not the flags).
    core.submit(make_proposal(ProposalType.METHOD_PROPOSAL, Operator.HUMAN_VALIDATE,
                payload={}, proposer="human", provenance=Provenance.from_human(),
                target_objects=(mid,)), actor="human")


# -- the trial itself ------------------------------------------------------------ #

def test_a_trial_is_an_improvement_over_baseline_not_executability():
    ok, detail = trial_runner._trial(_RICH, "r1")
    assert set(detail) == {"task", "fit", "baseline", "with_method", "improvement"}
    # a pass means the method beat the baseline by the required margin on a FOREIGN task
    assert (ok is (detail["improvement"] >= trial_runner._MIN_IMPROVEMENT))
    assert detail["with_method"] == round(detail["baseline"] + detail["improvement"], 4)


def test_a_thin_method_cannot_improve_anything_and_fails_every_task():
    for run in ("r1", "r2", "r3", "r4", "r5"):
        ok, detail = trial_runner._trial(_THIN, run)
        assert ok is False and detail["fit"] == 0.0     # no shape -> no improvement -> fail


def test_the_task_lies_outside_the_method_origin_domain():
    m = _M("M-9", "red_flags_first", "rule out the catastrophic; find the cheapest test",
           origin="clinical-triage")
    for run in ("r1", "r2", "r3"):
        task, _ = trial_runner._pick_task(m, run)
        assert task != "clinical-triage"                # never trialed on its home turf


def test_method_shape_is_inferred_content_free():
    assert "risk" in trial_runner._method_shape(_RICH)
    assert "adversarial" in trial_runner._method_shape(_RICH)
    assert trial_runner._method_shape(_THIN) == set()


# -- through the shared core ----------------------------------------------------- #

def test_trials_produce_both_passes_and_failures_not_a_flawless_streak():
    core = layer9_link.new_core()
    _propose(core, "premortem", _RICH.summary)                    # rich -> can pass
    _propose(core, "react-router", "a declarative routing framework", origin="https://x")
    _propose(core, "thing", "thing")                             # thin -> must fail
    rep = trial_runner.trial_methods(core, run_id="r1")
    assert rep["trialed"] == 3
    assert rep["failed"] >= 1                                     # the 5/0 streak is broken
    assert rep["succeeded"] + rep["failed"] == 3


def test_one_trial_per_method_per_run_id():
    core = layer9_link.new_core()
    mid = _propose(core, "premortem", _RICH.summary)
    trial_runner.trial_methods(core, run_id="r1")
    trial_runner.trial_methods(core, run_id="r1")                 # same run id -> no double count
    assert core.get(mid).trial_count == 1
    trial_runner.trial_methods(core, run_id="r2")                 # a new run -> a fresh trial
    assert core.get(mid).trial_count == 2


def test_kevin_makes_a_provisional_method_activation_ready_but_does_not_promote():
    core = layer9_link.new_core()
    mid = _propose(core, "premortem", _RICH.summary)             # passes on every foreign task
    _operator_promote(core, mid)                                  # human: candidate -> provisional
    assert core.get(mid).status is l9.Status.PROVISIONAL
    for r in ("r1", "r2", "r3"):
        rep = trial_runner.trial_methods(core, run_id=r)
    assert core.get(mid).success_count >= 3                       # it earned them on foreign tasks
    assert mid in rep["activation_ready"]
    assert core.get(mid).status is l9.Status.PROVISIONAL         # ...Kevin did NOT promote it
    _human_validate(core, mid)                                  # a human signs off on the taint
    _operator_promote(core, mid)                                  # only a human can activate
    assert core.get(mid).status is l9.Status.ACTIVE


def test_max_trials_bounds_the_work():
    core = layer9_link.new_core()
    for i in range(5):
        _propose(core, f"m{i}", "a reusable method for routing problems")
    rep = trial_runner.trial_methods(core, run_id="r1", max_trials=2)
    assert rep["trialed"] == 2


def test_trial_core_file_round_trips(tmp_path):
    path = tmp_path / "layer9.json"
    core = layer9_link.new_core()
    _propose(core, "premortem", _RICH.summary)
    from desi_layer9 import persistence
    persistence.save(core, path)
    rep = trial_runner.trial_core_file(path, run_id="r1")
    assert rep["trialed"] == 1
    reloaded = persistence.load(path)
    assert reloaded.all(l9.ObjectType.METHOD)[0].trial_count == 1
