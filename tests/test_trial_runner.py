"""Kevin trials the shared shelf's methods and records outcomes - but never promotes."""

import pytest

from kevin import layer9_link, trial_runner

pytest.importorskip("desi_layer9")
import desi_layer9 as l9  # noqa: E402
from desi_layer9 import Operator, ProposalType, make_proposal  # noqa: E402
from desi_layer9.provenance import Provenance  # noqa: E402


def _candidate(core, name, summary, *, applicable_to=("routing",)):
    return layer9_link.propose_method(core, name=name, summary=summary, steps=[],
                                      affinities=applicable_to)


def _operator_promote(core, mid):
    core.submit(make_proposal(ProposalType.METHOD_PROPOSAL, Operator.METHOD_PROMOTE,
                payload={}, proposer="human", provenance=Provenance.from_human(),
                target_objects=(mid,)), actor="human")


def test_a_method_shaped_candidate_passes_its_trial():
    core = layer9_link.new_core()
    mid = _candidate(core, "react-router", "a declarative routing framework for the web")
    rep = trial_runner.trial_methods(core, run_id="r1")
    assert rep["trialed"] == 1 and rep["succeeded"] == 1
    m = core.get(mid)
    assert m.trial_count == 1 and m.success_count == 1
    assert m.status is l9.Status.CANDIDATE          # still a candidate - Kevin never promotes


def test_a_thin_finding_fails_its_trial():
    core = layer9_link.new_core()
    # no domain, no shape-language, barely any words: nothing to transfer.
    mid = layer9_link.propose_method(core, name="thing", summary="thing", steps=[],
                                     affinities=())
    rep = trial_runner.trial_methods(core, run_id="r1")
    assert rep["failed"] == 1
    assert core.get(mid).failure_count == 1


def test_one_trial_per_method_per_run_id():
    core = layer9_link.new_core()
    mid = _candidate(core, "five_whys", "a technique to chase a cause to its root")
    trial_runner.trial_methods(core, run_id="r1")
    trial_runner.trial_methods(core, run_id="r1")          # same run id -> no double count
    assert core.get(mid).trial_count == 1
    trial_runner.trial_methods(core, run_id="r2")          # a new run -> a fresh trial
    assert core.get(mid).trial_count == 2


def test_kevin_makes_a_provisional_method_activation_ready_but_does_not_promote():
    core = layer9_link.new_core()
    mid = _candidate(core, "premortem", "a strategy: assume failure and work backward")
    _operator_promote(core, mid)                            # human: candidate -> provisional
    assert core.get(mid).status is l9.Status.PROVISIONAL

    for r in ("r1", "r2", "r3"):
        rep = trial_runner.trial_methods(core, run_id=r)
    assert mid in rep["activation_ready"]                   # earned, but...
    assert core.get(mid).status is l9.Status.PROVISIONAL    # ...Kevin did NOT promote it

    # only the human can flip provisional -> active, and now the trials justify it
    _operator_promote(core, mid)
    assert core.get(mid).status is l9.Status.ACTIVE
    assert mid in [m.id for m in layer9_link.usable_methods(core)]


def test_max_trials_bounds_the_work():
    core = layer9_link.new_core()
    for i in range(5):
        _candidate(core, f"m{i}", "a reusable method for routing problems")
    rep = trial_runner.trial_methods(core, run_id="r1", max_trials=2)
    assert rep["trialed"] == 2


def test_trial_core_file_round_trips(tmp_path):
    path = tmp_path / "layer9.json"
    core = layer9_link.new_core()
    _candidate(core, "react-router", "a declarative routing framework")
    from desi_layer9 import persistence
    persistence.save(core, path)

    rep = trial_runner.trial_core_file(path, run_id="r1")
    assert rep["trialed"] == 1
    reloaded = persistence.load(path)                       # the trial persisted
    m = reloaded.all(l9.ObjectType.METHOD)[0]
    assert m.trial_count == 1
