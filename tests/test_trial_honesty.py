"""Architecture invariant: the keyword-shape trial is a SIMULATION, and says so. Its numbers must
never be presentable as method effectiveness - every report carries the honesty tags."""

import desi_layer9 as l9

from kevin import layer9_link, trial_runner


def _shelf():
    core = l9.Layer9()
    from desi_layer9 import Operator, ProposalType, make_proposal
    from desi_layer9.provenance import Provenance
    for i in range(3):
        core.submit(make_proposal(
            ProposalType.METHOD_PROPOSAL, Operator.METHOD_PROPOSE,
            payload={"name": f"m{i}", "summary": "a decomposition method", "steps": [],
                     "origin": "joni:emergent", "applicable_to": ["x"]},
            proposer="joni", provenance=Provenance.from_operator()), actor="kevin")
    return core


def test_method_trial_is_tagged_synthetic_not_effectiveness():
    core = _shelf()
    rep = trial_runner.trial_methods(core, run_id="r1")
    # METHOD_TRIAL_NOT_MOCK_IN_PRODUCTION (negative form): if it IS the mock, it must declare it.
    assert rep["evaluation_mode"] == "synthetic_mock"
    assert rep["epistemic_weight"] == "none"
    assert trial_runner.EVALUATION_MODE == "synthetic_mock"


def test_trial_still_records_through_the_gate_only():
    # Kevin reports trials; it never promotes. (governance boundary intact)
    core = _shelf()
    if not layer9_link.available():
        return
    rep = trial_runner.trial_methods(core, run_id="r1")
    assert rep["trialed"] >= 1
    from desi_layer9 import ObjectType, Status
    assert all(m.status in (Status.CANDIDATE, Status.PROVISIONAL)
               for m in core.all(ObjectType.METHOD))   # never auto-promoted to ACTIVE
