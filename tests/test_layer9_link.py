"""PR 5 - Kevin proposes methods into the one core; it may never promote."""

import pytest

from kevin import layer9_link

pytest.importorskip("desi_layer9")
import desi_layer9 as l9  # noqa: E402


def test_kevin_method_lands_as_candidate_not_authoritative():
    core = layer9_link.new_core()
    mid = layer9_link.propose_method(core, name="red_flags_first", summary="s",
                                     steps=["a", "b"], origin="medicine")
    m = core.get(mid)
    assert m.status is l9.Status.CANDIDATE                 # not active, not authoritative
    assert m.authority is not l9.Authority.AUTHORITATIVE
    assert layer9_link.usable_methods(core) == []           # candidates aren't usable yet


def test_kevin_cannot_promote_its_own_method():
    core = layer9_link.new_core()
    mid = layer9_link.propose_method(core, name="m", summary="s", steps=["a"])
    # Kevin (model origin) attempts to promote -> the gate refuses.
    from desi_layer9 import Operator, ProposalType, make_proposal
    from desi_layer9.provenance import Provenance
    d = core.submit(make_proposal(ProposalType.METHOD_PROPOSAL, Operator.METHOD_PROMOTE,
                    payload={}, proposer="kevin",
                    provenance=Provenance.from_model(external=False), target_objects=(mid,)),
                    actor="kevin")
    assert not d.accepted
    assert core.get(mid).status is l9.Status.CANDIDATE


def test_kevin_reports_trials_then_an_operator_promotes():
    core = layer9_link.new_core()
    mid = layer9_link.propose_method(core, name="m", summary="s", steps=["a"])
    from desi_layer9 import Operator, ProposalType, make_proposal
    from desi_layer9.provenance import Provenance

    def operator_proposal(op, payload):
        return make_proposal(ProposalType.METHOD_PROPOSAL, op, payload=payload,
                             proposer="human", provenance=Provenance.from_human(),
                             target_objects=(mid,))

    # a human gate first moves it candidate -> provisional
    core.submit(operator_proposal(Operator.METHOD_PROMOTE, {}))
    assert core.get(mid).status is l9.Status.PROVISIONAL

    # Kevin records successful trials
    for _ in range(3):
        layer9_link.record_trial(core, mid, success=True, run_id="r")

    # Kevin's method is model-origin (tainted: unverified_model_output), so a HUMAN must sign off on
    # it before it may be made authoritative (the contamination flags stay on record)
    core.submit(operator_proposal(Operator.HUMAN_VALIDATE, {}))

    # only now can an operator promote provisional -> active
    core.submit(operator_proposal(Operator.METHOD_PROMOTE, {}))
    assert core.get(mid).status is l9.Status.ACTIVE
    assert mid in [m.id for m in layer9_link.usable_methods(core)]


def test_disabled_without_flag(monkeypatch):
    monkeypatch.delenv("KEVIN_USE_LAYER9", raising=False)
    assert layer9_link.enabled() is False                  # availability != enabled
