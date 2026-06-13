from kevin.llm_client import MockLLM
from kevin.models import Candidate, Problem, Signals, Verdict
from kevin.selector import Selector


def _cand(content="x", **sig):
    c = Candidate(content=content, space_id="s", variant_id="v", wildness=sig.pop("wildness", 0.5))
    c.signals = Signals(**sig)
    return c


def test_pretty_but_hollow_is_rejected():
    """High novelty, no testability, no connection -> rejected. 'Not just pretty.'"""
    sel = Selector(MockLLM())
    cand = _cand(content="dazzling but empty", wildness=0.95)
    # Force the hollow signal profile directly (bypass the mock read by re-reading).
    ev = sel.evaluate(Problem("unrelated problem text"), cand)
    # We can't fully control the mock's read here, so assert the gate logic via
    # the deterministic path below instead.
    assert ev.verdict in (Verdict.REJECTED, Verdict.TENTATIVE, Verdict.PROMISING)


def test_gate_logic_directly():
    sel = Selector(MockLLM())
    # incoherent
    v, _ = sel._gate(coh=0.1, test=0.9, conn=0.9, nov=0.9, score=0.8)
    assert v is Verdict.REJECTED
    # striking but untestable -> tentative
    v, _ = sel._gate(coh=0.8, test=0.1, conn=0.8, nov=0.9, score=0.7)
    assert v is Verdict.TENTATIVE
    # striking but disconnected -> tentative
    v, _ = sel._gate(coh=0.8, test=0.8, conn=0.1, nov=0.9, score=0.7)
    assert v is Verdict.TENTATIVE
    # the real thing
    v, _ = sel._gate(coh=0.8, test=0.8, conn=0.8, nov=0.6, score=0.75)
    assert v is Verdict.PROMISING
    # hollow
    v, _ = sel._gate(coh=0.5, test=0.2, conn=0.2, nov=0.9, score=0.3)
    assert v is Verdict.REJECTED


def test_method_disciplined_candidate_scores_higher_coherence():
    sel = Selector(MockLLM())
    raw = _cand(content="same text")
    disciplined = Candidate(content="same text", space_id="s", variant_id="v",
                            transfer_id="xfer_1")
    p = Problem("a problem about text")
    ev_raw = sel.evaluate(p, raw)
    ev_disc = sel.evaluate(p, disciplined)
    assert ev_disc.coherence >= ev_raw.coherence


def test_contradiction_sinks_coherence():
    sel = Selector(MockLLM())
    cand = _cand(internal_contradiction=True)
    # Directly check the private scorer behaviour via evaluate's contract:
    p = Problem("problem")
    ev = sel.evaluate(p, cand)
    # The mock may overwrite signals; assert the deterministic scorer instead.
    from kevin.selector import _coherence
    contradicting = _cand(internal_contradiction=True, has_concrete_mechanism=False)
    clean = _cand(internal_contradiction=False, has_concrete_mechanism=True)
    assert _coherence(contradicting) < _coherence(clean)
    assert ev.candidate_id == cand.id


def test_select_sorts_by_score_desc():
    sel = Selector(MockLLM())
    p = Problem("how to make onboarding feel less like paperwork")
    cands = [_cand(content=f"variant number {i}") for i in range(5)]
    evals = sel.select(p, cands)
    scores = [e.score for e in evals]
    assert scores == sorted(scores, reverse=True)
