"""Stage 4 - epistemic selection (DESi's job: keep the wild brother honest).

The wild brother produced spread. Method transfer gave some of that spread a
backbone. Now selection asks the question the design insists on:

    What is coherent, testable, connectable - and not *just* pretty?

This is where Kevin refuses the classic failure of creativity tools: rewarding
novelty for its own sake. A dazzling variant with no falsifiable claim and no
bridge back to the problem is **rejected**, however beautiful.

Strict boundary: the LLM only supplies structured ``Signals`` (it *reads*). Every
number and the final verdict are computed here with fixed arithmetic (logic). The
LLM never emits a score and never decides a verdict.
"""

from __future__ import annotations

from .llm_client import LLMClient
from .models import Candidate, Evaluation, Problem, Verdict

# Weights for the composite score. Connectivity and testability outweigh novelty on
# purpose: "not just pretty" is enforced numerically, not just rhetorically.
W_COHERENCE = 0.30
W_TESTABILITY = 0.30
W_CONNECTIVITY = 0.25
W_NOVELTY = 0.15

PROMISING_AT = 0.62
TENTATIVE_AT = 0.40

# A candidate that is pure novelty with no spine is capped, no matter its score.
_MIN_TESTABILITY_FOR_PROMISING = 0.34
_MIN_CONNECTIVITY_FOR_PROMISING = 0.34


def _coherence(c: Candidate) -> float:
    s = c.signals
    base = 0.55
    if s.internal_contradiction:
        base -= 0.4
    if s.has_concrete_mechanism:
        base += 0.25
    # Method-disciplined candidates inherit a structural coherence bonus: a
    # transferred method gives them an ordered backbone.
    if c.method_disciplined:
        base += 0.15
    return round(min(1.0, max(0.0, base)), 4)


def _testability(c: Candidate) -> float:
    s = c.signals
    base = 0.2
    if s.has_falsifiable_claim:
        base += 0.45
    if s.has_concrete_mechanism:
        base += 0.2
    if c.method_disciplined:
        base += 0.15
    return round(min(1.0, max(0.0, base)), 4)


def _connectivity(problem: Problem, c: Candidate) -> float:
    """Anschlussfähigkeit: does it bridge back to the problem field?"""
    s = c.signals
    anchor_strength = min(1.0, len(s.anchors) * 0.3)
    base = 0.15 + 0.55 * anchor_strength
    if s.has_concrete_mechanism:
        base += 0.15
    return round(min(1.0, max(0.0, base)), 4)


def _novelty(c: Candidate) -> float:
    """High when it does *not* overlap the known approaches; scaled by wildness."""
    base = 0.35 + 0.5 * c.wildness
    if c.signals.overlaps_known:
        base -= 0.45
    return round(min(1.0, max(0.0, base)), 4)


class Selector:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def evaluate(self, problem: Problem, candidate: Candidate) -> Evaluation:
        # Language step: read the candidate into structured signals.
        raw = self._llm.read_signals(problem, candidate.content)
        candidate.signals.has_falsifiable_claim = bool(raw.get("has_falsifiable_claim"))
        candidate.signals.internal_contradiction = bool(raw.get("internal_contradiction"))
        candidate.signals.has_concrete_mechanism = bool(raw.get("has_concrete_mechanism"))
        candidate.signals.anchors = tuple(raw.get("anchors", ()))
        candidate.signals.overlaps_known = bool(raw.get("overlaps_known"))

        # Logic step: deterministic scoring from the signals.
        coh = _coherence(candidate)
        test = _testability(candidate)
        conn = _connectivity(problem, candidate)
        nov = _novelty(candidate)
        score = round(
            W_COHERENCE * coh
            + W_TESTABILITY * test
            + W_CONNECTIVITY * conn
            + W_NOVELTY * nov,
            4,
        )

        verdict, reason = self._gate(coh, test, conn, nov, score)
        return Evaluation(
            candidate_id=candidate.id,
            coherence=coh,
            testability=test,
            connectivity=conn,
            novelty=nov,
            score=score,
            verdict=verdict,
            reason=reason,
        )

    @staticmethod
    def _gate(
        coh: float, test: float, conn: float, nov: float, score: float
    ) -> tuple[Verdict, str]:
        if coh < 0.3:
            return Verdict.REJECTED, "incoherent: internal contradiction outweighs structure"
        if score >= PROMISING_AT:
            if test < _MIN_TESTABILITY_FOR_PROMISING:
                reason = "striking but not yet testable - needs a falsifiable claim"
                return Verdict.TENTATIVE, reason
            if conn < _MIN_CONNECTIVITY_FOR_PROMISING:
                reason = "striking but weakly connected - needs a bridge to the problem"
                return Verdict.TENTATIVE, reason
            return Verdict.PROMISING, "coherent, testable and connected - worth a human look"
        if score >= TENTATIVE_AT:
            return Verdict.TENTATIVE, "partially supported - interesting but under-built"
        return Verdict.REJECTED, "pretty but hollow: low on coherence/testability/connection"

    def select(self, problem: Problem, candidates: list[Candidate]) -> list[Evaluation]:
        """Evaluate all candidates; sort best-first with a replay-stable tie-break."""
        evals = [self.evaluate(problem, c) for c in candidates]
        return sorted(evals, key=lambda e: (-e.score, e.candidate_id))
