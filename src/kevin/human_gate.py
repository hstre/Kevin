"""Stage 5 - the human gate.

Kevin stops here, deliberately. The machine has done what machines are good at:
mapped the possibility space, varied it wildly, disciplined it with transferred
methods, and filtered it for epistemic worth. What remains are the four decisions
the design reserves for the human:

    direction - which way is this actually going?
    taste     - which of these is *good*, not merely valid?
    risk      - how far out are we willing to bet?
    value     - what is this worth pursuing for?

So this module does not decide. It *presents* - a ranked, justified briefing - and
it *records* the human's choice, because that recorded choice is what later feeds
``method_library.extract_method`` to grow Layer 9 from successes.

    KI erzeugt strukturierte Moeglichkeitsraeume; der Mensch waehlt Bedeutung und Richtung.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Candidate, CreativeRun, Evaluation


@dataclass
class Briefing:
    """What Kevin hands a human: a ranked possibility space, nothing decided."""

    problem_statement: str
    promising: list[tuple[Evaluation, Candidate]]
    tentative: list[tuple[Evaluation, Candidate]]
    # The four axes the human owns - listed explicitly so the tool never pretends
    # to have an opinion on them.
    decision_axes: tuple[str, ...] = ("direction", "taste", "risk", "value")

    def render(self) -> str:
        lines = [
            f"PROBLEM: {self.problem_statement}",
            "",
            "Kevin has mapped, varied, transferred and filtered. It does not decide.",
            f"Yours to choose: {', '.join(self.decision_axes)}.",
            "",
            "PROMISING (coherent / testable / connected):",
        ]
        if not self.promising:
            lines.append("  (none cleared the gate this run)")
        for ev, cand in self.promising:
            disciplined = " [method-disciplined]" if cand.method_disciplined else ""
            lines.append(f"  - {cand.content}{disciplined}")
            lines.append(
                f"      score={ev.score}  coherence={ev.coherence} testability={ev.testability} "
                f"connectivity={ev.connectivity} novelty={ev.novelty}"
            )
            lines.append(f"      {ev.reason}")
        if self.tentative:
            lines.append("")
            lines.append("TENTATIVE (interesting, under-built - your call on risk):")
            for ev, cand in self.tentative:
                lines.append(f"  - {cand.content}  (score={ev.score}: {ev.reason})")
        return "\n".join(lines)


def build_briefing(run: CreativeRun) -> Briefing:
    by_id = {c.id: c for c in run.candidates}
    promising, tentative = [], []
    for ev in run.evaluations:
        cand = by_id.get(ev.candidate_id)
        if cand is None:
            continue
        if ev.verdict.value == "promising":
            promising.append((ev, cand))
        elif ev.verdict.value == "tentative":
            tentative.append((ev, cand))
    return Briefing(
        problem_statement=run.problem.statement,
        promising=promising,
        tentative=tentative,
    )
