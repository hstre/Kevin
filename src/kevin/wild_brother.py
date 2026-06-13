"""Stage 2 - the wild brother (der wilde Bruder).

A free, aggressive, associative instance. Unusual analogies, absurd combinations,
style breaks, distant domains, risky hypotheses, "what if?". It is **allowed to
spin** - its task is not truth, it is *variation*. Selection comes later; here we
maximise spread.

The wild brother is the one place we deliberately invite nonsense. Two guard-rails
keep it auditable rather than merely random:
  * Its repertoire is the closed ``WildMove`` enum - chaos you can enumerate.
  * Each move gets a fixed ``wildness`` weight, so downstream stages can see how
    far out a variant is and the human can dial risk.

The text of each variant is the LLM's job (language). Which moves fire, in which
spaces, and how wild each is rated - that is this engine's job (logic).
"""

from __future__ import annotations

from .llm_client import LLMClient
from .models import Problem, SolutionSpace, Variant, WildMove

# How far out each move is, by construction. Replay-stable, no PRNG.
WILDNESS: dict[WildMove, float] = {
    WildMove.ANALOGY: 0.4,
    WildMove.STYLE_BREAK: 0.5,
    WildMove.WHAT_IF: 0.6,
    WildMove.DISTANT_DOMAIN: 0.75,
    WildMove.ABSURD_COMBINATION: 0.85,
    WildMove.RISKY_HYPOTHESIS: 0.9,
}

# An affinity-light region gets gentler moves; a rich region can take the wild end.
_DEFAULT_MOVES = (
    WildMove.ANALOGY,
    WildMove.WHAT_IF,
    WildMove.DISTANT_DOMAIN,
    WildMove.RISKY_HYPOTHESIS,
)


class WildBrother:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def vary(
        self,
        problem: Problem,
        space: SolutionSpace,
        *,
        moves: tuple[WildMove, ...] = _DEFAULT_MOVES,
    ) -> list[Variant]:
        """Generate one variant per requested move inside one space."""
        variants: list[Variant] = []
        for move in moves:
            content = self._llm.write_variant(problem, space, move)
            variants.append(
                Variant(
                    space_id=space.id,
                    move=move,
                    content=content,
                    wildness=WILDNESS[move],
                )
            )
        return variants

    def storm(
        self,
        problem: Problem,
        spaces: list[SolutionSpace],
        *,
        moves: tuple[WildMove, ...] = _DEFAULT_MOVES,
    ) -> list[Variant]:
        """Run the wild brother across every routed space. Order is deterministic."""
        out: list[Variant] = []
        for space in spaces:
            out.extend(self.vary(problem, space, moves=moves))
        return out
