"""Stage 1 - the solution-space explorer (the DESi-flavoured part).

Kevin's first move is the one most "creativity" tools skip: it does **not** look
for the clever answer. It looks for the *region* that is structurally plausible
yet under-worked, because that is where new value actually hides.

Division of labour:
  * The LLM *proposes* candidate regions (language).
  * This engine *scores* them deterministically (logic): how plausible, how
    already-worked, and therefore how much opportunity each holds.

``exploration`` is estimated from overlap with the problem's declared
``known_approaches``: a region whose vocabulary the human has already been working
is "crowded"; a plausible region they have *not* touched is the prize.
"""

from __future__ import annotations

from .llm_client import LLMClient
from .models import Affinity, Problem, SolutionSpace


def _tokens(text: str) -> set[str]:
    return {w.strip(".,;:!?'\"()").lower() for w in text.split() if len(w) > 3}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _plausibility(problem: Problem, axis: str, description: str) -> float:
    """How structurally sensible is this region for this problem?

    Heuristic and deterministic: a region that shares vocabulary with the problem
    statement is grounded (plausible); one that violates a stated constraint is
    docked. Bounded to a sane [0.2, 0.95] so nothing is ever certain or hopeless.
    """
    base = 0.45 + 0.5 * _overlap(problem.statement, description)
    for constraint in problem.constraints:
        if _overlap(constraint, description) > 0.3 and constraint.lower().startswith("no "):
            base -= 0.15
    return round(min(0.95, max(0.2, base)), 4)


def _exploration(problem: Problem, description: str, axis: str) -> float:
    """How worked-over is this region already?

    Driven by overlap with ``known_approaches``. No declared approaches -> we assume
    a lightly-explored field (0.3) rather than pretend certainty.
    """
    if not problem.known_approaches:
        return 0.3
    worked = max((_overlap(ka, description + " " + axis) for ka in problem.known_approaches),
                 default=0.0)
    return round(min(0.95, 0.2 + 0.8 * worked), 4)


class SolutionSpaceExplorer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def explore(self, problem: Problem) -> list[SolutionSpace]:
        """Return all candidate spaces, scored, sorted by opportunity (desc)."""
        spaces: list[SolutionSpace] = []
        for raw in self._llm.propose_spaces(problem):
            affinities = tuple(
                Affinity(a) for a in raw.get("affinities", []) if a in Affinity._value2member_map_
            )
            desc = raw["description"]
            axis = raw["axis"]
            spaces.append(
                SolutionSpace(
                    label=raw["label"],
                    description=desc,
                    axis=axis,
                    affinities=affinities,
                    plausibility=_plausibility(problem, axis, desc),
                    exploration=_exploration(problem, desc, axis),
                )
            )
        # Deterministic tie-break on id keeps ordering replay-stable.
        return sorted(spaces, key=lambda s: (-s.opportunity, s.id))

    def select_underexplored(
        self, problem: Problem, *, top_k: int = 2, min_opportunity: float = 0.15
    ) -> list[SolutionSpace]:
        """Route to the most promising under-worked regions.

        This is the routing decision the user's design hinges on: send the wild
        brother *only* into plausible-but-unworked territory, not everywhere.
        """
        ranked = self.explore(problem)
        picked = [s for s in ranked if s.opportunity >= min_opportunity][:top_k]
        # Never return empty if anything plausible exists - take the best regardless.
        if not picked and ranked:
            picked = ranked[:1]
        return picked
