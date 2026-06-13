"""The language boundary.

Everything the LLM is *allowed* to do passes through this interface, and nothing
else does. The contract enforces the ecosystem rule **LLM for language, rules for
logic**:

  * The LLM proposes spaces, writes wild variants, phrases method transfers and
    *reads* candidates into structured signals.
  * The LLM never scores, never routes, never selects, never decides.

A ``MockLLM`` is the default so that the whole pipeline, the tests and CI run
offline and replay-stably with no API key. A real OpenAI-compatible client
(DeepSeek / GPT-4o, matching the rest of the ecosystem) can be dropped in behind
the same Protocol without touching any engine.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol, runtime_checkable

from .models import Affinity, Problem, SolutionSpace, WildMove


@runtime_checkable
class LLMClient(Protocol):
    """The only surface through which language enters Kevin."""

    def propose_spaces(self, problem: Problem) -> list[dict]:
        """Return candidate solution-space dicts: label/description/axis/affinities."""

    def write_variant(self, problem: Problem, space: SolutionSpace, move: WildMove) -> str:
        """Write one wild variant. May be absurd - that is the job."""

    def phrase_transfer(self, target_text: str, step: str) -> str:
        """Re-read one abstract method step against a concrete target. Language only."""

    def read_signals(self, problem: Problem, candidate_text: str) -> dict:
        """Read a candidate into structured booleans/anchors. No scoring."""


def _seed(*parts: str) -> int:
    """Deterministic integer seed from text - replaces a PRNG so runs replay."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16)


class MockLLM:
    """Deterministic, offline stand-in for a real language model.

    It produces plausible, varied, *replay-stable* text by hash-selecting from
    templates. It is not intelligent - it exists so the routing/selection logic can
    be exercised and tested without a network. Swap in a real client for real
    creativity; the engines do not change.
    """

    # Template banks. Hash selection over these gives deterministic variety.
    _SPACE_AXES = [
        ("mechanism", "how the underlying process actually works",
         (Affinity.CAUSAL, Affinity.DECOMPOSITION)),
        ("constraint", "which constraint, if removed, dissolves the problem",
         (Affinity.INVERSION, Affinity.INVARIANT)),
        ("boundary", "the behaviour at the extreme / limit cases",
         (Affinity.BOUNDARY, Affinity.RISK)),
        ("actor", "whose interests shape the available evidence",
         (Affinity.PROVENANCE, Affinity.ADVERSARIAL)),
        ("analogy", "a distant field with the same structural shape",
         (Affinity.ANALOGY, Affinity.DECOMPOSITION)),
    ]

    _WILD_TEMPLATES = {
        WildMove.ANALOGY: "Treat '{p}' as if it were {x} - what carries over?",
        WildMove.ABSURD_COMBINATION: "Fuse '{p}' with {x}, however absurd, and keep the seam.",
        WildMove.STYLE_BREAK: "Solve '{p}' in the register of {x} instead of its usual one.",
        WildMove.DISTANT_DOMAIN: "Import a working pattern from {x} into '{p}'.",
        WildMove.RISKY_HYPOTHESIS: "Bet that '{p}' is really driven by {x}, and chase it.",
        WildMove.WHAT_IF: "What if the opposite of {x} were the actual lever in '{p}'?",
    }

    _DISTANT = [
        "tidal hydraulics", "medieval guild law", "immune signalling", "jazz comping",
        "container logistics", "mycelial networks", "double-entry bookkeeping",
        "earthquake early-warning", "beekeeping", "film editing", "sourdough fermentation",
        "orbital mechanics",
    ]

    def propose_spaces(self, problem: Problem) -> list[dict]:
        out = []
        for axis_key, axis_desc, affinities in self._SPACE_AXES:
            out.append(
                {
                    "label": f"{axis_key.title()} space",
                    "description": f"Explore {problem.statement!r} along {axis_desc}.",
                    "axis": axis_key,
                    "affinities": [a.value for a in affinities],
                }
            )
        return out

    def write_variant(self, problem: Problem, space: SolutionSpace, move: WildMove) -> str:
        s = _seed(problem.statement, space.id, move.value)
        x = self._DISTANT[s % len(self._DISTANT)]
        template = self._WILD_TEMPLATES[move]
        return template.format(p=problem.statement, x=x)

    def phrase_transfer(self, target_text: str, step: str) -> str:
        # Content-free re-reading: keep the abstract step, bind it to the target.
        return f"{step} - applied to: {target_text}"

    def read_signals(self, problem: Problem, candidate_text: str) -> dict:
        s = _seed(problem.statement, candidate_text)
        text = candidate_text.lower()
        anchors = tuple(
            w for w in problem.statement.lower().split() if len(w) > 4 and w in text
        )
        return {
            # Deterministic pseudo-reads, biased by surface features so that
            # method-disciplined candidates (which contain mapped steps) tend to
            # score better - mirroring the real intent.
            "has_falsifiable_claim": ("if " in text or "bet that" in text or s % 2 == 0),
            "internal_contradiction": (s % 7 == 0),
            "has_concrete_mechanism": ("mechanism" in text or "lever" in text or "->" in text
                                       or "applied to" in text),
            "anchors": anchors,
            "overlaps_known": any(
                ka.lower() in text for ka in problem.known_approaches
            ),
        }


def get_default_client() -> LLMClient:
    """Return the configured client.

    Without ``KEVIN_USE_REAL_LLM=1`` and a key, returns the deterministic
    ``MockLLM`` - so tests, CI and a fresh clone all work with zero setup. Wiring a
    real OpenAI-compatible client is intentionally left as a single, obvious seam.
    """
    if os.getenv("KEVIN_USE_REAL_LLM") == "1":  # pragma: no cover - needs a key + network
        raise NotImplementedError(
            "Real LLM client not wired in this demonstrator. Implement an "
            "OpenAI-compatible LLMClient and return it here (see DESi/AleXiona "
            "llm_client.py for the pattern). Kevin's engines are unaffected."
        )
    return MockLLM()
