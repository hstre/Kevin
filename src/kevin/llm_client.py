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
import time
from typing import Protocol, runtime_checkable

from .models import Affinity, Problem, SolutionSpace, WildMove

# Substrings that mark a *transient* failure worth retrying. Notably includes the
# proxied-egress quirk seen in managed environments, where DNS occasionally returns
# no records ("resolve_no_records" / "private/reserved IP") for a host that is in
# fact reachable. Genuine auth/4xx errors do not match and are never retried.
_TRANSIENT_MARKERS = (
    "resolve_no_records", "private/reserved", "temporarily", "timeout", "timed out",
    "connection", "overloaded", "rate limit", "too many requests",
    "502", "503", "504", "bad gateway", "service unavailable", "gateway timeout",
)
_TRANSIENT_EXCEPTIONS = {
    "APIConnectionError", "APITimeoutError", "RateLimitError",
    "InternalServerError", "APIStatusError",
}


def _is_transient(exc: Exception) -> bool:
    if type(exc).__name__ in _TRANSIENT_EXCEPTIONS:
        return True
    blob = str(exc).lower()
    return any(marker in blob for marker in _TRANSIENT_MARKERS)


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
        ("level", "the right level of generality to attack it on",
         (Affinity.ABSTRACTION, Affinity.DECOMPOSITION)),
        ("synthesis", "what emerges when the parts interact, not the parts alone",
         (Affinity.COMPOSITION, Affinity.CAUSAL)),
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


class OpenAICompatibleLLM:
    """A real language layer: DeepSeek / OpenAI, via the OpenAI-compatible SDK.

    Matches the rest of the ecosystem (DESi / AleXiona ``llm_client.py``):
    ``DEEPSEEK_API_KEY`` takes priority over ``OPENAI_API_KEY``; the client selects
    ``deepseek-chat`` or ``gpt-4o`` accordingly. This class is the *only* place a
    network call happens - the engines never change.

    Temperature is the dial that used to be (wrongly) called "creativity": here it
    is just per-task. The wild brother runs hot; reading a candidate into signals
    runs cold. Routing and selection remain deterministic regardless, because they
    live in the engines, not here.
    """

    # The closed affinity vocabulary the model must choose from - no open-world tags.
    _AFFINITIES = ", ".join(a.value for a in Affinity)

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "The real LLM client needs the 'openai' package. "
                "Install it with: pip install 'kevin[llm]'"
            ) from exc

        # Accept either secret name (DEEPSEEK_API_KEY or DEEPSEEK_API_KEY2) so a
        # stored secret works regardless of which slot it was put in.
        deepseek = (
            api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY2")
        )
        openai_key = os.getenv("OPENAI_API_KEY")
        if deepseek:
            self._model = model or "deepseek-chat"
            self._client = OpenAI(api_key=deepseek,
                                  base_url=base_url or "https://api.deepseek.com")
        elif openai_key:
            self._model = model or "gpt-4o"
            self._client = OpenAI(api_key=openai_key, base_url=base_url)
        else:  # pragma: no cover - config error path
            raise RuntimeError(
                "No LLM key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY "
                "(see .env.example), or unset KEVIN_USE_REAL_LLM to use the MockLLM."
            )

        # Transient-failure retry (e.g. proxied-egress DNS flakiness). Configurable.
        self._max_retries = int(os.getenv("KEVIN_LLM_RETRIES", "4"))
        self._backoff_base = float(os.getenv("KEVIN_LLM_BACKOFF", "0.5"))

    # -- low-level calls ---------------------------------------------------- #
    def _chat(self, system: str, user: str, *, temperature: float, json: bool = False) -> str:
        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json:
            kwargs["response_format"] = {"type": "json_object"}

        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - retry only transient, re-raise the rest
                if attempt >= self._max_retries or not _is_transient(exc):
                    raise
                last = exc
                time.sleep(self._backoff_base * (2 ** attempt))
        raise last  # unreachable, but keeps the type checker honest

    @staticmethod
    def _parse_json(raw: str) -> dict:
        import json as _json

        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            # Tolerate a fenced or chatty reply: grab the outermost {...}.
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                try:
                    return _json.loads(raw[start : end + 1])
                except _json.JSONDecodeError:
                    pass
            return {}

    # -- LLMClient surface -------------------------------------------------- #
    def propose_spaces(self, problem: Problem) -> list[dict]:
        system = (
            "You map a problem into candidate SOLUTION SPACES (regions of possible "
            "solutions, not solutions). For each, give a short label, a one-sentence "
            "description, a one-word axis it varies along, and 1-3 affinity tags chosen "
            f"ONLY from this closed set: [{self._AFFINITIES}]. "
            'Reply as JSON: {"spaces": [{"label","description","axis","affinities"}]}.'
        )
        tried = ", ".join(problem.known_approaches) or "none"
        user = (
            f"Problem: {problem.statement}\nDomain: {problem.domain}\n"
            f"Already tried (avoid re-suggesting these): {tried}"
        )
        data = self._parse_json(self._chat(system, user, temperature=0.4, json=True))
        spaces = data.get("spaces", [])
        return spaces if isinstance(spaces, list) else []

    def write_variant(self, problem: Problem, space: SolutionSpace, move: WildMove) -> str:
        system = (
            "You are the WILD BROTHER: a free, aggressive, associative idea generator. "
            "Your job is variation, not truth. Be bold; one or two sentences; no hedging."
        )
        user = (
            f"Problem: {problem.statement}\n"
            f"Region to explore: {space.label} - {space.description}\n"
            f"Creative move to use: {move.value.replace('_', ' ')}.\n"
            "Produce one wild variant using exactly that move."
        )
        # Hot - this is the place we *want* spread.
        return self._chat(system, user, temperature=1.1).strip()

    def phrase_transfer(self, target_text: str, step: str) -> str:
        system = (
            "You transfer an abstract, content-free thinking step onto a concrete idea. "
            "Keep the step's STRUCTURE; bind it to the idea in one concrete sentence. "
            "Do not import any content from the step's original domain."
        )
        user = f"Abstract step: {step}\nIdea to apply it to: {target_text}"
        return self._chat(system, user, temperature=0.5).strip()

    def read_signals(self, problem: Problem, candidate_text: str) -> dict:
        system = (
            "You READ an idea and report structured signals. You do NOT score or judge. "
            'Reply as JSON with keys: has_falsifiable_claim (bool), '
            "internal_contradiction (bool), has_concrete_mechanism (bool), "
            "anchors (array of short strings tying it to the problem), "
            "overlaps_known (bool: does it merely restate an already-tried approach?)."
        )
        user = (
            f"Problem: {problem.statement}\n"
            f"Already tried: {', '.join(problem.known_approaches) or 'none'}\n"
            f"Idea: {candidate_text}"
        )
        data = self._parse_json(self._chat(system, user, temperature=0.0, json=True))
        return {
            "has_falsifiable_claim": bool(data.get("has_falsifiable_claim")),
            "internal_contradiction": bool(data.get("internal_contradiction")),
            "has_concrete_mechanism": bool(data.get("has_concrete_mechanism")),
            "anchors": tuple(data.get("anchors", []) or ()),
            "overlaps_known": bool(data.get("overlaps_known")),
        }


def get_default_client() -> LLMClient:
    """Return the configured client.

    Without ``KEVIN_USE_REAL_LLM=1``, returns the deterministic ``MockLLM`` - so
    tests, CI and a fresh clone all work with zero setup. With it set, returns the
    real OpenAI-compatible client (needs the ``llm`` extra and a key). The engines
    are identical either way; only the language layer moves.
    """
    if os.getenv("KEVIN_USE_REAL_LLM") == "1":  # pragma: no cover - needs a key + network
        return OpenAICompatibleLLM()
    return MockLLM()
