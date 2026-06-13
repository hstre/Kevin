"""Core data model for Kevin.

Kevin is a *creativity-routing* architecture. It does not "be creative" by
relaxing rules. It produces creativity by **routing** between four stages:

    unexplored solution spaces  ->  wild variation  ->  method transfer  ->  epistemic selection

and then hands a ranked, justified possibility space to a human, who owns the
only decisions a machine should not make: direction, taste, risk, value.

Following the ecosystem convention (DESi / AleXiona): **LLM for language, rules
for logic**. Everything in this module is plain, deterministic data. All scoring,
routing and selection live in the engines and operate on these structures only.

Design invariants borrowed from DESi:
  * Closed enumerations - no open-world category invention.
  * Replay-stable identity - ids are content hashes, never random.
  * Read-only provenance - records describe what happened; they are never
    silently rewritten.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum


def stable_id(prefix: str, *parts: str) -> str:
    """A replay-stable id: ``prefix_<first 12 hex of sha256(parts)>``.

    No PRNG anywhere in Kevin - identical inputs always yield identical ids, so
    a whole creative run is reproducible (the *routing* is; the language layer of
    a real LLM is not, and that is the point of the boundary).
    """
    digest = hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


# --------------------------------------------------------------------------- #
# Closed enumerations
# --------------------------------------------------------------------------- #


class WildMove(StrEnum):
    """The closed repertoire of the *wild brother*.

    The wild brother is allowed to spin. Its job is variation, not truth - so the
    moves are deliberately disreputable. But the *set* of moves is closed, so the
    chaos is enumerable and auditable.
    """

    ANALOGY = "analogy"
    ABSURD_COMBINATION = "absurd_combination"
    STYLE_BREAK = "style_break"
    DISTANT_DOMAIN = "distant_domain"
    RISKY_HYPOTHESIS = "risky_hypothesis"
    WHAT_IF = "what_if"


class Affinity(StrEnum):
    """Content-free tags describing the *shape* of a thinking move.

    Used to match abstract methods (Layer 9) to spaces and variants without ever
    touching the original content of either. A method tagged ``BOUNDARY`` transfers
    to anything also tagged ``BOUNDARY`` - regardless of domain.
    """

    BOUNDARY = "boundary"          # behaviour at the edges / limit cases
    EXCLUSION = "exclusion"        # rule out the dangerous/impossible first
    DECOMPOSITION = "decomposition"  # split into independently checkable parts
    CAUSAL = "causal"              # trace mechanism / cause and effect
    PROVENANCE = "provenance"      # weigh sources by interest / origin
    RISK = "risk"                  # enumerate failure modes
    INVERSION = "inversion"        # solve the opposite, then flip
    INVARIANT = "invariant"        # find what must not change
    ADVERSARIAL = "adversarial"    # attack your own answer
    ANALOGY = "analogy"            # transport structure across domains
    ABSTRACTION = "abstraction"    # move between levels of generality
    COMPOSITION = "composition"    # combine parts into an emergent whole


class Verdict(StrEnum):
    """The selector's closed verdict set. Mirrors DESi's gate discipline."""

    PROMISING = "promising"    # coherent, testable, connectable - worth a human look
    TENTATIVE = "tentative"    # interesting but under-supported - hold
    REJECTED = "rejected"      # pretty but hollow, or incoherent - drop


# --------------------------------------------------------------------------- #
# Stage 0 - the problem
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Problem:
    """The thing a human wants creative help with.

    ``known_approaches`` is what makes Kevin different from brainstorming: by
    declaring what has already been tried, the solution-space explorer can score
    *what is under-explored* instead of re-suggesting the obvious.
    """

    statement: str
    domain: str = "general"
    constraints: tuple[str, ...] = ()
    known_approaches: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return stable_id("prob", self.statement, self.domain)


# --------------------------------------------------------------------------- #
# Stage 1 - solution spaces (the DESi-flavoured explorer)
# --------------------------------------------------------------------------- #


@dataclass
class SolutionSpace:
    """A *region* of possible solutions, not a solution.

    Kevin's first move is not "find the creative answer" but "find the
    structurally plausible region nobody has worked yet".

    Scores (all in [0, 1], all assigned deterministically by the explorer):
      * ``plausibility`` - is this region structurally sensible for the problem?
      * ``exploration``  - how worked-over is it already? (high = crowded)
      * ``opportunity``  - derived: plausibility * (1 - exploration). High means
        "plausible *and* unworked" - exactly where Kevin wants the wild brother.
    """

    label: str
    description: str
    axis: str                       # the dimension this region varies along
    affinities: tuple[Affinity, ...] = ()
    plausibility: float = 0.5
    exploration: float = 0.5

    @property
    def id(self) -> str:
        return stable_id("space", self.label, self.axis)

    @property
    def opportunity(self) -> float:
        return round(self.plausibility * (1.0 - self.exploration), 4)


# --------------------------------------------------------------------------- #
# Stage 2 - wild variation
# --------------------------------------------------------------------------- #


@dataclass
class Variant:
    """One wild move inside one solution space.

    Produced by the wild brother. Not required to be true or even sane. Carries a
    ``wildness`` score so the selector and the human can see how far out it is.
    """

    space_id: str
    move: WildMove
    content: str
    wildness: float = 0.5

    @property
    def id(self) -> str:
        return stable_id("var", self.space_id, self.move.value, self.content)


# --------------------------------------------------------------------------- #
# Stage 3 - Layer 9 methods and their transfer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Method:
    """A *content-free* thinking move extracted from a successful process.

    This is the heart of Kevin's thesis. We do not reuse old *solutions*. We reuse
    the abstract *Denkbewegung* - the operator sequence - and apply it to a new
    field where nobody has tried it.

    ``steps`` are domain-agnostic by contract: "rule out the catastrophic options
    first", never "rule out a pulmonary embolism first". ``origin`` records where
    the move was first seen, for credit only - it carries no transferable content.
    """

    name: str
    origin: str                     # e.g. "medicine", "law", "DESi" - credit, not content
    summary: str
    steps: tuple[str, ...]
    affinities: tuple[Affinity, ...]

    @property
    def id(self) -> str:
        return stable_id("meth", self.name, self.origin)


@dataclass
class MethodTransfer:
    """The application of one abstract method to one target (space or variant).

    ``mapped_steps`` is the method's content-free steps re-read against the new
    problem. The mapping *structure* is deterministic; the *language* of each
    mapped step is the only thing the LLM is allowed to write here.
    """

    method_id: str
    method_name: str
    target_id: str
    mapped_steps: tuple[str, ...]
    rationale: str

    @property
    def id(self) -> str:
        return stable_id("xfer", self.method_id, self.target_id)


# --------------------------------------------------------------------------- #
# Stage 4 - candidates and epistemic selection
# --------------------------------------------------------------------------- #


@dataclass
class Signals:
    """Structured, LLM-extracted *signals* about a candidate.

    The LLM may *read* a candidate and report these booleans/anchors (that is
    language work). The selector then computes every score from them with fixed
    arithmetic (that is logic work). The LLM never emits a score.
    """

    has_falsifiable_claim: bool = False     # -> testability
    internal_contradiction: bool = False    # -> coherence penalty
    has_concrete_mechanism: bool = False     # -> coherence + connectivity
    anchors: tuple[str, ...] = ()            # terms tying it to the problem field
    overlaps_known: bool = False             # -> novelty penalty


@dataclass
class Candidate:
    """A single creative possibility offered for selection.

    A candidate is either a raw wild variant or a variant seen through a
    transferred method (the strong case: wild content disciplined by a method).
    """

    content: str
    space_id: str
    variant_id: str
    transfer_id: str | None = None
    wildness: float = 0.5
    signals: Signals = field(default_factory=Signals)

    @property
    def id(self) -> str:
        return stable_id("cand", self.variant_id, self.transfer_id or "raw")

    @property
    def method_disciplined(self) -> bool:
        return self.transfer_id is not None


@dataclass
class Evaluation:
    """The selector's deterministic read of one candidate."""

    candidate_id: str
    coherence: float
    testability: float
    connectivity: float
    novelty: float
    score: float
    verdict: Verdict
    reason: str


# --------------------------------------------------------------------------- #
# The run record (a local "Layer 9" of creative processes)
# --------------------------------------------------------------------------- #


@dataclass
class CreativeRun:
    """An append-only record of one full pass through Kevin.

    Kept so that, when a human later marks a candidate as having *worked*, the
    method-extractor can mine this record for a new abstract method - closing the
    loop the user described: Layer 9 grows from successful processes.
    """

    problem: Problem
    space_prediction: dict | None = None   # where DESi predicts the open regions are
    spaces: list[SolutionSpace] = field(default_factory=list)
    chosen_spaces: list[str] = field(default_factory=list)
    variants: list[Variant] = field(default_factory=list)
    transfers: list[MethodTransfer] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)

    @property
    def id(self) -> str:
        return stable_id("run", self.problem.id, *(c.id for c in self.candidates))

    def promising(self) -> list[Evaluation]:
        return [e for e in self.evaluations if e.verdict is Verdict.PROMISING]
