"""Stage 3 - Layer 9: the method library and the transfer engine.

This is the core of Kevin's thesis and the part that separates it from ordinary
"structured creativity":

    We do not reuse old *solutions*. We extract the abstract *method* - the
    Denkbewegung, the operator sequence - strip it of all content, and transfer it
    to a new problem field where nobody has tried it.

So a method from medicine ("rule out the catastrophic options first") becomes,
content-free, a move you can run on a software-architecture problem. A method from
law ("element -> exception -> consequence") becomes a move you can run on an ethics
question. The library stores *only* shape; the original content never travels.

Two engines live here:
  * ``MethodLibrary`` - the seeded store of content-free methods, plus matching by
    affinity (logic) and the transfer of a method onto a target (structure is logic,
    wording is the LLM).
  * ``extract_method`` - the loop-closer: when a human marks a run as having worked,
    mine a *new* abstract method out of it and add it to the library. This is how
    Layer 9 grows from successful processes, exactly as the design calls for.
"""

from __future__ import annotations

from .llm_client import LLMClient
from .models import Affinity, CreativeRun, Method, MethodTransfer

# --------------------------------------------------------------------------- #
# The seed library: validated thinking moves, each reduced to content-free steps.
# Origins are credit only; the steps never name a domain object.
# --------------------------------------------------------------------------- #

SEED_METHODS: tuple[Method, ...] = (
    Method(
        name="limit_case_analysis",
        origin="mathematics",
        summary="Probe behaviour at the extremes; the edges expose the rule.",
        steps=(
            "Push each variable to its extreme and to zero.",
            "Describe what the system does at each extreme.",
            "Read off the rule that must hold in between.",
        ),
        affinities=(Affinity.BOUNDARY, Affinity.INVARIANT),
    ),
    Method(
        name="element_exception_consequence",
        origin="law",
        summary="Separate the qualifying conditions from their exceptions and effects.",
        steps=(
            "List the conditions that must all hold for the case to apply.",
            "List the exceptions that defeat it even when conditions hold.",
            "State the consequence that follows when conditions hold and no exception fires.",
        ),
        affinities=(Affinity.DECOMPOSITION, Affinity.EXCLUSION),
    ),
    Method(
        name="red_flags_first",
        origin="medicine",
        summary="Exclude the catastrophic possibilities before chasing the likely ones.",
        steps=(
            "Enumerate the outcomes that would be unrecoverable if missed.",
            "Find the cheapest test that rules each one in or out.",
            "Only then explore the comfortable, probable explanations.",
        ),
        affinities=(Affinity.EXCLUSION, Affinity.RISK),
    ),
    Method(
        name="source_by_interest",
        origin="history",
        summary="Weigh every source by whose interest it served.",
        steps=(
            "For each piece of evidence, identify who produced it and why.",
            "Discount in proportion to the producer's stake in the conclusion.",
            "Trust most what survives despite cutting against its source's interest.",
        ),
        affinities=(Affinity.PROVENANCE, Affinity.ADVERSARIAL),
    ),
    Method(
        name="failure_mode_analysis",
        origin="engineering",
        summary="Enumerate how it breaks before asking how it works.",
        steps=(
            "List every way the thing could fail.",
            "Rate each by severity and by how hidden it is.",
            "Design against the severe, hidden failures first.",
        ),
        affinities=(Affinity.RISK, Affinity.CAUSAL),
    ),
    Method(
        name="claim_splitting",
        origin="DESi",
        summary="Split a compound claim, check each part, mark conflicts, consolidate.",
        steps=(
            "Break the assertion into independently checkable atomic claims.",
            "Test each atom against evidence on its own.",
            "Flag where atoms contradict, then consolidate what survives.",
        ),
        affinities=(Affinity.DECOMPOSITION, Affinity.ADVERSARIAL),
    ),
    Method(
        name="invert_then_flip",
        origin="design",
        summary="Solve the opposite problem, then invert the answer.",
        steps=(
            "Ask how you would guarantee the worst possible outcome.",
            "Enumerate the moves that would cause it.",
            "Invert each move into a defence for the real problem.",
        ),
        affinities=(Affinity.INVERSION, Affinity.RISK),
    ),
    Method(
        name="structural_analogy_transport",
        origin="physics",
        summary="Find a distant system with the same shape and borrow its mathematics.",
        steps=(
            "Strip the problem to its relational skeleton, ignoring content.",
            "Find a well-understood system with the same skeleton.",
            "Transport that system's solved relations back, and check the fit.",
        ),
        affinities=(Affinity.ANALOGY, Affinity.DECOMPOSITION),
    ),
    Method(
        name="first_principles_reduction",
        origin="philosophy",
        summary="Strip away convention to the irreducible, then rebuild from there.",
        steps=(
            "Drop every assumption that is merely inherited or conventional.",
            "Keep only what cannot be reduced further without contradiction.",
            "Rebuild upward from those irreducibles, admitting nothing unearned.",
        ),
        affinities=(Affinity.DECOMPOSITION, Affinity.ABSTRACTION),
    ),
    Method(
        name="dimensional_consistency",
        origin="physics",
        summary="Check that the kinds of things balance before trusting any combination.",
        steps=(
            "Name the kind of thing each quantity is, not just its size.",
            "Check that combined quantities are of compatible kinds.",
            "Reject any step where the kinds fail to balance.",
        ),
        affinities=(Affinity.INVARIANT, Affinity.BOUNDARY),
    ),
    Method(
        name="conservation_tracking",
        origin="physics",
        summary="Find what can be neither created nor destroyed, and let the books balance.",
        steps=(
            "Identify a quantity that can be neither created nor destroyed here.",
            "Account for every place it can flow into or out of.",
            "Use the ledger that must balance to pin down the unknown.",
        ),
        affinities=(Affinity.INVARIANT, Affinity.CAUSAL),
    ),
    Method(
        name="five_whys",
        origin="manufacturing",
        summary="Chase the causal chain past symptoms to the removable root.",
        steps=(
            "State the surface symptom plainly.",
            "Ask why it occurs, then ask why of that answer, repeatedly.",
            "Stop at the first cause whose removal prevents all the rest.",
        ),
        affinities=(Affinity.CAUSAL, Affinity.DECOMPOSITION),
    ),
    Method(
        name="premortem",
        origin="decision-science",
        summary="Assume it already failed, then work backward into present safeguards.",
        steps=(
            "Assume the effort has already failed completely.",
            "From that future, narrate the most likely story of how it failed.",
            "Convert each step of that story into a safeguard you add now.",
        ),
        affinities=(Affinity.RISK, Affinity.INVERSION),
    ),
    Method(
        name="steelman_then_test",
        origin="rhetoric",
        summary="Build the strongest opposing case, then keep only what survives it.",
        steps=(
            "Build the strongest possible version of the opposing position.",
            "Grant it every charitable assumption it could fairly claim.",
            "Keep your conclusion only if it survives that strongest opponent.",
        ),
        affinities=(Affinity.ADVERSARIAL, Affinity.PROVENANCE),
    ),
    Method(
        name="base_rate_first",
        origin="statistics",
        summary="Anchor on how often it happens in general before the vivid particulars.",
        steps=(
            "Before the vivid particulars, find how often this happens in general.",
            "Anchor your estimate on that background frequency.",
            "Adjust only as far as specific, reliable evidence warrants.",
        ),
        affinities=(Affinity.PROVENANCE, Affinity.RISK),
    ),
    Method(
        name="constraint_relaxation",
        origin="optimization",
        summary="Drop the hardest constraint, solve freely, then re-impose and keep what survives.",
        steps=(
            "Identify the single constraint that currently binds hardest.",
            "Imagine it briefly removed and solve the freed problem.",
            "Re-impose it and keep whatever of the free solution still stands.",
        ),
        affinities=(Affinity.INVERSION, Affinity.BOUNDARY),
    ),
    Method(
        name="separation_of_concerns",
        origin="software",
        summary="Cut along axes that change for different reasons; recombine through narrow seams.",
        steps=(
            "Find the axes along which parts change for different reasons.",
            "Cut along those axes so each part has one reason to change.",
            "Recombine only through narrow, explicit interfaces.",
        ),
        affinities=(Affinity.DECOMPOSITION, Affinity.COMPOSITION),
    ),
    Method(
        name="emergence_search",
        origin="complexity",
        summary="Specify simple local rules; read the global pattern off their interaction.",
        steps=(
            "List the simple local rules each part follows on its own.",
            "Let many parts interact and watch for patterns no single part holds.",
            "Attribute the global behaviour to the interaction, not the parts.",
        ),
        affinities=(Affinity.COMPOSITION, Affinity.CAUSAL),
    ),
    Method(
        name="abstraction_ladder",
        origin="general-semantics",
        summary="Move up and down levels of generality; solve where the obstacle vanishes.",
        steps=(
            "Climb up: restate the problem at a more general level.",
            "Climb down: restate it at a more concrete level.",
            "Solve where the obstacle disappears, then translate the answer back.",
        ),
        affinities=(Affinity.ABSTRACTION, Affinity.ANALOGY),
    ),
    Method(
        name="occams_pruning",
        origin="philosophy",
        summary="Among equally adequate explanations, prefer the one needing fewest assumptions.",
        steps=(
            "List every explanation that fits the evidence equally well.",
            "Count the independent assumptions each one quietly requires.",
            "Prefer the one carrying the fewest unearned assumptions.",
        ),
        affinities=(Affinity.EXCLUSION, Affinity.INVARIANT),
    ),
)


class MethodLibrary:
    """A store of content-free methods with affinity-based matching and transfer."""

    def __init__(self, methods: tuple[Method, ...] = SEED_METHODS) -> None:
        # Keyed by id; dict preserves insertion order so iteration is deterministic.
        self._methods: dict[str, Method] = {m.id: m for m in methods}

    # -- access ------------------------------------------------------------- #
    def all(self) -> list[Method]:
        return list(self._methods.values())

    def add(self, method: Method) -> None:
        self._methods[method.id] = method

    # -- matching (logic) --------------------------------------------------- #
    def match(self, affinities: tuple[Affinity, ...], *, top_k: int = 2) -> list[Method]:
        """Return methods whose shape best fits the given affinities.

        Pure set overlap on affinity tags - never on content. A high score means
        "this thinking move has the same shape as this region", regardless of the
        domains involved. Ties break on method id for replay stability.
        """
        target = set(affinities)
        scored = []
        for m in self._methods.values():
            overlap = len(target & set(m.affinities))
            if overlap:
                scored.append((overlap, m))
        scored.sort(key=lambda t: (-t[0], t[1].id))
        return [m for _, m in scored[:top_k]]

    # -- transfer (structure = logic, wording = LLM) ------------------------ #
    def transfer(
        self, llm: LLMClient, method: Method, target_id: str, target_text: str
    ) -> MethodTransfer:
        """Re-read a method's content-free steps against a concrete target.

        The *mapping* (one mapped step per abstract step, in order) is deterministic
        structure. The LLM only phrases each mapped step. The original domain content
        of the method never appears - only its shape crosses the boundary.
        """
        mapped = tuple(self._phrase(llm, target_text, step) for step in method.steps)
        rationale = (
            f"Method '{method.name}' (from {method.origin}) transferred by shape, "
            f"not content: its steps re-read against the target."
        )
        return MethodTransfer(
            method_id=method.id,
            method_name=method.name,
            target_id=target_id,
            mapped_steps=mapped,
            rationale=rationale,
        )

    @staticmethod
    def _phrase(llm: LLMClient, target_text: str, step: str) -> str:
        return llm.phrase_transfer(target_text, step)


# --------------------------------------------------------------------------- #
# Loop-closer: grow Layer 9 from a successful process.
# --------------------------------------------------------------------------- #


def extract_method(
    run: CreativeRun, winning_candidate_id: str, *, name: str | None = None
) -> Method:
    """Mine a new, content-free method out of a process that worked.

    When a human marks a candidate as successful, the *sequence of moves* that led
    to it (which space was opened, which wild move fired, which method disciplined
    it) is itself a reusable Denkbewegung. We abstract that sequence into steps with
    no surviving content, tag it with the affinities that were in play, and return
    it for ``MethodLibrary.add``.

    This realises the user's loop: "Layer 9 extracts abstract methods from earlier
    successful thinking processes." A ``name`` may be supplied by the human; if not,
    one is derived from the (axis, move) shape - which also makes extraction
    idempotent, since the same shape yields the same method id.
    """
    cand = next((c for c in run.candidates if c.id == winning_candidate_id), None)
    if cand is None:
        raise ValueError(f"unknown candidate id: {winning_candidate_id}")

    space = next((s for s in run.spaces if s.id == cand.space_id), None)
    variant = next((v for v in run.variants if v.id == cand.variant_id), None)

    affinities: tuple[Affinity, ...] = space.affinities if space else (Affinity.ANALOGY,)
    move_label = variant.move.value if variant else "free variation"
    axis_label = space.axis if space else "an open region"

    steps = (
        f"Open the under-worked region characterised by '{axis_label}'.",
        f"Generate variation there using a '{move_label}' move, without demanding truth yet.",
        "Discipline the wildest variant with a transferred method, then keep what is testable.",
    )
    return Method(
        name=name or f"learned_{axis_label}_{move_label}",
        origin="kevin",
        summary="A process pattern abstracted from a creative run that a human marked as working.",
        steps=steps,
        affinities=affinities,
    )
