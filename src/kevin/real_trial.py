"""real_trial_protocol_v1 - a method trial that actually measures, instead of simulating.

The synthetic ``trial_runner`` scores transfer by keyword-shape overlap: it never runs a method on
real cases and carries no epistemic weight. This is the **generic scaffold** of the real protocol
that replaces it. The contract (and the reason it is trustworthy) is:

  * a **frozen task set** (content hashed, so the set is auditable and cannot drift);
  * a **baseline** solver (the task *without* the method) and an **intervention** solver (the task
    *with* the method) measured on the SAME cases;
  * a **predefined metric** fixed before the run - the pass/fail **decision rests only on this
    metric and a threshold**, never on a model's opinion;
  * **repetitions** (the slot is here even when the example is deterministic);
  * a **negative control**: a sham method that should NOT help - if it does, the apparatus is
    measuring noise and the trial is inconclusive;
  * **stored outputs** + a **result with full provenance** (metric, baseline, intervention,
    negative control, delta, repetitions, uncertainty, the model+config used to *process* cases).

Models may **process** cases (annotate / extract / draft) and produce artefacts; the **decision is
deterministic** on the metric. v1 ships with one simple, fully-checkable example task whose solvers
are plain Python (no model needed) so the protocol itself is testable offline; a real task plugs in
model-backed solvers without changing the runner.

Kevin still never promotes: a result is recorded as a provisional, evidence-bearing trial through
the gate (``layer9_link``); status changes remain human-gated.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

_NUM = re.compile(r"\d+(?:\.\d+)?")
_NEGATION = re.compile(r"\b(not|no|never|n't)\b")
_WORD = re.compile(r"[a-z0-9]+")

EVALUATION_MODE = "real_trial_protocol_v1"


@dataclass(frozen=True)
class TaskCase:
    """One frozen case: the input the solvers see, and the gold the metric checks against."""

    id: str
    payload: dict
    gold: dict


@dataclass(frozen=True)
class TaskSet:
    """A frozen, content-addressed battery of cases. The sha makes drift detectable."""

    id: str
    version: str
    cases: tuple[TaskCase, ...]

    def sha(self) -> str:
        blob = json.dumps([asdict(c) for c in self.cases], sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class TrialResult:
    """The evidence-bearing outcome - real, but still provisional (no human confirmation)."""

    method_id: str
    task_set: str
    task_set_sha: str
    metric: str
    lower_is_better: bool
    baseline: float
    intervention: float
    negative_control: float
    delta: float                       # signed improvement of intervention over baseline
    repetitions: int
    direction: str                     # "positive" | "negative" | "none"
    uncertainty: str                   # "low" | "medium" | "high"
    passed: bool                       # decided ONLY by the predefined metric + threshold
    processor_model: str               # model used to PROCESS cases (artefacts only), or "none"
    config: dict = field(default_factory=dict)
    evaluation_mode: str = EVALUATION_MODE
    epistemic_weight: str = "provisional"

    def to_dict(self) -> dict:
        return asdict(self)


# A solver maps a case payload -> a structured answer dict the metric can score.
Solver = Callable[[dict], dict]
# A metric maps (answers, cases) -> a single number (lower or higher is better, declared).
Metric = Callable[[Sequence[dict], Sequence[TaskCase]], float]


def _measure(solver: Solver, cases: Sequence[TaskCase], metric: Metric, repetitions: int) -> float:
    """Run a solver over the frozen cases ``repetitions`` times and average the metric. The slot is
    real even when the solver is deterministic (then every repetition agrees)."""
    vals = []
    for _ in range(max(1, repetitions)):
        answers = [solver(c.payload) for c in cases]
        vals.append(metric(answers, cases))
    return round(sum(vals) / len(vals), 6)


def run_real_trial(*, method_id: str, task_set: TaskSet, metric_name: str, metric: Metric,
                   baseline: Solver, intervention: Solver, negative_control: Solver,
                   lower_is_better: bool = True, repetitions: int = 5,
                   min_effect: float = 0.1, processor_model: str = "none",
                   config: dict | None = None) -> TrialResult:
    """Run the protocol and DECIDE by the metric alone. ``passed`` requires (a) the intervention
    beats the baseline by at least ``min_effect`` in the declared direction AND (b) the negative
    control does NOT (else the rig is measuring noise -> inconclusive, not passed)."""
    b = _measure(baseline, task_set.cases, metric, repetitions)
    i = _measure(intervention, task_set.cases, metric, repetitions)
    n = _measure(negative_control, task_set.cases, metric, repetitions)
    # signed improvement, oriented so that positive == better
    delta = (b - i) if lower_is_better else (i - b)
    neg_delta = (b - n) if lower_is_better else (n - b)
    real_effect = delta >= min_effect
    control_clean = neg_delta < min_effect           # the sham must NOT show the same effect
    passed = bool(real_effect and control_clean)
    direction = "positive" if delta >= min_effect else ("negative" if delta <= -min_effect
                                                         else "none")
    # uncertainty: clean control + clear effect -> low; effect but noisy control -> high
    uncertainty = "low" if (real_effect and control_clean) else (
        "high" if (real_effect and not control_clean) else "medium")
    return TrialResult(
        method_id=method_id, task_set=f"{task_set.id}@{task_set.version}",
        task_set_sha=task_set.sha(), metric=metric_name, lower_is_better=lower_is_better,
        baseline=b, intervention=i, negative_control=n, delta=round(delta, 6),
        repetitions=repetitions, direction=direction, uncertainty=uncertainty, passed=passed,
        processor_model=processor_model, config=config or {})


def record(core, result: TrialResult, *, run_id: str = "kevin-real") -> str | None:
    """Record the trial as an evidence-bearing report through the gate (Kevin never promotes).
    Returns the trial outcome id, or None if Layer 9 is unavailable. The full result (with
    provenance) is attached so the proposal is auditable."""
    from . import layer9_link
    if not layer9_link.available():
        return None
    # The gate's METHOD_TRIAL_RECORD payload is fixed (success, run_id) - the full result with
    # provenance is the stored artefact (``result.to_dict()``), persisted by the caller.
    return layer9_link.record_trial(
        core, result.method_id, success=result.passed, run_id=run_id)


# --------------------------------------------------------------------------------------------- #
# v1 example task set: a simple, fully-checkable conflict-classification task. A pair of claims is
# either a near-duplicate (same stance, differs only in a number) or a real contradiction (opposite
# polarity). The metric is the TWO-SIDED misclassification_rate (lower is better). The PLAUSIBLE
# lexical-similarity baseline is fooled by negation; the "contradiction-first review" method checks
# polarity first and beats it - but NOT perfectly (it fails a double-negation case), so the win is
# earned. The negative control is a structureless hash-parity guess (could spuriously help, so it is
# a real noise floor). All deterministic - the protocol, not a model, is what is under test here.
# --------------------------------------------------------------------------------------------- #
def example_task_set() -> TaskSet:
    # a small subset of the same KIND of cases as the frozen Joni set (numeric paraphrases,
    # pure-negation contradictions, and a double-negation duplicate the method gets WRONG).
    cases = (
        TaskCase("d1", {"a": "the thread had 31 exchanges", "b": "the thread had 34 exchanges"},
                 {"label": "duplicate"}),
        TaskCase("d2", {"a": "we saw 12 retries here", "b": "we saw 9 retries here"},
                 {"label": "duplicate"}),
        TaskCase("d3", {"a": "the corpus has 3 sources", "b": "the corpus has 5 sources"},
                 {"label": "duplicate"}),
        TaskCase("c1", {"a": "the cache is safe", "b": "the cache is not safe"},
                 {"label": "contradiction"}),
        TaskCase("c2", {"a": "the proof is complete", "b": "the proof is not complete"},
                 {"label": "contradiction"}),
        TaskCase("n1", {"a": "the cache is not unsafe", "b": "the cache is safe"},
                 {"label": "duplicate"}),                      # double negation: method FAILS this
    )
    return TaskSet(id="conflict_classification_example", version="v1", cases=cases)


def _strip_numbers(s: str) -> str:
    return _NUM.sub("#", (s or "").lower()).strip()


def _tokens(s: str) -> set:
    return set(_WORD.findall((s or "").lower()))


def baseline_solver(payload: dict) -> dict:
    """A PLAUSIBLE naive baseline (not a strawman): lexical similarity. High token overlap ->
    'duplicate', else 'contradiction'. It is genuinely useful on numeric paraphrases but - the
    point of the trial - it is fooled by negation: 'X is safe' vs 'X is not safe' overlap heavily,
    so it wrongly calls a real contradiction a duplicate."""
    a, b = payload.get("a", ""), payload.get("b", "")
    ta, tb = _tokens(a), _tokens(b)
    jacc = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return {"label": "duplicate" if jacc >= 0.6 else "contradiction"}


def method_solver(payload: dict) -> dict:
    """contradiction-first review: check POLARITY first; only an opposite-polarity pair is a
    contradiction; same polarity + identical once numbers are stripped -> duplicate. Beats the
    lexical baseline on negation, but it is NOT perfect - a DOUBLE negation ('not unsafe' vs
    'safe') reads as opposite polarity, so the method wrongly calls an agreeing pair a
    contradiction. That residual error is real and shows up in the metric (the pass is earned,
    not preordained)."""
    a, b = payload.get("a", ""), payload.get("b", "")
    pol_a, pol_b = bool(_NEGATION.search(a)), bool(_NEGATION.search(b))
    if pol_a != pol_b:
        return {"label": "contradiction"}
    if _strip_numbers(a) == _strip_numbers(b):
        return {"label": "duplicate"}
    return {"label": "contradiction"}


def negative_control_solver(payload: dict) -> dict:
    """A STRUCTURELESS sham that COULD spuriously help: it labels by the parity of a hash of the
    text (no real signal, but not a constant - so it sometimes gets cases right by luck). If this
    matched the method's gain over baseline, the apparatus would be measuring noise and the trial
    is inconclusive, not a pass. A constant guess would be a vacuous control (it can never help);
    this one can, which is exactly what makes it a real noise floor."""
    a, b = payload.get("a", ""), payload.get("b", "")
    h = int(hashlib.sha256(f"{a}|{b}".encode()).hexdigest(), 16)
    return {"label": "duplicate" if h % 2 == 0 else "contradiction"}


def misclassification_rate(answers: Sequence[dict], cases: Sequence[TaskCase]) -> float:
    """TWO-SIDED error: fraction of ALL cases mislabelled (both false-contradictions AND
    false-duplicates). A one-sided rate would let a constant 'always duplicate' solver score a
    perfect 0; this cannot be gamed by collapsing to a single label. Lower is better."""
    if not cases:
        return 0.0
    wrong = sum(1 for ans, c in zip(answers, cases, strict=False)
                if ans.get("label") != c.gold["label"])
    return round(wrong / len(cases), 6)


def run_example() -> TrialResult:
    """Run the v1 example end to end - used by tests and as the reference trial."""
    return run_real_trial(
        method_id="contradiction-first-review", task_set=example_task_set(),
        metric_name="misclassification_rate", metric=misclassification_rate,
        baseline=baseline_solver, intervention=method_solver,
        negative_control=negative_control_solver, lower_is_better=True, repetitions=5,
        min_effect=0.2, processor_model="none")


# --------------------------------------------------------------------------------------------- #
# The first CONCRETE trial: frozen_joni_conflict_cases_v1. A curated, hand-labelled battery
# representative of the conflict pairs Joni actually produces. Three case families, chosen so the
# trial genuinely DISCRIMINATES (the outcome is not preordained):
#   * numeric paraphrases (gold=duplicate) - both baseline and method get these right;
#   * pure-negation contradictions ('X' vs 'X not') (gold=contradiction) - the LEXICAL baseline is
#     fooled (high overlap -> 'duplicate'), the polarity method is right -> this is where the method
#     earns its win;
#   * double-negation duplicates ('not unsafe' vs 'safe') (gold=duplicate) - the method is WRONG
#     here (reads opposite polarity), so the method's error is non-zero and the pass is earned.
# Gold labels are the curator's, NOT the method's (non-circular). Measured TWO-SIDED
# (misclassification_rate), with a structureless hash-parity negative control as the noise floor.
# --------------------------------------------------------------------------------------------- #
def frozen_joni_conflict_cases_v1() -> TaskSet:
    cases = (
        # numeric paraphrases -> gold = duplicate (the C-71/C-87 case: 31 vs 34 exchanges)
        TaskCase("fc1", {"a": "the thread had 31 exchanges before resolution",
                         "b": "the thread had 34 exchanges before resolution"},
                 {"label": "duplicate"}),
        TaskCase("fc2", {"a": "session contamination dominates 62 percent of variance",
                         "b": "session contamination dominates 67 percent of variance"},
                 {"label": "duplicate"}),
        TaskCase("fc3", {"a": "the model needed 12 retries on the benchmark",
                         "b": "the model needed 9 retries on the benchmark"},
                 {"label": "duplicate"}),
        TaskCase("fc4", {"a": "routing cut p95 latency to 120 ms",
                         "b": "routing cut p95 latency to 140 ms"},
                 {"label": "duplicate"}),
        TaskCase("fc5", {"a": "the corpus contains 3 independent sources",
                         "b": "the corpus contains 5 independent sources"},
                 {"label": "duplicate"}),
        # pure-negation contradictions ('X' vs 'X not') -> gold = contradiction; the lexical
        # baseline is fooled by the heavy overlap, the polarity method is not.
        TaskCase("tc1", {"a": "the cache is safe under load",
                         "b": "the cache is not safe under load"},
                 {"label": "contradiction"}),
        TaskCase("tc2", {"a": "the result is statistically significant",
                         "b": "the result is not statistically significant"},
                 {"label": "contradiction"}),
        TaskCase("tc3", {"a": "the model is well calibrated",
                         "b": "the model is not well calibrated"},
                 {"label": "contradiction"}),
        TaskCase("tc4", {"a": "the routing layer is correct",
                         "b": "the routing layer is not correct"},
                 {"label": "contradiction"}),
        TaskCase("tc5", {"a": "the proof is complete",
                         "b": "the proof is not complete"},
                 {"label": "contradiction"}),
        # double-negation duplicates (they AGREE) -> gold = duplicate; the polarity method FAILS
        # these, so the method's error is non-zero and the win is earned, not preordained.
        TaskCase("dn1", {"a": "the cache is not unsafe under load",
                         "b": "the cache is safe under load"},
                 {"label": "duplicate"}),
        TaskCase("dn2", {"a": "the result is not insignificant",
                         "b": "the result is significant"},
                 {"label": "duplicate"}),
    )
    return TaskSet(id="frozen_joni_conflict_cases", version="v1", cases=cases)


def run_joni_conflict_trial() -> TrialResult:
    """The first concrete real trial on Joni's own kind of material. Deterministic in v1 (the
    PROTOCOL is what is under test); a model-backed intervention solver (Granite annotating each
    pair) plugs into the same runner without changing the decision, which stays on the metric."""
    return run_real_trial(
        method_id="contradiction-first-review", task_set=frozen_joni_conflict_cases_v1(),
        metric_name="misclassification_rate", metric=misclassification_rate,
        baseline=baseline_solver, intervention=method_solver,
        negative_control=negative_control_solver, lower_is_better=True, repetitions=5,
        min_effect=0.2, processor_model="none")
