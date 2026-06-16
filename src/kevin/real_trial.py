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
# polarity). The metric is the false-contradiction rate (lower is better): how often near-duplicates
# are wrongly called contradictions. The "contradiction-first review" method checks polarity before
# similarity and so avoids the false contradictions a naive similarity baseline makes. The negative
# control ("alphabetise") is irrelevant and must not help. All deterministic - the protocol, not a
# model, is what is under test here.
# --------------------------------------------------------------------------------------------- #
def example_task_set() -> TaskSet:
    cases = (
        TaskCase("d1", {"a": "the thread had 31 exchanges", "b": "the thread had 34 exchanges"},
                 {"label": "duplicate"}),
        TaskCase("d2", {"a": "latency was 12 ms", "b": "latency was 15 ms"},
                 {"label": "duplicate"}),
        TaskCase("d3", {"a": "we saw 5 retries", "b": "we saw 8 retries"},
                 {"label": "duplicate"}),
        TaskCase("c1", {"a": "routing reduces latency", "b": "routing does not reduce latency"},
                 {"label": "contradiction"}),
        TaskCase("c2", {"a": "the cache is safe", "b": "the cache is not safe"},
                 {"label": "contradiction"}),
    )
    return TaskSet(id="conflict_classification_example", version="v1", cases=cases)


def _strip_numbers(s: str) -> str:
    return _NUM.sub("#", (s or "").lower()).strip()


def baseline_solver(payload: dict) -> dict:
    """Naive similarity: if the two texts are not identical, call it a contradiction. This is the
    mistake the review flagged - near-duplicates differing only in a number get mislabelled."""
    a, b = payload.get("a", ""), payload.get("b", "")
    return {"label": "duplicate" if a.strip().lower() == b.strip().lower() else "contradiction"}


def method_solver(payload: dict) -> dict:
    """contradiction-first review: check POLARITY first; only an opposite-polarity pair is a
    contradiction. Same polarity + identical once numbers are stripped -> duplicate."""
    a, b = payload.get("a", ""), payload.get("b", "")
    pol_a, pol_b = bool(_NEGATION.search(a)), bool(_NEGATION.search(b))
    if pol_a != pol_b:
        return {"label": "contradiction"}
    if _strip_numbers(a) == _strip_numbers(b):
        return {"label": "duplicate"}
    return {"label": "contradiction"}


def negative_control_solver(payload: dict) -> dict:
    """A sham 'method' that adds NO relevant structure: it ignores the content and always guesses
    'contradiction'. It must NOT improve over the baseline - if a structureless guess showed the
    same gain as the real method, the apparatus would be measuring noise, and the trial is
    inconclusive (not a pass)."""
    return {"label": "contradiction"}


def false_contradiction_rate(answers: Sequence[dict], cases: Sequence[TaskCase]) -> float:
    """Fraction of true duplicates wrongly labelled 'contradiction' (lower is better)."""
    dups = [(ans, c) for ans, c in zip(answers, cases, strict=False)
            if c.gold["label"] == "duplicate"]
    if not dups:
        return 0.0
    wrong = sum(1 for ans, _ in dups if ans.get("label") == "contradiction")
    return round(wrong / len(dups), 6)


def run_example() -> TrialResult:
    """Run the v1 example end to end - used by tests and as the reference trial."""
    return run_real_trial(
        method_id="contradiction-first-review", task_set=example_task_set(),
        metric_name="false_contradiction_rate", metric=false_contradiction_rate,
        baseline=baseline_solver, intervention=method_solver,
        negative_control=negative_control_solver, lower_is_better=True, repetitions=5,
        min_effect=0.34, processor_model="none")


# --------------------------------------------------------------------------------------------- #
# The first CONCRETE trial: frozen_joni_conflict_cases_v1. A curated, hand-labelled battery
# representative of the conflict pairs Joni actually produces - numeric paraphrases that the loop
# wrongly opened as HARD conflicts (the C-71/C-87 case: 31 vs 34 exchanges) versus real negations
# that genuinely contradict. The gold labels are assigned by the curator, NOT by the method under
# test (so the trial is not circular). It measures whether the "contradiction-first review" method
# (check polarity before similarity) reduces the false-contradiction rate versus the naive
# similarity baseline, with the structureless sham as a negative control.
# --------------------------------------------------------------------------------------------- #
def frozen_joni_conflict_cases_v1() -> TaskSet:
    cases = (
        # false conflicts: same stance, differ only in a number -> gold = duplicate
        TaskCase("fc1", {"a": "the thread had 31 exchanges before resolution",
                         "b": "the thread had 34 exchanges before resolution"},
                 {"label": "duplicate"}),
        TaskCase("fc2", {"a": "session contamination dominates 62% of model variance",
                         "b": "session contamination dominates 67% of model variance"},
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
        # real contradictions: opposite polarity -> gold = contradiction
        TaskCase("tc1", {"a": "local routing reduces latency",
                         "b": "local routing does not reduce latency"},
                 {"label": "contradiction"}),
        TaskCase("tc2", {"a": "episodic memory improves continuity",
                         "b": "episodic memory never improves continuity"},
                 {"label": "contradiction"}),
        TaskCase("tc3", {"a": "the cache is safe under load",
                         "b": "the cache is not safe under load"},
                 {"label": "contradiction"}),
        TaskCase("tc4", {"a": "distillation preserves calibration",
                         "b": "distillation does not preserve calibration"},
                 {"label": "contradiction"}),
    )
    return TaskSet(id="frozen_joni_conflict_cases", version="v1", cases=cases)


def run_joni_conflict_trial() -> TrialResult:
    """The first concrete real trial on Joni's own kind of material. Deterministic in v1 (the
    PROTOCOL is what is under test); a model-backed intervention solver (Granite annotating each
    pair) plugs into the same runner without changing the decision, which stays on the metric."""
    return run_real_trial(
        method_id="contradiction-first-review", task_set=frozen_joni_conflict_cases_v1(),
        metric_name="false_contradiction_rate", metric=false_contradiction_rate,
        baseline=baseline_solver, intervention=method_solver,
        negative_control=negative_control_solver, lower_is_better=True, repetitions=5,
        min_effect=0.34, processor_model="none")
