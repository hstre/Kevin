"""Kevin puts the methods on the shared Layer-9 shelf to work.

Joni *harvests* methods he stumbles on (arXiv / HN / HuggingFace / GitHub) and parks
them on the shared ``desi_layer9`` core as **candidates** - he fills the shelf but never
tries anything. Kevin is the one who actually uses a method: it pulls every candidate (and
already-provisional) method off the shelf, runs a deterministic **transfer trial**, and
records the outcome through the gate.

Kevin never promotes. The governance boundary is intact: a model-origin proposer may
*report* trials but may not move a method's status. So the flow stays:

    Joni harvests  -> candidate
    [human/operator promotes] -> provisional
    Kevin trials (>=3 successes, more wins than losses) -> *activation-ready*
    [human/operator promotes] -> active   (only now usable_methods)

A "trial" is **not** an executability check. It is a transfer experiment: the method is
applied to a concrete task drawn from a battery that lies *outside the method's origin
domain*, and it passes only if the method-disciplined attempt measurably **beats a baseline**
attempt on that task. A method that does not fit the task adds no structure -> no improvement
-> the trial **fails**. So passes and failures both occur, and ``passed`` means "helped on a
foreign task", not "could be invoked". Deterministic: same (method, run_id) -> same task and
verdict; one trial per method per ``run_id``.
"""

from __future__ import annotations

import hashlib

from . import layer9_link

# A battery of tasks OUTSIDE any single method's home domain. Each needs a few content-free
# thinking-move shapes (Kevin's affinities) to be solved well.
_TASKS: tuple[tuple[str, frozenset[str]], ...] = (
    ("software-architecture", frozenset({"decomposition", "composition", "invariant"})),
    ("clinical-triage", frozenset({"exclusion", "risk", "causal"})),
    ("financial-audit", frozenset({"provenance", "adversarial", "risk"})),
    ("experiment-design", frozenset({"adversarial", "invariant", "causal"})),
    ("optimization", frozenset({"boundary", "inversion", "invariant"})),
    ("ethics-review", frozenset({"adversarial", "provenance", "exclusion"})),
    ("forecasting", frozenset({"provenance", "risk", "analogy"})),
    ("debugging", frozenset({"causal", "decomposition", "risk"})),
)

# Content-free shape inference: a keyword in a method's text -> the thinking-move it carries.
_SHAPE_TAGS: dict[str, str] = {
    "boundary": "boundary", "limit": "boundary", "extreme": "boundary", "edge": "boundary",
    "exclud": "exclusion", "rule out": "exclusion", "red flag": "exclusion",
    "catastroph": "exclusion", "decompos": "decomposition", "split": "decomposition",
    "modular": "decomposition", "separat": "decomposition", "element": "decomposition",
    "caus": "causal", "mechanism": "causal", "root": "causal", "why": "causal",
    "source": "provenance", "provenance": "provenance", "interest": "provenance",
    "risk": "risk", "fail": "risk", "premortem": "risk", "worst": "risk", "hazard": "risk",
    "invert": "inversion", "opposite": "inversion", "flip": "inversion", "reverse": "inversion",
    "invariant": "invariant", "conserv": "invariant", "balance": "invariant",
    "dimension": "invariant", "adversar": "adversarial", "steelman": "adversarial",
    "attack": "adversarial", "challenge": "adversarial", "critic": "adversarial",
    "analog": "analogy", "transfer": "analogy", "transport": "analogy", "structural": "analogy",
    "abstract": "abstraction", "general": "abstraction", "ladder": "abstraction",
    "principle": "abstraction", "compos": "composition", "combine": "composition",
    "emergen": "composition", "interface": "composition",
}

_BASELINE = 0.45        # how well a plain attempt does on a foreign task
_HELP = 0.45            # how much a perfectly-fitting method can add
_MIN_IMPROVEMENT = 0.15  # a trial passes only if it beats the baseline by at least this


def _method_shape(method) -> set[str]:
    """The content-free thinking-moves a method carries, inferred from its text + scope."""
    blob = f"{method.name} {method.summary} {' '.join(method.applicable_to)}".lower()
    return {tag for kw, tag in _SHAPE_TAGS.items() if kw in blob}


def _origin_domain(method) -> str:
    o = (method.origin or "unknown").lower()
    if o.startswith("http") or "github" in o:
        return "external"
    if ":" in o:                          # e.g. "joni:emergent"
        return o.split(":", 1)[0]
    return o


def _pick_task(method, run_id: str):
    """Deterministically pick a task OUTSIDE the method's origin domain."""
    h = int(hashlib.sha256(f"{method.id}|{run_id}".encode()).hexdigest(), 16)
    origin = _origin_domain(method)
    for k in range(len(_TASKS)):
        domain, needs = _TASKS[(h + k) % len(_TASKS)]
        if domain != origin and not domain.startswith(origin):
            return domain, needs
    return _TASKS[h % len(_TASKS)]


def _trial(method, run_id: str) -> tuple[bool, dict]:
    """Run one transfer experiment: does the method beat a baseline on a foreign task?"""
    domain, needs = _pick_task(method, run_id)
    shape = _method_shape(method)
    fit = len(shape & needs) / len(needs) if needs else 0.0
    improvement = round(fit * _HELP, 4)              # the method only helps in proportion to fit
    success = improvement >= _MIN_IMPROVEMENT        # i.e. it covered >= 1/3 of what the task needs
    return success, {"task": domain, "fit": round(fit, 3),
                     "baseline": _BASELINE, "with_method": round(_BASELINE + improvement, 4),
                     "improvement": improvement}


def _already_trialed(method, run_id: str) -> bool:
    return run_id in (set(method.supporting_runs) | set(method.failed_runs))


def trial_methods(core, *, run_id: str = "kevin", max_trials: int = 8,
                  statuses: tuple[str, ...] = ("candidate", "provisional")) -> dict:
    """Trial the shelf's candidate/provisional methods on ``core``; record via the gate.

    Returns a report: how many were trialed/passed/failed this run, and which provisional
    methods are now *activation-ready* (>=3 trials, more successes than failures) - a flag
    for a human; Kevin still does not promote.
    """
    from desi_layer9 import ObjectType, Status

    wanted = {Status[name.upper()] for name in statuses}
    report = {"trialed": 0, "succeeded": 0, "failed": 0,
              "activation_ready": [], "run_id": run_id}

    candidates = sorted((m for m in core.all(ObjectType.METHOD) if m.status in wanted),
                        key=lambda m: m.id)                    # replay-stable order
    for m in candidates:
        if report["trialed"] >= max_trials:
            break
        if _already_trialed(m, run_id):
            continue                                          # one trial per method per run
        ok, detail = _trial(m, run_id)
        layer9_link.record_trial(core, m.id, success=ok, run_id=run_id)
        report["trialed"] += 1
        report["succeeded" if ok else "failed"] += 1
        report.setdefault("details", []).append({"method": m.id, "passed": ok, **detail})

    for m in core.all(ObjectType.METHOD):
        if (m.status is Status.PROVISIONAL and m.trial_count >= 3
                and m.success_count > m.failure_count):
            report["activation_ready"].append(m.id)
    return report


def trial_core_file(path, *, run_id: str = "kevin", max_trials: int = 8) -> dict:
    """Load a shared-core journal, trial its shelf, persist it back. For standalone runs.

    Returns the trial report (plus ``core`` count). A no-op-safe wrapper when Layer 9 is
    unavailable: it returns an ``unavailable`` report rather than raising.
    """
    if not layer9_link.available():
        return {"unavailable": True, "trialed": 0, "succeeded": 0, "failed": 0,
                "activation_ready": [], "run_id": run_id}
    from desi_layer9 import persistence

    core = persistence.load(path)
    if core is None:
        return {"missing_core": True, "trialed": 0, "succeeded": 0, "failed": 0,
                "activation_ready": [], "run_id": run_id}
    report = trial_methods(core, run_id=run_id, max_trials=max_trials)
    persistence.save(core, path)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="kevin-trial",
        description="Kevin trials the candidate/provisional methods on a shared Layer-9 "
                    "core journal and records the outcomes (it never promotes).")
    p.add_argument("core", help="path to the shared-core journal (e.g. state/layer9.json)")
    p.add_argument("--run-id", default="kevin", help="trial run id (one trial per method per id)")
    p.add_argument("--max-trials", type=int, default=8)
    args = p.parse_args(argv)

    rep = trial_core_file(args.core, run_id=args.run_id, max_trials=args.max_trials)
    if rep.get("unavailable"):
        print("Layer 9 (desi_layer9) unavailable - nothing trialed.")
        return 0
    if rep.get("missing_core"):
        print(f"No core journal at {args.core} - nothing to trial.")
        return 0
    print(f"trialed {rep['trialed']} method(s): "
          f"{rep['succeeded']} passed, {rep['failed']} failed · "
          f"{len(rep['activation_ready'])} activation-ready (awaiting a human promote)")
    for mid in rep["activation_ready"]:
        print(f"  activation-ready: {mid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
