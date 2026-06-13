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

A "trial" here is a transfer attempt judged purely from the method's own content - can
this thing be re-applied as a content-free move? It is deterministic (same method -> same
verdict), so accumulation across Kevin's real runs is honest: one trial per method per
``run_id``, never inflated by re-running the same id.
"""

from __future__ import annotations

from . import layer9_link

# Shape-words a transferable thinking-move tends to carry. Content-free matching: we are
# looking for the *form* of a reusable technique, not any particular domain.
_SHAPE_WORDS = frozenset({
    "method", "technique", "approach", "algorithm", "framework", "procedure", "strategy",
    "heuristic", "pipeline", "protocol", "pattern", "model", "analysis", "search",
    "routing", "transfer", "decomposition", "reduction", "optimization", "inference",
    "scoring", "selection", "detection", "mapping", "planning", "reasoning", "library",
    "toolkit", "scheme", "recipe", "how to",
})

# A method passes a trial when it carries enough shape to be re-applied.
_PASS_AT = 0.50


def _transfer_score(method) -> float:
    """How transferable is this method, from its own content alone? Deterministic [0,1].

    Three additive signals, mirroring Kevin's selection spirit:
      * connectivity - it declares a domain it bridges to (``applicable_to``);
      * structure    - its text speaks in method-shape language;
      * substance    - it carries more than a bare restated name.
    """
    text = f"{method.name} {method.summary}".lower()
    summary = (method.summary or "").strip()
    connectivity = 0.40 if method.applicable_to else 0.0
    structure = 0.35 if any(w in text for w in _SHAPE_WORDS) else 0.0
    substance = 0.25 if (len(summary.split()) >= 4
                         and summary.lower() != method.name.strip().lower()) else 0.0
    return round(connectivity + structure + substance, 4)


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
        ok = _transfer_score(m) >= _PASS_AT
        layer9_link.record_trial(core, m.id, success=ok, run_id=run_id)
        report["trialed"] += 1
        report["succeeded" if ok else "failed"] += 1

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
