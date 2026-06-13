"""Kevin CLI demonstrator.

    python -m kevin "how do we make onboarding feel less like paperwork?" \
        --constraint "no extra headcount" --known "checklists" --known "a welcome email"

Runs one full creative pass offline (deterministic MockLLM) and prints the human
briefing plus a trace of how the possibility space was routed.
"""

from __future__ import annotations

import argparse

from .human_gate import build_briefing
from .models import Problem
from .orchestrator import Kevin


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kevin",
        description="Kevin - route to creativity: unexplored spaces -> wild variation "
        "-> method transfer -> epistemic selection -> human direction.",
    )
    p.add_argument("statement", help="the problem you want creative help with")
    p.add_argument("--domain", default="general")
    p.add_argument("--constraint", action="append", default=[], dest="constraints")
    p.add_argument("--known", action="append", default=[], dest="known",
                   help="an already-tried approach (repeatable); makes Kevin avoid crowded space")
    p.add_argument("--top-spaces", type=int, default=2)
    p.add_argument("--trace", action="store_true", help="show the full routing trace")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    problem = Problem(
        statement=args.statement,
        domain=args.domain,
        constraints=tuple(args.constraints),
        known_approaches=tuple(args.known),
    )
    run = Kevin().run(problem, top_spaces=args.top_spaces)

    if args.trace:
        print("=" * 72)
        print("ROUTING TRACE")
        print("=" * 72)
        print("\nSolution spaces (sorted by opportunity = plausibility * (1 - exploration)):")
        for s in run.spaces:
            chosen = "  <- ROUTED" if s.id in run.chosen_spaces else ""
            print(f"  [{s.opportunity:.2f}] {s.label}: plaus={s.plausibility} "
                  f"explored={s.exploration}{chosen}")
        print(f"\nWild variants generated: {len(run.variants)}")
        print(f"Method transfers (Layer 9): {len(run.transfers)}")
        for t in run.transfers:
            print(f"  via '{t.method_name}'")
        print(f"Candidates evaluated: {len(run.candidates)}\n")

    print("=" * 72)
    print(build_briefing(run).render())
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
