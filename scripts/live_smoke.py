#!/usr/bin/env python
"""A bounded, real-LLM smoke test for the full Kevin pipeline.

Run offline (MockLLM) for a structural check, or against real DeepSeek in CI:

    KEVIN_USE_REAL_LLM=1 DEEPSEEK_API_KEY=sk-... python scripts/live_smoke.py

It runs one creative pass with ``top_spaces=1`` to keep token use modest, then
asserts the invariants that only hold when the *whole* pipeline works end to end -
in particular that method transfer actually fired (the stage that silently broke
under a non-deterministic LLM before the explore-once fix).

If the real LLM is requested but no key is present (e.g. a fork PR, where secrets
are unavailable), it skips cleanly with exit 0 rather than failing the build.
"""

from __future__ import annotations

import os
import sys
import time

from kevin import Kevin, Problem
from kevin.human_gate import build_briefing


def main() -> int:
    real = os.getenv("KEVIN_USE_REAL_LLM") == "1"
    has_key = bool(
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY2")
        or os.getenv("OPENAI_API_KEY")
    )
    if real and not has_key:
        print("No LLM key present (e.g. fork PR) - skipping live smoke test.")
        return 0

    problem = Problem(
        "how do we make onboarding feel less like paperwork?",
        domain="people-ops",
        constraints=("no extra headcount",),
        known_approaches=("checklists", "a welcome email"),
    )

    start = time.time()
    run = Kevin().run(problem, top_spaces=1)
    elapsed = time.time() - start

    mode = "DeepSeek" if real else "MockLLM"
    print(
        f"[{mode}] {elapsed:.1f}s  spaces={len(run.spaces)} routed={len(run.chosen_spaces)} "
        f"variants={len(run.variants)} transfers={len(run.transfers)} "
        f"candidates={len(run.candidates)}"
    )
    if run.transfers:
        print("methods transferred:", ", ".join(t.method_name for t in run.transfers))

    # Invariants the full pipeline must satisfy regardless of the language model.
    assert run.spaces, "no solution spaces were explored"
    assert run.chosen_spaces, "nothing was routed"
    assert set(run.chosen_spaces) <= {s.id for s in run.spaces}, \
        "a routed space is missing from the explored set (explore ran twice?)"
    assert run.variants, "the wild brother produced no variants"
    assert run.transfers, "no Layer-9 method transfer fired"
    assert run.candidates, "no candidates were produced"
    assert len(run.evaluations) == len(run.candidates), "every candidate must be evaluated"

    briefing = build_briefing(run)
    print(f"promising={len(briefing.promising)} tentative={len(briefing.tentative)}")
    print("OK: routed -> varied -> transferred methods -> selected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
