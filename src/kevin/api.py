"""Kevin - HTTP/API surface (FastAPI).

A thin web layer over the same engines. It adds no creativity logic of its own;
it runs the orchestrator and serialises the ``CreativeRun`` plus the human briefing
for a browser. Like the rest of the ecosystem, the API is a *port*, not the system.

    uvicorn kevin.api:app --reload      # or: kevin-serve

Endpoints (all JSON unless noted):
    GET  /              -> the single-page UI (static HTML, no build step)
    GET  /health        -> liveness
    GET  /api/methods   -> the Layer-9 method library (content-free)
    POST /api/run       -> run one creative pass; returns spaces, stats, verdicts
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .human_gate import build_briefing
from .method_library import MethodLibrary
from .models import Candidate, CreativeRun, Evaluation, Problem, Verdict
from .orchestrator import Kevin

app = FastAPI(
    title="Kevin",
    description="A creativity-routing architecture: unexplored spaces -> wild variation "
    "-> method transfer -> epistemic selection -> human direction.",
    version="0.1.0",
)

_WEB = Path(__file__).parent / "web"


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class RunRequest(BaseModel):
    statement: str = Field(..., min_length=3, description="the problem you want creative help with")
    domain: str = "general"
    constraints: list[str] = Field(default_factory=list)
    known_approaches: list[str] = Field(default_factory=list)
    top_spaces: int = Field(2, ge=1, le=5)


def _space_dict(space, chosen_ids: set[str]) -> dict:
    return {
        "id": space.id,
        "label": space.label,
        "axis": space.axis,
        "description": space.description,
        "plausibility": space.plausibility,
        "exploration": space.exploration,
        "opportunity": space.opportunity,
        "affinities": [a.value for a in space.affinities],
        "routed": space.id in chosen_ids,
    }


def _verdict_dict(ev: Evaluation, cand: Candidate) -> dict:
    return {
        "content": cand.content,
        "method_disciplined": cand.method_disciplined,
        "wildness": cand.wildness,
        "score": ev.score,
        "coherence": ev.coherence,
        "testability": ev.testability,
        "connectivity": ev.connectivity,
        "novelty": ev.novelty,
        "reason": ev.reason,
    }


def _serialise(run: CreativeRun) -> dict:
    chosen = set(run.chosen_spaces)
    by_id = {c.id: c for c in run.candidates}
    buckets: dict[str, list[dict]] = {"promising": [], "tentative": [], "rejected": []}
    for ev in run.evaluations:
        cand = by_id.get(ev.candidate_id)
        if cand is None:
            continue
        buckets[ev.verdict.value].append(_verdict_dict(ev, cand))
    briefing = build_briefing(run)
    return {
        "problem": {
            "statement": run.problem.statement,
            "domain": run.problem.domain,
            "constraints": list(run.problem.constraints),
            "known_approaches": list(run.problem.known_approaches),
        },
        "spaces": [_space_dict(s, chosen) for s in run.spaces],
        "stats": {
            "spaces": len(run.spaces),
            "routed": len(run.chosen_spaces),
            "variants": len(run.variants),
            "transfers": len(run.transfers),
            "candidates": len(run.candidates),
        },
        "transfers": [{"method": t.method_name} for t in run.transfers],
        "decision_axes": list(briefing.decision_axes),
        "promising": buckets["promising"],
        "tentative": buckets["tentative"],
        "rejected": buckets["rejected"],
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_WEB / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "kevin", "version": app.version}


@app.get("/api/methods")
def methods() -> dict:
    lib = MethodLibrary()
    return {
        "methods": [
            {
                "name": m.name,
                "origin": m.origin,
                "summary": m.summary,
                "steps": list(m.steps),
                "affinities": [a.value for a in m.affinities],
            }
            for m in lib.all()
        ]
    }


@app.post("/api/run")
def run(req: RunRequest) -> dict:
    problem = Problem(
        statement=req.statement,
        domain=req.domain,
        constraints=tuple(req.constraints),
        known_approaches=tuple(req.known_approaches),
    )
    creative_run = Kevin().run(problem, top_spaces=req.top_spaces)
    return _serialise(creative_run)


# Keep the closed verdict set discoverable by the frontend / clients.
@app.get("/api/verdicts")
def verdicts() -> dict:
    return {"verdicts": [v.value for v in Verdict]}


def serve() -> None:  # pragma: no cover - entry point, exercised manually
    """``kevin-serve`` - run the app with uvicorn."""
    import os

    import uvicorn

    uvicorn.run(
        "kevin.api:app",
        host=os.getenv("KEVIN_HOST", "127.0.0.1"),
        port=int(os.getenv("KEVIN_PORT", "8000")),
        reload=bool(os.getenv("KEVIN_RELOAD")),
    )
