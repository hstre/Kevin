"""Kevin - HTTP/API surface (FastAPI).

A thin web layer over the same engines. It adds no creativity logic of its own;
it runs the orchestrator, serialises the ``CreativeRun`` plus the human briefing,
and - crucially - lets a human mark a candidate as *worked* so the method library
grows from real runs (the loop the design calls for). Like the rest of the
ecosystem, the API is a *port*, not the system.

    uvicorn kevin.api:app --reload      # or: kevin-serve

Endpoints (all JSON unless noted):
    GET  /              -> the single-page UI (static HTML, no build step)
    GET  /health        -> liveness
    GET  /api/methods   -> the Layer-9 method library (seed + learned)
    POST /api/run       -> run one creative pass; returns spaces, stats, verdicts
    POST /api/promote   -> mark a candidate as worked; learn a method from the run
    GET  /api/verdicts  -> the closed verdict set

The library, the Layer-9 ledger and the orchestrator are shared process-wide, so a
method learned from one run is immediately available to the next - and persists
across restarts via the append-only ledger (see ``layer9.py``).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .human_gate import build_briefing
from .layer9 import Layer9Store
from .method_library import SEED_METHODS, MethodLibrary, extract_method
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
# Shared, process-wide state
#   * the ledger persists methods learned from runs (append-only, JSONL);
#   * the library is seed + everything the ledger has ever learned;
#   * one Kevin shares that library, so learning feeds straight back into runs;
#   * a bounded run cache lets /api/promote find the run a human is judging.
# --------------------------------------------------------------------------- #

_STORE = Layer9Store()
_LIBRARY = MethodLibrary(SEED_METHODS + tuple(_STORE.load_methods()))
_KEVIN = Kevin(library=_LIBRARY)

_RUN_CACHE_MAX = 200
_RUNS: OrderedDict[str, CreativeRun] = OrderedDict()


def _remember(run: CreativeRun) -> None:
    _RUNS[run.id] = run
    _RUNS.move_to_end(run.id)
    while len(_RUNS) > _RUN_CACHE_MAX:
        _RUNS.popitem(last=False)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class RunRequest(BaseModel):
    statement: str = Field(..., min_length=3, description="the problem you want creative help with")
    domain: str = "general"
    constraints: list[str] = Field(default_factory=list)
    known_approaches: list[str] = Field(default_factory=list)
    top_spaces: int = Field(2, ge=1, le=5)


class PromoteRequest(BaseModel):
    run_id: str = Field(..., description="the run the candidate came from")
    candidate_id: str = Field(..., description="the candidate the human marks as worked")
    name: str | None = Field(None, description="optional human name for the learned method")


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
        "id": cand.id,
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


def _method_dict(m) -> dict:
    return {
        "name": m.name,
        "origin": m.origin,
        "summary": m.summary,
        "steps": list(m.steps),
        "affinities": [a.value for a in m.affinities],
        "learned": m.origin == "kevin",
    }


def _library_info() -> dict:
    methods = _LIBRARY.all()
    learned = sum(1 for m in methods if m.origin == "kevin")
    return {"total": len(methods), "learned": learned, "seed": len(methods) - learned}


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
        "run_id": run.id,
        "space_prediction": run.space_prediction,
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
        "library": _library_info(),
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
    return {"methods": [_method_dict(m) for m in _LIBRARY.all()], "library": _library_info()}


@app.post("/api/run")
def run(req: RunRequest) -> dict:
    problem = Problem(
        statement=req.statement,
        domain=req.domain,
        constraints=tuple(req.constraints),
        known_approaches=tuple(req.known_approaches),
    )
    creative_run = _KEVIN.run(problem, top_spaces=req.top_spaces)
    _remember(creative_run)
    return _serialise(creative_run)


@app.post("/api/promote")
def promote(req: PromoteRequest) -> dict:
    """Mark a candidate as having worked, and grow Layer 9 from the run.

    The machine never decides this - a human does. The human's verdict is what mines
    a new content-free method out of the process and persists it to the ledger, so
    the library carries the shape of every run a person found valuable.
    """
    run = _RUNS.get(req.run_id)
    if run is None:
        raise HTTPException(404, detail="unknown or expired run_id - re-run the problem")
    try:
        method = extract_method(run, req.candidate_id, name=req.name)
    except ValueError as exc:
        raise HTTPException(404, detail=str(exc)) from exc

    already = method.id in {m.id for m in _LIBRARY.all()}
    if not already:
        _LIBRARY.add(method)
        _STORE.append(method, run_id=req.run_id, candidate_id=req.candidate_id)

    return {
        "learned": _method_dict(method),
        "already_known": already,  # idempotent: re-learning the same shape is a no-op
        "library": _library_info(),
    }


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
