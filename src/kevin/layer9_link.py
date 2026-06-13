"""Kevin's clean binding to the one authoritative Layer 9.

Kevin is a creativity module. It may *discover* and *abstract* methods, report trials and
failures, and hand over candidates with provenance - but it may never set authoritative
state. So instead of Kevin's old append-only methods JSONL (which made a method
authoritative on a single human click), Kevin now submits **method proposals** into the
shared ``desi_layer9`` core through its gate:

    Kevin run -> METHOD_PROPOSE (candidate) -> [trials] -> a human/operator promotes
                 -> provisional -> (more trials) -> active

The gate enforces that Kevin - a model-origin proposer - cannot promote, confirm, or
resolve anything. Promotion is somebody else's authority. This is a soft dependency:
without ``desi_layer9`` (or with ``KEVIN_USE_LAYER9`` unset) Kevin falls back to its
legacy local library, which is then only a demonstrator, not an authoritative store.
"""

from __future__ import annotations

import os


def available() -> bool:
    try:
        import desi_layer9  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def enabled() -> bool:
    return os.getenv("KEVIN_USE_LAYER9") == "1" and available()


def new_core():
    """A fresh authoritative core (or None if Layer 9 is unavailable)."""
    if not available():
        return None
    from desi_layer9 import Layer9
    return Layer9()


def propose_method(core, *, name: str, summary: str, steps, affinities=(),
                   origin: str = "kevin", run_id: str = "unknown") -> str:
    """Submit a Kevin-discovered method as a candidate. Returns the method id.

    Provenance is model-origin (Kevin's candidates are model-generated), so the gate
    keeps it a candidate and Kevin can never push it further by itself.
    """
    from desi_layer9 import Operator, ProposalType, make_proposal
    from desi_layer9.objects import Method
    from desi_layer9.provenance import Provenance

    core.submit(
        make_proposal(
            ProposalType.METHOD_PROPOSAL, Operator.METHOD_PROPOSE,
            payload={"name": name, "summary": summary, "steps": list(steps),
                     "origin": origin, "applicable_to": [str(a) for a in affinities]},
            proposer="kevin",
            provenance=Provenance.from_model(external=False, model_id="kevin", run_id=run_id),
        ),
        actor="kevin",
    )
    methods = [o for o in core.objects.values() if isinstance(o, Method)]
    return max(methods, key=lambda m: int(m.id.split("-")[1])).id


def record_trial(core, method_id: str, *, success: bool, run_id: str = "unknown"):
    """Report a trial outcome for a method (Kevin may report; it may not promote)."""
    from desi_layer9 import Operator, ProposalType, make_proposal
    from desi_layer9.provenance import Provenance

    return core.submit(
        make_proposal(
            ProposalType.METHOD_PROPOSAL, Operator.METHOD_TRIAL_RECORD,
            payload={"success": bool(success), "run_id": run_id}, proposer="kevin",
            provenance=Provenance.from_model(external=False, model_id="kevin"),
            target_objects=(method_id,)),
        actor="kevin",
    )


def usable_methods(core) -> list:
    """Methods a human/operator has promoted to provisional or active - the ones Kevin
    may actually transfer. Candidates are not yet usable."""
    from desi_layer9 import ObjectType, Status
    return [m for m in core.all(ObjectType.METHOD)
            if m.status in (Status.PROVISIONAL, Status.ACTIVE)]
