"""DESi link - use DESi to predict *where the solution spaces are*.

Kevin's first move is to find the plausible-but-unworked regions. DESi already has
machinery for exactly that: ``blind_spot_mapping.coverage`` does set-coverage analysis
(union / intersection / symmetric difference) to find the blind spots in a covered
space. We encode Kevin's structural axes as a coverage set and let DESi's real
``pair_coverage`` predict how much of the space is unexplored and which regions are open.

Opt in with ``KEVIN_USE_DESI=1`` and the ``desi-governance`` package importable (or a
``DESI_ROOT`` checkout). Absent or disabled, the space predictor falls back to the same
set arithmetic computed in-house - deterministic and replay-stable either way.
"""

from __future__ import annotations

import importlib
import os
import sys


def _desi():
    try:
        return importlib.import_module("desi.blind_spot_mapping.coverage")
    except Exception:  # noqa: BLE001
        root = os.getenv("DESI_ROOT")
        if root:
            for candidate in (os.path.join(root, "src"), root):
                if os.path.isdir(candidate) and candidate not in sys.path:
                    sys.path.insert(0, candidate)
            try:
                return importlib.import_module("desi.blind_spot_mapping.coverage")
            except Exception:  # noqa: BLE001
                return None
        return None


def available() -> bool:
    return _desi() is not None


def enabled() -> bool:
    return os.getenv("KEVIN_USE_DESI") == "1" and available()


def engine() -> str:
    return "DESi" if enabled() else "kevin-builtin"


def coverage(covered_ids: set[int], universe_ids: set[int]) -> dict | None:
    """Run DESi's real blind-spot coverage over Kevin's axis sets.

    Returns DESi's metrics: how many regions are blind spots, the fraction of the
    structural space that is unexplored, and how redundant the worked regions are.
    None when DESi is off/absent (caller falls back to in-house set arithmetic).
    """
    if not enabled():
        return None
    mod = _desi()
    if mod is None:
        return None
    try:
        worked = mod.AnchorCoverage("worked", frozenset(covered_ids))
        whole = mod.AnchorCoverage("plausible_universe", frozenset(universe_ids))
        pc = mod.pair_coverage(worked, whole)
        return {
            "engine": "DESi",
            "blindspot_count": pc.symmetric_diff_size,
            "new_region_fraction": pc.new_region_fraction,
            "redundancy": pc.redundancy,
            "universe_size": len(universe_ids),
            "covered_size": len(covered_ids),
        }
    except Exception:  # noqa: BLE001
        return None
