"""Layer 9 - the append-only ledger that lets the method library grow from runs.

This is the persistence behind the loop the design calls for:

    DESi extracts abstract methods from earlier successful thinking processes.

When a human marks a candidate as having *worked*, ``method_library.extract_method``
mines a content-free method from that run; this store appends it to a durable,
append-only ledger (JSONL) and replays it back into the library on the next start.
Over time the library is no longer just the seed set - it carries the abstracted
shape of every process a human found valuable.

Append-only and replay-stable, in the spirit of DESi's "local Layer 9":
  * we never rewrite history - only append;
  * method ids are content hashes, so re-learning the same shape is idempotent.

Storage is a plain JSONL file (one method per line). No database - this is a
demonstrator, and the ledger is meant to be readable and diff-able.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from .models import Affinity, Method


def default_store_path() -> Path:
    """Where the ledger lives. Override with ``KEVIN_LAYER9``."""
    env = os.getenv("KEVIN_LAYER9")
    if env:
        return Path(env)
    return Path.home() / ".kevin" / "layer9.jsonl"


class Layer9Store:
    """An append-only JSONL ledger of methods learned from real runs."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_store_path()

    # -- write -------------------------------------------------------------- #
    def append(self, method: Method, *, run_id: str, candidate_id: str) -> None:
        """Append one learned method. Append-only; never rewrites existing lines."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "id": method.id,
            "name": method.name,
            "origin": method.origin,
            "summary": method.summary,
            "steps": list(method.steps),
            "affinities": [a.value for a in method.affinities],
            "run_id": run_id,
            "candidate_id": candidate_id,
            "learned_at": datetime.now(UTC).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- read --------------------------------------------------------------- #
    def load_methods(self) -> list[Method]:
        """Replay the ledger into Method objects.

        De-duplicates by method id (the last write wins on metadata, but the id is a
        content hash so duplicates are identical anyway). Malformed lines are
        skipped rather than crashing a startup.
        """
        if not self.path.exists():
            return []
        by_id: dict[str, Method] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                method = Method(
                    name=rec["name"],
                    origin=rec.get("origin", "kevin"),
                    summary=rec.get("summary", ""),
                    steps=tuple(rec.get("steps", ())),
                    affinities=tuple(
                        Affinity(a) for a in rec.get("affinities", [])
                        if a in Affinity._value2member_map_
                    ),
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            by_id[method.id] = method
        return list(by_id.values())

    def known_ids(self) -> set[str]:
        return {m.id for m in self.load_methods()}
