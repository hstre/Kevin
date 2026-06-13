"""Test isolation.

Point the Layer-9 ledger at a throwaway temp file *before* any test module imports
``kevin.api`` (which builds its shared store at import time). This keeps tests from
reading or writing the real ``~/.kevin/layer9.jsonl``.
"""

import os
import tempfile
from pathlib import Path

_LEDGER = Path(tempfile.mkdtemp(prefix="kevin-test-")) / "layer9.jsonl"
os.environ["KEVIN_LAYER9"] = str(_LEDGER)
