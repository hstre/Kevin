"""Test isolation.

Point the Layer-9 ledger at a throwaway temp file before any test imports the API
(which builds its shared store at import time), and use the in-house coverage rather
than DESi by default.
"""

import os
import tempfile
from pathlib import Path

os.environ["KEVIN_LAYER9"] = str(Path(tempfile.mkdtemp(prefix="kevin-test-")) / "layer9.jsonl")
os.environ.pop("KEVIN_USE_DESI", None)
