"""Pytest bootstrap: put src/ on the import path (src layout, no install needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
