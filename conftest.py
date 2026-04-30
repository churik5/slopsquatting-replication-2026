"""Pytest bootstrap.

Falls back to adding `src/` to sys.path in case the editable install's .pth
file got UF_HIDDEN'd by uv on macOS (seen with uv 0.11.x). Idempotent.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
