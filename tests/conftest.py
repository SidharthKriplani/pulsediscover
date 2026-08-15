"""Make `src/` importable as top-level packages/modules (matches PYTHONPATH=src used by
Docker/uvicorn) without requiring PYTHONPATH to be set for local `pytest` runs."""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
