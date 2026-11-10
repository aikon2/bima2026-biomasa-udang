"""Shared pytest setup: make the repo root importable."""
import sys
from pathlib import Path

REPO_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(REPO_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DIR))
