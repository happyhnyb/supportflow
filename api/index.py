"""Vercel serverless entry point."""

import sys
from pathlib import Path

# Vercel imports this module from /api; include the repository's src-layout
# package directory before importing the FastAPI application.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.main import app

__all__ = ["app"]
