"""Paths for running the insurance application from this repository."""

from pathlib import Path

INSURANCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INSURANCE_DIR.parent.parent
PROMPT_PROFILE_DIR = INSURANCE_DIR / "prompt_profiles"
