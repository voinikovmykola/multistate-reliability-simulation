"""The outputs in outputs/ must reproduce the numbers and figures of the article."""

from __future__ import annotations

import subprocess
import sys

import pytest

from multistate_reliability.config import OUTPUT_DIR, REPO_ROOT


@pytest.mark.skipif(not (OUTPUT_DIR / "tables" / "sensitivity_ranges.csv").exists(),
                    reason="run the experiment scripts first")
def test_outputs_match_the_article():
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_article_values.py")],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
