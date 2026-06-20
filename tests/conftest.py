from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_baseline_gate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv(
        "RESEARCH_BASELINE_GATE_FILE",
        str(tmp_path / "baseline_gate" / "BASELINE_GATE_FAILED.txt"),
    )
