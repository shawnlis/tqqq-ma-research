from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.voltarget_risk_dashboard import (
    REQUIRED_POLICY_FIELDS,
    build_policy_breaches,
    build_voltarget_risk_dashboard,
    load_risk_policy,
)


def _policy() -> dict[str, object]:
    return {
        "max_account_allocation": 0.05,
        "max_allowed_exposure": 1.25,
        "max_daily_trade_delta": 0.10,
        "max_drawdown_warning": -0.30,
        "max_drawdown_stop_review": -0.50,
        "stale_data_stop": 3,
        "financing_cost_assumption": 0.09,
        "paper_trade_start_date": "2026-06-16",
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "stage3"
    candidate_dir = input_dir / "candidate"
    candidate_dir.mkdir(parents=True)
    candidate = {
        "family": "VolTargetTQQQStrategy",
        "category": "tqqq_voltarget",
        "is_primary_candidate": True,
        "classification": "crash_control_candidate",
        "baseline_output_dir": "candidate",
    }
    pd.DataFrame([candidate]).to_csv(input_dir / "candidate_decision_table.csv", index=False)
    pd.DataFrame([candidate]).to_csv(input_dir / "voltarget_stage3_summary.csv", index=False)
    rows = pd.DataFrame(
        {
            "Date": pd.date_range("2026-06-08", periods=5, freq="B"),
            "ret": [0.0, 0.02, -0.05, 0.01, 0.01],
            "daily_ret_tqqq": [0.0, 0.01, -0.03, 0.02, 0.01],
            "equity": [1.0, 1.02, 0.969, 0.97869, 0.9884769],
            "drawdown": [0.0, 0.0, -0.05, -0.0405, -0.031],
            "position": [1.0, 1.1, 1.2, 1.3, 1.35],
            "target_exposure": [1.0, 1.1, 1.2, 1.3, 1.55],
            "realized_ann_vol": [0.5, 0.5, 0.6, 0.7, 0.8],
            "below_trend": [0, 0, 1, 0, 0],
            "strong_momentum": [0, 1, 0, 0, 0],
            "crash_regime": [0, 0, 0, 0, 0],
        }
    )
    rows.to_csv(candidate_dir / "stitched_equity.csv", index=False)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(_policy()), encoding="utf-8")
    return input_dir, policy_path


def test_load_risk_policy_requires_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    data = _policy()
    data.pop("stale_data_stop")
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="stale_data_stop"):
        load_risk_policy(path)

    complete = tmp_path / "complete.yaml"
    complete.write_text(yaml.safe_dump(_policy()), encoding="utf-8")
    policy = load_risk_policy(complete)
    assert set(REQUIRED_POLICY_FIELDS).issubset(policy)


def test_policy_breaches_are_detected_without_account_balance() -> None:
    state = {
        "current_exposure": 1.20,
        "target_exposure_next_session": 1.55,
        "required_trade_delta": 0.35,
        "current_drawdown": -0.40,
        "stale_days": 5,
    }

    breaches = build_policy_breaches(_policy(), state)

    status_by_policy = dict(zip(breaches["policy"], breaches["status"]))
    assert status_by_policy["max_account_allocation"] == "not_evaluated_no_account_balance"
    assert status_by_policy["max_allowed_exposure"] == "breach"
    assert status_by_policy["max_daily_trade_delta"] == "breach"
    assert status_by_policy["max_drawdown_warning"] == "warning"
    assert status_by_policy["stale_data_stop"] == "stop_stale_data"


def test_dashboard_outputs_required_files_and_language(tmp_path: Path) -> None:
    input_dir, policy_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "dashboard"

    result = build_voltarget_risk_dashboard(
        config_path=policy_path,
        input_dir=input_dir,
        output_dir=output_dir,
        as_of_date=pd.Timestamp("2026-06-20"),
    )

    assert result.dashboard_path.exists()
    assert result.summary_path.exists()
    assert result.breaches_path.exists()
    assert (output_dir / "current_signal_box.png").exists()
    assert (output_dir / "drawdown_status.png").exists()
    dashboard = result.dashboard_path.read_text(encoding="utf-8")
    assert "paper trading only" in dashboard
    assert "not production-ready" in dashboard
    assert "crash-control candidate, not persistent alpha" in dashboard
    summary = pd.read_csv(result.summary_path)
    assert int(summary["stale_days"].iloc[0]) == 8
