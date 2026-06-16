from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.execution import CLOSE_TO_CLOSE_SHIFTED, NEXT_OPEN_TO_CLOSE
from research.execution_financing import (
    apply_slippage_and_financing,
    identify_breaking_assumption,
    run_execution_financing_audit,
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "stage3"
    candidate_dir = input_dir / "candidate"
    candidate_dir.mkdir(parents=True)

    candidate_row = {
        "family": "VolTargetTQQQStrategy",
        "category": "tqqq_voltarget",
        "is_primary_candidate": True,
        "classification": "crash_control_candidate",
        "baseline_output_dir": "candidate",
    }
    pd.DataFrame([candidate_row]).to_csv(input_dir / "candidate_decision_table.csv", index=False)
    pd.DataFrame([candidate_row]).to_csv(input_dir / "voltarget_stage3_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "train_start": "2020-01-01",
                "train_end": "2020-01-06",
                "test_start": "2020-01-07",
                "test_end": "2020-01-14",
                "target_ann_vol": 0.9,
                "realized_vol_window": 2,
                "min_exposure": 0.5,
                "max_exposure": 2.0,
                "trend_window": 2,
                "momentum_window": 1,
                "trend_multiplier_below_ma": 0.5,
                "momentum_boost": 1.15,
                "crash_vol_cutoff": 1.0,
                "crash_exposure": 0.25,
                "use_rebound_module": 0,
                "use_drawdown_governor": 0,
                "risk_off_symbol": "CASH",
                "risk_off_weight": 0.0,
                "trade_asset": "TQQQ",
                "primary_signal_asset": "QQQ",
                "secondary_filter_asset": "QQQ",
                "benchmark_symbol": "TQQQ",
                "asset_config_risk_off_symbol": "CASH",
                "use_market_internals": 0,
            }
        ]
    ).to_csv(candidate_dir / "walk_forward_windows.csv", index=False)
    (candidate_dir / "run_config.json").write_text(
        '{"symbols": ["TQQQ", "QQQ"], "start_date": "2020-01-01", "end_date": "2020-01-14"}',
        encoding="utf-8",
    )

    dates = pd.bdate_range("2020-01-01", periods=10)
    tqqq = [100, 102, 101, 104, 108, 106, 110, 115, 112, 118]
    qqq = [100, 101, 102, 103, 104, 103, 105, 107, 106, 108]
    rows = []
    for idx, date in enumerate(dates):
        rows.append(
            {
                "Date": date,
                "TQQQ": tqqq[idx],
                "TQQQ_OPEN": tqqq[idx] * 0.99,
                "TQQQ_HIGH": tqqq[idx] * 1.01,
                "TQQQ_LOW": tqqq[idx] * 0.98,
                "TQQQ_CLOSE": tqqq[idx],
                "QQQ": qqq[idx],
                "QQQ_OPEN": qqq[idx] * 0.99,
                "QQQ_HIGH": qqq[idx] * 1.01,
                "QQQ_LOW": qqq[idx] * 0.98,
                "QQQ_CLOSE": qqq[idx],
            }
        )
    data_csv = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(data_csv, index=False)
    return input_dir, data_csv


def test_financing_cost_applies_only_above_one_times_exposure() -> None:
    index = pd.date_range("2020-01-01", periods=4)
    stitched = pd.DataFrame(
        {
            "ret": [0.01, 0.01, 0.01, 0.01],
            "turnover": [0.0, 0.5, 1.0, 0.0],
            "trade_weight": [0.8, 1.0, 1.5, 2.0],
            "position": [0.8, 1.0, 1.5, 2.0],
            "daily_ret_tqqq": [0.01, 0.01, 0.01, 0.01],
            "cost": [0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )

    adjusted = apply_slippage_and_financing(
        stitched,
        slippage_bps=10.0,
        financing_annual_cost=0.252,
        periods_per_year=252,
    )

    assert adjusted["financing_cost"].tolist() == pytest.approx([0.0, 0.0, 0.0005, 0.001])
    assert adjusted["slippage_cost"].tolist() == pytest.approx([0.0, 0.0005, 0.001, 0.0])
    assert adjusted.loc[index[2], "ret"] == pytest.approx(0.01 - 0.0005 - 0.001)


def test_identifies_first_breaking_assumption_by_severity() -> None:
    summary = pd.DataFrame(
        [
            {"execution_model": CLOSE_TO_CLOSE_SHIFTED, "transaction_cost_bps": 10.0, "slippage_bps": 0.0, "financing_annual_cost": 0.0, "final_equity_ratio_vs_tqqq": 1.2, "beats_tqqq": True},
            {"execution_model": NEXT_OPEN_TO_CLOSE, "transaction_cost_bps": 10.0, "slippage_bps": 0.0, "financing_annual_cost": 0.03, "final_equity_ratio_vs_tqqq": 0.99, "beats_tqqq": False},
            {"execution_model": CLOSE_TO_CLOSE_SHIFTED, "transaction_cost_bps": 25.0, "slippage_bps": 5.0, "financing_annual_cost": 0.03, "final_equity_ratio_vs_tqqq": 0.8, "beats_tqqq": False},
        ]
    )

    message = identify_breaking_assumption(summary)

    assert "financing_annual_cost=0.03" in message
    assert "transaction_cost_bps=10.0" in message


def test_run_execution_financing_audit_writes_expected_outputs(tmp_path: Path) -> None:
    input_dir, data_csv = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = run_execution_financing_audit(
        input_dir=input_dir,
        output_dir=output_dir,
        data_csv=data_csv,
        execution_models=(CLOSE_TO_CLOSE_SHIFTED, NEXT_OPEN_TO_CLOSE),
        transaction_cost_bps_scenarios=(10.0, 25.0),
        slippage_bps_scenarios=(0.0, 5.0),
        financing_annual_cost_scenarios=(0.0, 0.03),
    )

    assert len(result.summary) == 16
    assert (output_dir / "execution_financing_summary.csv").exists()
    assert (output_dir / "execution_financing_report.md").exists()
    assert (output_dir / "sensitivity_heatmap.png").exists()
    report = (output_dir / "execution_financing_report.md").read_text(encoding="utf-8")
    assert "Does VolTarget still beat TQQQ after realistic financing?" in report
    assert "Extra financing cost is applied only to exposure above 1.0x" in report
    assert set(result.summary["transaction_cost_bps"]) == {10.0, 25.0}
    assert set(result.summary["slippage_bps"]) == {0.0, 5.0}
    assert set(result.summary["financing_annual_cost"]) == {0.0, 0.03}