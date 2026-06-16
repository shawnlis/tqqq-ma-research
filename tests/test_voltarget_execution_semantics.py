from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.execution import CLOSE_TO_CLOSE_SHIFTED, NEXT_OPEN_TO_CLOSE, NEXT_OPEN_TO_NEXT_OPEN
from research.execution_financing import apply_slippage_and_financing
from research.voltarget_execution_semantics import (
    classify_execution_semantics,
    compute_return_decomposition,
    is_valid_default_execution_model_for_overnight_strategy,
    run_execution_semantics_audit,
)


def _ohlc_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TQQQ": [102.0, 108.0, 117.6],
            "TQQQ_OPEN": [100.0, 105.0, 112.0],
            "TQQQ_HIGH": [103.0, 109.0, 118.0],
            "TQQQ_LOW": [99.0, 104.0, 111.0],
            "TQQQ_CLOSE": [102.0, 108.0, 117.6],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )


def _write_stage3_fixture(tmp_path: Path) -> tuple[Path, Path]:
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
    pd.DataFrame(
        [
            {
                "train_start": "2020-01-01",
                "train_end": "2020-01-10",
                "test_start": "2020-01-13",
                "test_end": "2020-02-14",
                "target_ann_vol": 0.9,
                "realized_vol_window": 3,
                "min_exposure": 0.5,
                "max_exposure": 2.0,
                "trend_window": 3,
                "momentum_window": 2,
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
        (
            '{"symbols": ["TQQQ", "QQQ"], "start_date": "2020-01-01", '
            '"end_date": "2020-02-14", "transaction_cost_bps": 10.0, '
            '"execution_model": "close_to_close_shifted"}'
        ),
        encoding="utf-8",
    )
    dates = pd.bdate_range("2020-01-01", periods=33)
    rows = []
    for idx, date in enumerate(dates):
        tqqq_close = 100.0 + idx * 1.5 + (idx % 5) * 0.4
        qqq_close = 100.0 + idx * 0.6
        rows.append(
            {
                "Date": date,
                "TQQQ": tqqq_close,
                "TQQQ_OPEN": tqqq_close * (0.99 + (idx % 3) * 0.002),
                "TQQQ_HIGH": tqqq_close * 1.01,
                "TQQQ_LOW": tqqq_close * 0.98,
                "TQQQ_CLOSE": tqqq_close,
                "QQQ": qqq_close,
                "QQQ_OPEN": qqq_close * 0.995,
                "QQQ_HIGH": qqq_close * 1.005,
                "QQQ_LOW": qqq_close * 0.99,
                "QQQ_CLOSE": qqq_close,
            }
        )
    data_csv = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(data_csv, index=False)
    return input_dir, data_csv


def test_open_to_open_return_decomposition_is_correct() -> None:
    data = _ohlc_fixture()

    decomposition = compute_return_decomposition(data, "TQQQ")

    assert decomposition["open_to_open_return"].iloc[0] == pytest.approx(105.0 / 100.0 - 1.0)
    assert decomposition["open_to_open_return"].iloc[1] == pytest.approx(112.0 / 105.0 - 1.0)


def test_overnight_and_intraday_compound_to_close_to_close() -> None:
    data = _ohlc_fixture()

    decomposition = compute_return_decomposition(data, "TQQQ")
    row = decomposition.iloc[1]

    compounded = (1.0 + row["overnight_return"]) * (1.0 + row["intraday_return"]) - 1.0
    assert compounded == pytest.approx(row["close_to_close_return"])


def test_next_open_to_close_is_not_default_for_overnight_holding_strategy() -> None:
    assert is_valid_default_execution_model_for_overnight_strategy(NEXT_OPEN_TO_CLOSE) is False
    assert is_valid_default_execution_model_for_overnight_strategy(NEXT_OPEN_TO_NEXT_OPEN) is True
    assert is_valid_default_execution_model_for_overnight_strategy(CLOSE_TO_CLOSE_SHIFTED) is True


def test_financing_cost_applies_only_to_exposure_above_one() -> None:
    frame = pd.DataFrame(
        {
            "ret": [0.01, 0.01, 0.01],
            "turnover": [0.0, 0.0, 0.0],
            "position": [0.75, 1.0, 1.5],
            "trade_weight": [0.75, 1.0, 1.5],
            "daily_ret_tqqq": [0.01, 0.01, 0.01],
            "cost": [0.0, 0.0, 0.0],
        },
        index=pd.date_range("2020-01-01", periods=3),
    )

    adjusted = apply_slippage_and_financing(frame, slippage_bps=0.0, financing_annual_cost=0.252, periods_per_year=252)

    assert adjusted["financing_cost"].tolist() == pytest.approx([0.0, 0.0, 0.0005])


def test_report_classification_rules_on_fixtures() -> None:
    ok = pd.DataFrame(
        [
            {"scenario_type": "base_execution", "execution_model": CLOSE_TO_CLOSE_SHIFTED, "full_period_ratio_vs_tqqq": 2.0, "ex_2022_ratio_vs_tqqq": 2.0, "max_drawdown_improvement": 0.2, "transaction_cost_bps": 10.0, "slippage_bps": 0.0, "financing_annual_cost": 0.0},
            {"scenario_type": "base_execution", "execution_model": NEXT_OPEN_TO_NEXT_OPEN, "full_period_ratio_vs_tqqq": 1.3, "ex_2022_ratio_vs_tqqq": 1.05, "max_drawdown_improvement": 0.2, "transaction_cost_bps": 10.0, "slippage_bps": 0.0, "financing_annual_cost": 0.0},
            {"scenario_type": "financing_sensitivity", "execution_model": NEXT_OPEN_TO_NEXT_OPEN, "full_period_ratio_vs_tqqq": 1.1, "ex_2022_ratio_vs_tqqq": 1.0, "max_drawdown_improvement": 0.2, "transaction_cost_bps": 10.0, "slippage_bps": 0.0, "financing_annual_cost": 0.06},
        ]
    )
    invalid = ok.copy()
    invalid.loc[invalid["execution_model"].eq(NEXT_OPEN_TO_NEXT_OPEN), "full_period_ratio_vs_tqqq"] = 0.5

    assert classify_execution_semantics(ok) == "execution_ok_for_paper_monitor"
    assert classify_execution_semantics(invalid) == "execution_invalidates_strategy"


def test_run_execution_semantics_audit_writes_outputs(tmp_path: Path) -> None:
    input_dir, data_csv = _write_stage3_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = run_execution_semantics_audit(input_dir=input_dir, output_dir=output_dir, data_csv=data_csv)

    assert result.summary_path.exists()
    assert result.decomposition_path.exists()
    assert result.report_path.exists()
    assert (output_dir / "execution_model_equity_curves.png").exists()
    assert (output_dir / "overnight_vs_intraday_contribution.png").exists()
    report = result.report_path.read_text(encoding="utf-8")
    assert "Is next_open_to_close a valid model for this strategy?" in report
    assert "next_close_to_next_close is not implemented" in report
