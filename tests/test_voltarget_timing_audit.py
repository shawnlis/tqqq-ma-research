from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.backtest import backtest_vol_target_with_positions
from research.cli import main
from research.validation import _vol_target_row_to_params
from research.voltarget_timing_audit import audit_voltarget_timing_frames


def _price_data() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=90)
    steps = np.arange(len(dates), dtype=float)
    qqq_returns = 0.001 + 0.012 * np.sin(steps / 3.0) + 0.006 * np.cos(steps / 7.0)
    tqqq_returns = 3.0 * qqq_returns - 0.00005
    return pd.DataFrame(
        {
            "TQQQ": 100.0 * np.cumprod(1.0 + tqqq_returns),
            "QQQ": 100.0 * np.cumprod(1.0 + qqq_returns),
        },
        index=dates,
    )


def _window_row(data: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "train_start": data.index[0].date().isoformat(),
            "train_end": data.index[39].date().isoformat(),
            "test_start": data.index[40].date().isoformat(),
            "test_end": data.index[79].date().isoformat(),
            "target_ann_vol": 0.60,
            "realized_vol_window": 5,
            "min_exposure": 0.10,
            "max_exposure": 2.0,
            "trend_window": 10,
            "momentum_window": 5,
            "trend_multiplier_below_ma": 0.50,
            "momentum_boost": 1.15,
            "crash_vol_cutoff": 0.20,
            "crash_exposure": 0.25,
            "risk_off_symbol": "CASH",
            "risk_off_weight": 0.0,
            "trade_asset": "TQQQ",
            "primary_signal_asset": "QQQ",
            "secondary_filter_asset": "QQQ",
            "benchmark_symbol": "TQQQ",
            "asset_config_risk_off_symbol": "CASH",
        }
    )


def _fixture_frames() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = _price_data()
    row = _window_row(data)
    params = _vol_target_row_to_params(row)
    bt = backtest_vol_target_with_positions(
        data.loc[: pd.Timestamp(row["test_end"])],
        params=params,
        transaction_cost_bps=10.0,
    )
    stitched = bt.loc[pd.Timestamp(row["test_start"]) : pd.Timestamp(row["test_end"])].copy()
    stitched["equity"] = (1.0 + stitched["ret"]).cumprod()
    windows = pd.DataFrame([row])
    config = {
        "experiment_name": "vol_target_timing_fixture",
        "symbols": ["TQQQ", "QQQ"],
        "start_date": data.index[0].date().isoformat(),
        "end_date": data.index[-1].date().isoformat(),
        "strategy_name": "vol_target",
        "benchmark_symbol": "TQQQ",
        "transaction_cost_bps": 10.0,
        "train_years": 1,
        "test_years": 1,
        "objective": "objective_final_ratio",
        "output_dir": "unused",
    }
    return config, data, stitched, windows


def test_voltarget_timing_audit_passes_for_correctly_shifted_fixture() -> None:
    config, data, stitched, windows = _fixture_frames()

    summary = audit_voltarget_timing_frames(
        config=config,
        stitched=stitched,
        windows=windows,
        price_data=data,
    )

    row = summary.iloc[0]
    assert bool(row["passed"])
    assert bool(row["realized_vol_prior_bar_pass"])
    assert bool(row["position_shift_pass"])


def test_voltarget_timing_audit_fails_when_realized_vol_signal_is_used_same_day() -> None:
    config, data, stitched, windows = _fixture_frames()
    leaked = stitched.copy()
    leaked["trade_weight"] = leaked["target_exposure"]
    leaked["tqqq_weight"] = leaked["target_exposure"]
    leaked["position"] = leaked["target_exposure"]

    summary = audit_voltarget_timing_frames(
        config=config,
        stitched=leaked,
        windows=windows,
        price_data=data,
    )

    row = summary.iloc[0]
    assert not bool(row["passed"])
    assert not bool(row["realized_vol_prior_bar_pass"])
    assert not bool(row["no_current_day_return_pass"])
    assert row["first_current_day_leak_date"]


def test_voltarget_timing_audit_fails_when_position_is_not_shifted() -> None:
    config, data, stitched, windows = _fixture_frames()
    unshifted = stitched.copy()
    unshifted.loc[unshifted.index[5:], "trade_weight"] = 0.0
    unshifted.loc[unshifted.index[5:], "tqqq_weight"] = 0.0
    unshifted.loc[unshifted.index[5:], "position"] = 0.0

    summary = audit_voltarget_timing_frames(
        config=config,
        stitched=unshifted,
        windows=windows,
        price_data=data,
    )

    row = summary.iloc[0]
    assert not bool(row["passed"])
    assert not bool(row["position_shift_pass"])
    assert row["first_position_shift_failure_date"]


def test_audit_voltarget_timing_cli_reports_warmup_date(tmp_path: Path) -> None:
    config, data, stitched, windows = _fixture_frames()
    output_dir = tmp_path / "voltarget_output"
    output_dir.mkdir()
    stitched.to_csv(output_dir / "stitched_equity.csv", index_label="Date")
    windows.to_csv(output_dir / "walk_forward_windows.csv", index=False)

    prices = data.reset_index().rename(columns={"index": "Date"})
    data_csv = tmp_path / "prices.csv"
    prices.to_csv(data_csv, index=False)

    config.update(
        {
            "output_dir": str(output_dir),
            "data_csv": str(data_csv),
        }
    )
    config_path = tmp_path / "vol_target_fixture.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    audit_dir = tmp_path / "audit"

    assert main(["audit-voltarget-timing", str(config_path), "--output-dir", str(audit_dir)]) == 0

    summary_path = audit_dir / "voltarget_timing_audit_summary.csv"
    report_path = audit_dir / "voltarget_timing_audit_report.md"
    assert summary_path.exists()
    assert report_path.exists()
    summary = pd.read_csv(summary_path)
    assert summary.loc[0, "first_tradable_after_warmup"]
    assert "Warmup date reported" in report_path.read_text(encoding="utf-8")
