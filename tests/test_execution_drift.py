from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.cli import main
from research.execution_drift import run_execution_drift_tracking


def _write_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "input_dir": "unused",
                "cache_dir": "./price_cache",
                "execution_assumption": "next_open_to_next_open",
                "transaction_cost_bps": 10.0,
                "slippage_bps": 0.0,
                "financing_annual_cost": 0.06,
                "stale_data_warning_days": 5,
                "target_symbol": "TQQQ",
                "benchmark_symbol": "TQQQ",
                "classification": "crash_control_candidate",
                "production_ready": False,
                "paper_trading_only": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_signals(signal_dir: Path) -> Path:
    signal_dir.mkdir(parents=True)
    path = signal_dir / "signal_history.csv"
    pd.DataFrame(
        [
            {
                "latest_price_date": "2020-01-01",
                "target_symbol": "TQQQ",
                "target_exposure_next_session": 1.5,
                "estimated_transaction_cost": 0.001,
                "estimated_financing_cost": 0.0005,
                "current_paper_equity": 1.0,
            },
            {
                "latest_price_date": "2020-01-02",
                "target_symbol": "TQQQ",
                "target_exposure_next_session": 1.0,
                "estimated_transaction_cost": 0.0,
                "estimated_financing_cost": 0.0,
                "current_paper_equity": 1.02,
            },
        ]
    ).to_csv(path, index=False)
    return path


def _write_prices(path: Path, *, future_open: float = 999.0, missing_next_open: bool = False) -> Path:
    next_open = None if missing_next_open else 101.0
    pd.DataFrame(
        [
            {"Date": "2020-01-01", "TQQQ": 100.0, "TQQQ_OPEN": 99.0, "TQQQ_CLOSE": 100.0},
            {"Date": "2020-01-02", "TQQQ": 102.0, "TQQQ_OPEN": next_open, "TQQQ_CLOSE": 102.0},
            {"Date": "2020-01-03", "TQQQ": 104.0, "TQQQ_OPEN": 103.0, "TQQQ_CLOSE": 104.0},
            {"Date": "2020-01-06", "TQQQ": 106.0, "TQQQ_OPEN": future_open, "TQQQ_CLOSE": 106.0},
        ]
    ).to_csv(path, index=False)
    return path


def test_drift_calculation_fixture(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir)
    prices = _write_prices(tmp_path / "prices.csv")

    result = run_execution_drift_tracking(
        config_path=config,
        signal_dir=signal_dir,
        output_dir=tmp_path / "out",
        data_csv=prices,
        drift_threshold=0.01,
    )

    first = result.summary.iloc[0]
    assert first["previous_close"] == pytest.approx(100.0)
    assert first["next_open"] == pytest.approx(101.0)
    assert first["next_close"] == pytest.approx(102.0)
    assert first["close_to_close_return"] == pytest.approx(0.02)
    assert first["open_to_close_return"] == pytest.approx(102.0 / 101.0 - 1.0)
    assert first["open_to_open_return"] == pytest.approx(103.0 / 101.0 - 1.0)
    assert first["close_to_close_research_model_return"] == pytest.approx(1.5 * 0.02)
    assert first["next_open_to_next_open_model_return"] == pytest.approx(1.5 * (103.0 / 101.0 - 1.0))
    assert first["paper_execution_return"] == pytest.approx(1.5 * (103.0 / 101.0 - 1.0) - 0.0015)
    assert result.summary_path.exists()
    assert result.report_path.exists()
    assert (tmp_path / "out" / "model_vs_execution_equity.png").exists()
    assert (tmp_path / "out" / "daily_drift.png").exists()


def test_missing_open_price_is_handled_gracefully(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir)
    prices = _write_prices(tmp_path / "prices.csv", missing_next_open=True)

    result = run_execution_drift_tracking(
        config_path=config,
        signal_dir=signal_dir,
        output_dir=tmp_path / "out",
        data_csv=prices,
    )

    first = result.summary.iloc[0]
    assert "missing_next_open" in first["warnings"]
    assert pd.isna(first["open_to_open_return"])
    assert pd.isna(first["paper_execution_return"])


def test_execution_drift_does_not_use_prices_after_following_open(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir)
    original_prices = _write_prices(tmp_path / "prices_original.csv", future_open=999.0)
    changed_prices = _write_prices(tmp_path / "prices_changed.csv", future_open=1.0)

    original = run_execution_drift_tracking(
        config_path=config,
        signal_dir=signal_dir,
        output_dir=tmp_path / "out_original",
        data_csv=original_prices,
    ).summary.iloc[0]
    changed = run_execution_drift_tracking(
        config_path=config,
        signal_dir=signal_dir,
        output_dir=tmp_path / "out_changed",
        data_csv=changed_prices,
    ).summary.iloc[0]

    compared = [
        "previous_close",
        "next_open",
        "next_close",
        "following_open",
        "close_to_close_return",
        "open_to_open_return",
        "paper_execution_return",
    ]
    for column in compared:
        assert original[column] == pytest.approx(changed[column])


def test_execution_drift_cli_writes_expected_outputs(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir)
    prices = _write_prices(tmp_path / "prices.csv")
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "track-execution-drift",
                "--config",
                str(config),
                "--signal-dir",
                str(signal_dir),
                "--output-dir",
                str(output_dir),
                "--data-csv",
                str(prices),
            ]
        )
        == 0
    )
    assert (output_dir / "execution_drift_summary.csv").exists()
    assert (output_dir / "execution_drift_report.md").exists()
    assert (output_dir / "model_vs_execution_equity.png").exists()
    assert (output_dir / "daily_drift.png").exists()
