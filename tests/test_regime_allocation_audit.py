from pathlib import Path

import pandas as pd

from research.cli import main
from research.regime_allocation_audit import classify_regime_allocation_source


def test_same_raw_signal_different_cost_classified_as_turnover_cost_mismatch() -> None:
    likely = classify_regime_allocation_source(
        {
            "max_abs_raw_tqqq_weight_diff": 0.0,
            "max_abs_tqqq_weight_diff": 0.0,
            "max_abs_qqq_weight_diff": 0.0,
            "max_abs_position_diff": 0.0,
            "max_abs_turnover_diff": 0.50,
            "max_abs_cost_diff": 0.0005,
            "max_abs_tqqq_return_diff": 0.0,
            "max_abs_qqq_return_diff": 0.0,
            "max_abs_ret_diff": 0.0005,
            "max_abs_equity_diff": 0.0005,
        }
    )

    assert "turnover/cost mismatch" in likely
    assert "pre-shift signal mismatch" not in likely


def test_same_weights_different_returns_classified_as_data_return_composition_mismatch() -> None:
    likely = classify_regime_allocation_source(
        {
            "max_abs_raw_tqqq_weight_diff": 0.0,
            "max_abs_tqqq_weight_diff": 0.0,
            "max_abs_qqq_weight_diff": 0.0,
            "max_abs_position_diff": 0.0,
            "max_abs_turnover_diff": 0.0,
            "max_abs_cost_diff": 0.0,
            "max_abs_tqqq_return_diff": 0.001,
            "max_abs_qqq_return_diff": 0.0,
            "max_abs_ret_diff": 0.001,
            "max_abs_equity_diff": 0.001,
        }
    )

    assert "data mismatch" in likely
    assert "return composition mismatch" in likely
    assert "shifted weight mismatch" not in likely


def test_different_raw_signal_classified_as_signal_mismatch() -> None:
    likely = classify_regime_allocation_source(
        {
            "max_abs_raw_tqqq_weight_diff": 1.0,
            "max_abs_tqqq_weight_diff": 0.0,
            "max_abs_qqq_weight_diff": 0.0,
            "max_abs_position_diff": 0.0,
            "max_abs_turnover_diff": 0.0,
            "max_abs_cost_diff": 0.0,
            "max_abs_tqqq_return_diff": 0.0,
            "max_abs_qqq_return_diff": 0.0,
            "max_abs_ret_diff": 0.0,
            "max_abs_equity_diff": 0.0,
        }
    )

    assert likely.startswith("pre-shift signal mismatch")


def test_same_returns_different_equity_classified_as_compounding_mismatch() -> None:
    likely = classify_regime_allocation_source(
        {
            "max_abs_raw_tqqq_weight_diff": 0.0,
            "max_abs_tqqq_weight_diff": 0.0,
            "max_abs_qqq_weight_diff": 0.0,
            "max_abs_position_diff": 0.0,
            "max_abs_turnover_diff": 0.0,
            "max_abs_cost_diff": 0.0,
            "max_abs_tqqq_return_diff": 0.0,
            "max_abs_qqq_return_diff": 0.0,
            "max_abs_ret_diff": 0.0,
            "max_abs_equity_diff": 0.01,
        }
    )

    assert likely == "equity compounding mismatch"


def _write_prices(path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", "2020-02-14")
    prices = pd.DataFrame(
        {
            "Date": dates,
            "TQQQ": [100.0 + i * 0.50 for i in range(len(dates))],
            "QQQ": [100.0 + i * 0.20 for i in range(len(dates))],
        }
    )
    prices.to_csv(path, index=False)


def _write_windows(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "train_start": "2020-01-01",
                "train_end": "2020-01-31",
                "test_start": "2020-02-03",
                "test_end": "2020-02-14",
                "trend_window": 2,
                "momentum_window": 1,
                "vol_window": 2,
                "trend_on": 0.0,
                "trend_off": 0.0,
                "mom_on": 0.0,
                "mom_off": -0.01,
                "vol_cap": 1.0,
                "risk_on_leverage": 1.0,
                "risk_off_qqq_position": 1.0,
                "transition_position": 0.25,
                "cooldown_days": 0,
            }
        ]
    ).to_csv(path, index=False)


def _write_old_stitched(path: Path) -> None:
    dates = pd.bdate_range("2020-02-03", "2020-02-14")
    pd.DataFrame(
        {
            "Date": dates,
            "ret": [0.123] + [0.0] * (len(dates) - 1),
            "equity": [1.123] * len(dates),
            "position": [9.0] + [1.0] * (len(dates) - 1),
            "tqqq_weight": [9.0] + [1.0] * (len(dates) - 1),
            "qqq_weight": [9.0] + [0.0] * (len(dates) - 1),
            "turnover": [9.0] + [0.0] * (len(dates) - 1),
        }
    ).to_csv(path, index=False)


def test_audit_regime_allocation_output_contract_is_complete(tmp_path: Path) -> None:
    prices_path = tmp_path / "prices.csv"
    windows_path = tmp_path / "windows.csv"
    old_path = tmp_path / "old_stitched.csv"
    output_dir = tmp_path / "allocation_audit"
    _write_prices(prices_path)
    _write_windows(windows_path)
    _write_old_stitched(old_path)

    assert (
        main(
            [
                "audit-regime-allocation",
                "--windows",
                str(windows_path),
                "--old-stitched",
                str(old_path),
                "--output-dir",
                str(output_dir),
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-02-14",
                "--data-csv",
                str(prices_path),
            ]
        )
        == 0
    )

    for filename in [
        "regime_allocation_audit_daily.csv",
        "regime_allocation_audit_summary.csv",
        "regime_allocation_first_divergence.csv",
        "regime_allocation_audit_report.md",
    ]:
        assert (output_dir / filename).exists()

    daily = pd.read_csv(output_dir / "regime_allocation_audit_daily.csv")
    expected_daily_columns = {
        "QQQ_close",
        "TQQQ_close",
        "qqq_ma",
        "trend",
        "momentum",
        "vol",
        "risk_on",
        "super_risk_on",
        "transition",
        "raw_tqqq_weight",
        "shifted_tqqq_weight",
        "shifted_qqq_weight",
        "total_position",
        "turnover",
        "cost",
        "tqqq_return",
        "qqq_return",
        "strategy_return",
        "equity",
    }
    assert expected_daily_columns.issubset(daily.columns)

    summary = pd.read_csv(output_dir / "regime_allocation_audit_summary.csv")
    assert "likely_source" in summary.columns
    assert not bool(summary.loc[0, "passed"])
