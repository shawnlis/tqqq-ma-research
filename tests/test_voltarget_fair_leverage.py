from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.fairness import constant_exposure_backtest
from research.voltarget_fair_leverage import (
    SIMPLE_NO_TREND_CRASH_VOL_CUTOFF,
    build_stage2_fair_leverage_summary,
    classify_fair_leverage_candidate,
    realized_average_exposure,
    simple_voltarget_no_trend_params_from_row,
    write_stage2_fair_leverage_report,
)


def _price_fixture() -> pd.DataFrame:
    dates = pd.bdate_range("2021-10-01", periods=90)
    tqqq = [100.0]
    qqq = [100.0]
    for i in range(1, len(dates)):
        ret = 0.002 if i % 7 else -0.006
        qqq.append(qqq[-1] * (1.0 + ret / 3.0))
        tqqq.append(tqqq[-1] * (1.0 + ret))
    data = pd.DataFrame({"Date": dates, "TQQQ": tqqq, "QQQ": qqq})
    return data


def _write_minimal_stage2_tree(tmp_path: Path) -> Path:
    output_dir = tmp_path / "stage2"
    candidate_dir = output_dir / "experiments" / "plain" / "standard_5y_1y__10bps"
    candidate_dir.mkdir(parents=True)

    prices = _price_fixture()
    data_csv = tmp_path / "prices.csv"
    prices.to_csv(data_csv, index=False)
    data = prices.set_index("Date")
    test_dates = data.index[-30:]
    stitched = constant_exposure_backtest(
        data,
        dates=test_dates,
        exposure=0.75,
        transaction_cost_bps=10.0,
    )
    stitched["trade_weight"] = 0.75
    stitched["tqqq_weight"] = 0.75
    stitched.to_csv(candidate_dir / "stitched_equity.csv")

    wf = pd.DataFrame(
        [
            {
                "train_start": data.index[0].date().isoformat(),
                "train_end": data.index[-31].date().isoformat(),
                "test_start": test_dates[0].date().isoformat(),
                "test_end": test_dates[-1].date().isoformat(),
                "target_ann_vol": 0.60,
                "realized_vol_window": 5,
                "min_exposure": 0.25,
                "max_exposure": 1.0,
                "trend_window": 10,
                "momentum_window": 5,
                "trend_multiplier_below_ma": 0.5,
                "momentum_boost": 1.15,
                "crash_vol_cutoff": 0.05,
                "crash_exposure": 0.25,
                "use_rebound_module": 1,
                "use_drawdown_governor": 1,
                "risk_off_symbol": "CASH",
                "risk_off_weight": 0.0,
                "trade_asset": "TQQQ",
                "primary_signal_asset": "QQQ",
                "secondary_filter_asset": "QQQ",
                "benchmark_symbol": "TQQQ",
                "asset_config_risk_off_symbol": "CASH",
            }
        ]
    )
    wf.to_csv(candidate_dir / "walk_forward_windows.csv", index=False)
    (candidate_dir / "run_config.json").write_text(
        """
{
  "symbols": ["TQQQ", "QQQ"],
  "start_date": "2021-10-01",
  "end_date": null,
  "benchmark_symbol": "TQQQ",
  "transaction_cost_bps": 10.0,
  "execution_model": "close_to_close_shifted",
  "data_csv": "%s",
  "download_ohlc": false
}
""".strip()
        % str(data_csv).replace("\\", "\\\\"),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "family": "VolTargetTQQQStrategy",
                "category": "tqqq_voltarget",
                "baseline_output_dir": str(candidate_dir),
                "final_equity_ratio": 1.2,
                "final_equity_ratio_ex_2022": 1.1,
                "min_leave_one_year_out_ratio": 1.05,
                "max_drawdown_improvement_ex_2022": 0.1,
                "one_year_dominated": True,
                "largest_contribution_year": 2022,
                "largest_single_year_contribution_share": 0.7,
                "governor_adds_value_outside_2022": "",
                "status": "ok",
            }
        ]
    ).to_csv(output_dir / "voltarget_stage2_summary.csv", index=False)
    return output_dir


def test_constant_1x_exposure_matches_buy_and_hold_after_initial_cost_convention() -> None:
    prices = pd.DataFrame(
        {"TQQQ": [100.0, 110.0, 99.0]},
        index=pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]),
    )
    bt = constant_exposure_backtest(prices, exposure=1.0, transaction_cost_bps=10.0)

    expected_first_ret = -0.001
    expected_equity = (1.0 + expected_first_ret) * 1.10 * 0.90
    assert float(bt["equity"].iloc[-1]) == pytest.approx(expected_equity)


def test_same_average_exposure_benchmark_uses_realized_candidate_exposure(tmp_path: Path) -> None:
    output_dir = _write_minimal_stage2_tree(tmp_path)

    summary = build_stage2_fair_leverage_summary(output_dir)

    row = summary.iloc[0]
    assert float(row["realized_average_exposure"]) == pytest.approx(0.75)
    assert pd.notna(float(row["ratio_vs_same_avg_exposure_constant_tqqq"]))


def test_realized_average_exposure_prefers_trade_weight() -> None:
    stitched = pd.DataFrame(
        {
            "trade_weight": [0.5, 1.0],
            "tqqq_weight": [2.0, 2.0],
            "position": [3.0, 3.0],
        }
    )

    assert realized_average_exposure(stitched) == pytest.approx(0.75)


def test_simple_voltarget_no_trend_excludes_trend_momentum_and_governor_controls() -> None:
    row = pd.Series(
        {
            "target_ann_vol": 0.75,
            "realized_vol_window": 20,
            "min_exposure": 0.25,
            "max_exposure": 2.0,
            "trend_window": 150,
            "momentum_window": 63,
            "trend_multiplier_below_ma": 0.25,
            "momentum_boost": 1.30,
            "crash_vol_cutoff": 0.04,
            "crash_exposure": 0.0,
            "use_rebound_module": 1,
            "use_drawdown_governor": 1,
            "use_market_internals": 1,
            "trade_asset": "TQQQ",
            "primary_signal_asset": "QQQ",
            "secondary_filter_asset": "QQQ",
            "benchmark_symbol": "TQQQ",
        }
    )

    params = simple_voltarget_no_trend_params_from_row(row)

    assert params.trend_multiplier_below_ma == pytest.approx(1.0)
    assert params.momentum_boost == pytest.approx(1.0)
    assert params.crash_vol_cutoff == pytest.approx(SIMPLE_NO_TREND_CRASH_VOL_CUTOFF)
    assert not params.rebound.use_rebound_module
    assert not params.governor.use_drawdown_governor
    assert not params.market_internals.use_market_internals


def test_fair_leverage_report_includes_ex_2022_attribution(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "family": "VolTargetTQQQStrategy",
                "category": "tqqq_voltarget",
                "classification": "persistent_candidate",
                "ratio_vs_1x_tqqq": 1.5,
                "ratio_vs_same_max_constant_tqqq": 0.9,
                "ratio_vs_same_avg_exposure_constant_tqqq": 1.1,
                "ratio_vs_simple_voltarget_no_trend": 1.2,
                "ex_2022_ratio_vs_same_avg_exposure_constant_tqqq": 1.05,
                "ex_2022_ratio_vs_simple_voltarget_no_trend": 1.04,
                "ratio_2022_only_vs_simple_voltarget_no_trend": 1.3,
                "stage2_final_equity_ratio_ex_2022": 1.08,
                "stage2_one_year_dominated": True,
            }
        ]
    )
    report_path = tmp_path / "fair_leverage_benchmark_report.md"

    write_stage2_fair_leverage_report(report_path, summary)

    report = report_path.read_text(encoding="utf-8")
    assert "Ex-2022 and 2022-only ratios are recomputed" in report
    assert "ex_2022_ratio_vs_simple_voltarget_no_trend" in report
    assert "yes versus TQQQ and fair leverage/no-trend baselines" in report


def test_governor_overfit_to_2022_classification_triggers() -> None:
    row = {
        "category": "tqqq_voltarget_governor",
        "stage2_final_equity_ratio": 2.0,
        "stage2_final_equity_ratio_ex_2022": 0.99,
        "governor_adds_value_outside_2022": False,
        "ratio_vs_1x_tqqq": 2.0,
    }

    assert classify_fair_leverage_candidate(row) == "governor_overfit_to_2022"


def test_persistent_candidate_requires_beating_simple_no_trend() -> None:
    row = {
        "category": "tqqq_voltarget",
        "stage2_final_equity_ratio": 1.4,
        "stage2_final_equity_ratio_ex_2022": 1.1,
        "stage2_min_leave_one_year_out_ratio": 1.05,
        "ratio_vs_1x_tqqq": 1.4,
        "ratio_vs_same_avg_exposure_constant_tqqq": 1.1,
        "ratio_vs_simple_voltarget_no_trend": 0.99,
    }

    assert classify_fair_leverage_candidate(row) != "persistent_candidate"
