from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.voltarget_simplification_battle import (
    _make_model_frames,
    _trend_momentum_answer,
    run_simplification_battle,
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

    dates = pd.bdate_range("2021-01-01", "2023-03-31")
    trend = np.linspace(100.0, 180.0, len(dates))
    seasonal = np.sin(np.linspace(0.0, 20.0, len(dates))) * 5.0
    tqqq = trend + seasonal
    qqq = np.linspace(100.0, 135.0, len(dates)) + seasonal * 0.25
    price_rows = []
    for date, tqqq_close, qqq_close in zip(dates, tqqq, qqq):
        price_rows.append(
            {
                "Date": date,
                "TQQQ": tqqq_close,
                "TQQQ_OPEN": tqqq_close * 0.995,
                "TQQQ_HIGH": tqqq_close * 1.01,
                "TQQQ_LOW": tqqq_close * 0.99,
                "TQQQ_CLOSE": tqqq_close,
                "QQQ": qqq_close,
                "QQQ_OPEN": qqq_close * 0.995,
                "QQQ_HIGH": qqq_close * 1.01,
                "QQQ_LOW": qqq_close * 0.99,
                "QQQ_CLOSE": qqq_close,
            }
        )
    data_csv = tmp_path / "prices.csv"
    pd.DataFrame(price_rows).to_csv(data_csv, index=False)

    pd.DataFrame(
        [
            {
                "train_start": "2021-01-01",
                "train_end": "2021-03-31",
                "test_start": "2021-04-01",
                "test_end": "2023-03-31",
                "target_ann_vol": 0.9,
                "realized_vol_window": 5,
                "min_exposure": 0.5,
                "max_exposure": 2.0,
                "trend_window": 5,
                "momentum_window": 3,
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
            '{"symbols": ["TQQQ", "QQQ"], "start_date": "2021-01-01", '
            '"end_date": "2023-03-31", "transaction_cost_bps": 10.0, '
            '"execution_model": "close_to_close_shifted", "benchmark_symbol": "TQQQ"}'
        ),
        encoding="utf-8",
    )

    price_data = pd.DataFrame(price_rows).set_index("Date")
    test_data = price_data.loc[pd.Timestamp("2021-04-01") :]
    ret = pd.Series(test_data["TQQQ"]).pct_change().fillna(0.0) * 1.2
    stitched = pd.DataFrame(
        {
            "Date": test_data.index,
            "ret": ret.to_numpy(),
            "daily_ret_tqqq": pd.Series(test_data["TQQQ"]).pct_change().fillna(0.0).to_numpy(),
            "position": 1.2,
            "trade_weight": 1.2,
            "tqqq_weight": 1.2,
            "turnover": 0.0,
            "cost": 0.0,
        }
    )
    stitched["equity"] = (1.0 + stitched["ret"]).cumprod()
    stitched["drawdown"] = stitched["equity"] / stitched["equity"].cummax() - 1.0
    stitched.to_csv(candidate_dir / "stitched_equity.csv", index=False)
    return input_dir, data_csv


def test_model_frames_use_identical_dates(tmp_path: Path) -> None:
    input_dir, data_csv = _write_fixture(tmp_path)

    frames, _ = _make_model_frames(input_dir=input_dir, data_csv=data_csv, cache_dir="./price_cache")

    indexes = [tuple(frame.index) for frame in frames.values()]
    assert len(frames) == 5
    assert all(index == indexes[0] for index in indexes)


def test_run_simplification_battle_writes_expected_outputs(tmp_path: Path) -> None:
    input_dir, data_csv = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = run_simplification_battle(input_dir=input_dir, output_dir=output_dir, data_csv=data_csv)

    assert result.summary_path.exists()
    assert result.report_path.exists()
    assert (output_dir / "relative_equity_comparison.png").exists()
    assert (output_dir / "yearly_relative_returns.png").exists()
    assert set(result.summary["model"]) == {
        "locked_voltarget",
        "simple_voltarget_no_trend",
        "constant_same_average_exposure_tqqq",
        "constant_same_max_exposure_tqqq",
        "tqqq_buy_and_hold",
    }
    report = result.report_path.read_text(encoding="utf-8")
    for text in [
        "Which model wins full period?",
        "Which wins ex-2022?",
        "Which wins post-2022?",
        "Which has better drawdown?",
        "Is trend/momentum layer worth keeping?",
    ]:
        assert text in report


def test_trend_momentum_answer_flags_ex_2022_failure() -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "locked_voltarget",
                "full_period_final_multiple": 3.0,
                "ex_2022_final_multiple": 1.2,
                "full_period_max_drawdown": -0.5,
            },
            {
                "model": "simple_voltarget_no_trend",
                "full_period_final_multiple": 2.0,
                "ex_2022_final_multiple": 1.5,
                "full_period_max_drawdown": -0.4,
            },
        ]
    )

    assert "fails ex-2022" in _trend_momentum_answer(summary)
