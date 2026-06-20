from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.voltarget_live_monitor import (
    _financing_sensitivity,
    build_financing_cost_history,
    generate_voltarget_signal,
    load_monitor_config,
)


def _write_fixture(tmp_path: Path, *, future_multiplier: float = 1.0) -> tuple[Path, Path, Path]:
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
        multiplier = future_multiplier if date > pd.Timestamp("2020-01-31") else 1.0
        tqqq_close = (100.0 + idx * 1.4 + (idx % 4) * 0.5) * multiplier
        qqq_close = (100.0 + idx * 0.55 + (idx % 3) * 0.2) * multiplier
        rows.append(
            {
                "Date": date,
                "TQQQ": tqqq_close,
                "TQQQ_OPEN": tqqq_close * 0.995,
                "TQQQ_HIGH": tqqq_close * 1.01,
                "TQQQ_LOW": tqqq_close * 0.99,
                "TQQQ_CLOSE": tqqq_close,
                "QQQ": qqq_close,
                "QQQ_OPEN": qqq_close * 0.995,
                "QQQ_HIGH": qqq_close * 1.005,
                "QQQ_LOW": qqq_close * 0.99,
                "QQQ_CLOSE": qqq_close,
            }
        )
    data_csv = tmp_path / f"prices_{future_multiplier}.csv"
    pd.DataFrame(rows).to_csv(data_csv, index=False)
    policy_path = tmp_path / "risk_policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "max_account_allocation": 0.05,
                "max_allowed_exposure": 2.0,
                "max_daily_trade_delta": 1.0,
                "max_drawdown_warning": -0.50,
                "max_drawdown_stop_review": -0.65,
                "stale_data_stop": 5,
                "financing_cost_assumption": 0.06,
                "paper_trade_start_date": "2020-01-13",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "monitor.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input_dir": str(input_dir),
                "cache_dir": "./price_cache",
                "execution_assumption": "next_open_to_next_open",
                "auto_refresh_market_data": True,
                "refresh_lookback_days": 10,
                "fail_on_stale_data": False,
                "stale_data_blocks_new_signal": True,
                "transaction_cost_bps": 10.0,
                "slippage_bps": 0.0,
                "paper_ledger_mode": "auto",
                "paper_starting_equity": 100000,
                "paper_execution_model": "next_open_to_next_open",
                "paper_fill_price_source": "next_open",
                "fallback_fill_price_source": "latest_close",
                "assumed_slippage_bps": 5.0,
                "assumed_transaction_cost_bps": 10.0,
                "financing_annual_cost": 0.06,
                "annual_financing_rate_assumption": 0.06,
                "financing_applies_above_exposure": 1.0,
                "financing_day_count_basis": 252,
                "stale_data_warning_days": 5,
                "max_allowed_stale_days": 1,
                "target_symbol": "TQQQ",
                "benchmark_symbol": "TQQQ",
                "risk_policy_config": str(policy_path),
                "classification": "crash_control_candidate",
                "production_ready": False,
                "paper_trading_only": True,
                "auto_paper_ledger_enabled": True,
                "manual_ledger_required": False,
                "paper_ledger_start_mode": "live_from_config_date",
                "paper_ledger_start_date": None,
                "paper_ledger_allow_historical_backfill": False,
                "paper_ledger_reset_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    return config_path, data_csv, input_dir


def test_signal_uses_only_data_available_through_latest_close(tmp_path: Path) -> None:
    config_path, data_csv, _ = _write_fixture(tmp_path, future_multiplier=1.0)
    changed_config_path, changed_data_csv, _ = _write_fixture(tmp_path / "changed", future_multiplier=50.0)

    first = generate_voltarget_signal(
        config_path=config_path,
        output_dir=tmp_path / "out1",
        data_csv=data_csv,
        signal_as_of_date=pd.Timestamp("2020-01-31"),
        generated_at="2026-01-01T00:00:00+00:00",
    ).signal_today
    second = generate_voltarget_signal(
        config_path=changed_config_path,
        output_dir=tmp_path / "out2",
        data_csv=changed_data_csv,
        signal_as_of_date=pd.Timestamp("2020-01-31"),
        generated_at="2026-01-01T00:00:00+00:00",
    ).signal_today

    assert first["latest_price_date"] == second["latest_price_date"] == "2020-01-31"
    assert first["target_exposure_next_session"] == pytest.approx(second["target_exposure_next_session"])
    assert first["previous_target_exposure"] == pytest.approx(second["previous_target_exposure"])
    assert first["trade_delta"] == pytest.approx(second["trade_delta"])


def test_stale_data_warning_works(tmp_path: Path) -> None:
    config_path, data_csv, _ = _write_fixture(tmp_path)

    result = generate_voltarget_signal(
        config_path=config_path,
        output_dir=tmp_path / "out",
        data_csv=data_csv,
        signal_as_of_date=pd.Timestamp("2020-03-01"),
    )

    quality = pd.read_csv(result.data_quality_path)
    assert quality["status"].iloc[0] == "stale_data_warning"
    assert result.signal_today["data_quality_status"] == "stale_data_warning"


def test_signal_history_appends_without_duplicate_dates(tmp_path: Path) -> None:
    config_path, data_csv, _ = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"

    generate_voltarget_signal(config_path=config_path, output_dir=output_dir, data_csv=data_csv)
    generate_voltarget_signal(config_path=config_path, output_dir=output_dir, data_csv=data_csv)

    history = pd.read_csv(output_dir / "signal_history.csv")
    assert history["latest_price_date"].is_unique


def test_no_auto_trading_or_broker_integration_exists(tmp_path: Path) -> None:
    config_path, data_csv, _ = _write_fixture(tmp_path)

    result = generate_voltarget_signal(config_path=config_path, output_dir=tmp_path / "out", data_csv=data_csv)
    signal = json.loads(result.signal_today_path.read_text(encoding="utf-8"))

    forbidden = {"order", "orders", "broker", "submit_order", "trade_instruction"}
    assert forbidden.isdisjoint(signal)
    assert signal["no_auto_trading"] is True
    assert signal["production_ready"] is False


def test_paper_equity_uses_execution_assumption_selected_in_config(tmp_path: Path) -> None:
    config_path, data_csv, _ = _write_fixture(tmp_path)

    result = generate_voltarget_signal(config_path=config_path, output_dir=tmp_path / "out", data_csv=data_csv)
    history = pd.read_csv(result.signal_history_path)

    assert set(history["execution_assumption"]) == {"next_open_to_next_open"}
    assert result.signal_today["execution_assumption"] == "next_open_to_next_open"
    assert (tmp_path / "out" / "paper_equity_curve.png").exists()
    assert (tmp_path / "out" / "relative_equity_vs_tqqq.png").exists()
    assert (tmp_path / "out" / "exposure_history.png").exists()


def test_financing_cost_history_uses_excess_exposure_and_day_count() -> None:
    config = {
        "annual_financing_rate_assumption": 0.252,
        "financing_applies_above_exposure": 1.0,
        "financing_day_count_basis": 252,
    }
    frame = pd.DataFrame(
        {
            "position": [0.8, 1.0, 1.5, 2.0],
            "ret": [0.01, 0.01, 0.01 - 0.0005, 0.01 - 0.001],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="B"),
    )

    history = build_financing_cost_history(config, frame)

    assert history["excess_exposure"].tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0])
    assert history["daily_financing_cost"].tolist() == pytest.approx([0.0, 0.0, 0.0005, 0.001])
    assert history["cumulative_financing_cost"].tolist() == pytest.approx([0.0, 0.0, 0.0005, 0.0015])
    assert history["strategy_equity_before_financing"].iloc[-1] > history["strategy_equity_after_financing"].iloc[-1]


def test_financing_sensitivity_uses_same_before_financing_baseline() -> None:
    config = {
        "annual_financing_rate_assumption": 0.06,
        "financing_applies_above_exposure": 1.0,
        "financing_day_count_basis": 252,
    }
    frame = pd.DataFrame(
        {
            "position": [1.5, 2.0],
            "ret": [0.01 - (0.5 * 0.06 / 252), 0.02 - (1.0 * 0.06 / 252)],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="B"),
    )

    sensitivity = _financing_sensitivity(config, frame, rates=(0.03, 0.06, 0.09, 0.12))

    assert sensitivity["final_equity_before_financing"].nunique() == 1
    assert sensitivity["final_equity_after_financing"].is_monotonic_decreasing


def test_generate_signal_writes_financing_cost_outputs(tmp_path: Path) -> None:
    config_path, data_csv, _ = _write_fixture(tmp_path)

    result = generate_voltarget_signal(config_path=config_path, output_dir=tmp_path / "out", data_csv=data_csv)

    assert result.financing_history_path.exists()
    assert result.financing_report_path.exists()
    history = pd.read_csv(result.financing_history_path)
    for column in [
        "exposure",
        "excess_exposure",
        "daily_financing_cost",
        "cumulative_financing_cost",
        "strategy_equity_before_financing",
        "strategy_equity_after_financing",
    ]:
        assert column in history.columns
    report = result.financing_report_path.read_text(encoding="utf-8")
    assert "Days exposure > 1.0" in report
    assert "Average excess exposure" in report
    assert "Cumulative financing drag" in report
    assert "0.03" in report
    assert "0.06" in report
    assert "0.09" in report
    assert "0.12" in report


def test_live_monitor_config_loads_new_financing_fields(tmp_path: Path) -> None:
    config_path, _, _ = _write_fixture(tmp_path)

    config = load_monitor_config(config_path)

    assert config["annual_financing_rate_assumption"] == pytest.approx(0.06)
    assert config["auto_refresh_market_data"] is True
    assert config["refresh_lookback_days"] == 10
    assert config["fail_on_stale_data"] is False
    assert config["stale_data_blocks_new_signal"] is True
    assert config["financing_applies_above_exposure"] == pytest.approx(1.0)
    assert config["financing_day_count_basis"] == 252
    assert config["paper_ledger_mode"] == "auto"
    assert config["paper_starting_equity"] == pytest.approx(100000)
    assert config["paper_execution_model"] == "next_open_to_next_open"
    assert config["paper_fill_price_source"] == "next_open"
    assert config["fallback_fill_price_source"] == "latest_close"
    assert config["assumed_slippage_bps"] == pytest.approx(5.0)
    assert config["assumed_transaction_cost_bps"] == pytest.approx(10.0)
    assert config["max_allowed_stale_days"] == 1
    assert config["auto_paper_ledger_enabled"] is True
    assert config["manual_ledger_required"] is False
    assert config["paper_ledger_start_mode"] == "live_from_config_date"
    assert config["paper_ledger_start_date"] is None
    assert config["paper_ledger_allow_historical_backfill"] is False
    assert config["paper_ledger_reset_allowed"] is False
