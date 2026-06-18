from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from research import voltarget_daily_monitor
from research.market_data_refresh import MarketDataRefreshResult
from research.voltarget_daily_monitor import run_daily_voltarget_monitor


def _write_daily_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_dir = tmp_path / "stage3"
    candidate_dir = input_dir / "candidate"
    candidate_dir.mkdir(parents=True)
    candidate = {
        "family": "VolTargetTQQQStrategy",
        "category": "tqqq_voltarget",
        "is_primary_candidate": True,
        "classification": "crash_control_candidate",
        "baseline_output_dir": "candidate",
        "run_deep_tournament_recommendation": "defer_deep_tournament",
        "one_year_dominance_share": 0.54,
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
    stitched = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-13", periods=20),
            "ret": [0.0, 0.02, -0.01, 0.01, 0.005] * 4,
            "daily_ret_tqqq": [0.0, 0.01, -0.02, 0.015, 0.005] * 4,
            "equity": [
                1.0,
                1.02,
                1.0098,
                1.019898,
                1.02499749,
                1.02499749,
                1.04549744,
                1.03504247,
                1.04539289,
                1.05061985,
                1.05061985,
                1.07163225,
                1.06091593,
                1.07152509,
                1.07688272,
                1.07688272,
                1.09842037,
                1.08743617,
                1.09831053,
                1.10380208,
            ],
            "drawdown": [0.0, 0.0, -0.01, 0.0, 0.0] * 4,
            "position": [1.0, 1.1, 1.2, 1.1, 1.0] * 4,
            "target_exposure": [1.0, 1.1, 1.2, 1.1, 1.0] * 4,
            "realized_ann_vol": [0.5, 0.55, 0.6, 0.58, 0.57] * 4,
            "below_trend": [0, 0, 1, 0, 0] * 4,
            "strong_momentum": [0, 1, 0, 0, 0] * 4,
            "crash_regime": [0, 0, 0, 0, 0] * 4,
        }
    )
    stitched.to_csv(candidate_dir / "stitched_equity.csv", index=False)
    pd.DataFrame([{"benchmark_symbol": "TQQQ"}]).to_csv(
        candidate_dir / "same_period_benchmark_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [{"year": 2020, "strategy_return": 0.10, "benchmark_return": 0.08}]
    ).to_csv(candidate_dir / "yearly_returns.csv", index=False)

    dates = pd.bdate_range("2020-01-01", periods=33)
    rows = []
    for idx, date in enumerate(dates):
        tqqq_close = 100.0 + idx * 1.4 + (idx % 4) * 0.5
        qqq_close = 100.0 + idx * 0.55 + (idx % 3) * 0.2
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
    data_csv = tmp_path / "prices.csv"
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
                "transaction_cost_bps": 10.0,
                "slippage_bps": 0.0,
                "financing_annual_cost": 0.06,
                "stale_data_warning_days": 5,
                "max_allowed_stale_days": 5,
                "auto_refresh_market_data": True,
                "refresh_lookback_days": 10,
                "fail_on_stale_data": False,
                "stale_data_blocks_new_signal": True,
                "target_symbol": "TQQQ",
                "benchmark_symbol": "TQQQ",
                "risk_policy_config": str(policy_path),
                "risk_dashboard_output_dir": str(tmp_path / "out"),
                "visualization_output_dir": str(tmp_path / "viz"),
                "classification": "crash_control_candidate",
                "production_ready": False,
                "paper_trading_only": True,
            }
        ),
        encoding="utf-8",
    )
    return config_path, data_csv, input_dir


def test_stale_data_produces_warning(tmp_path: Path) -> None:
    config_path, data_csv, _ = _write_daily_fixture(tmp_path)

    result = run_daily_voltarget_monitor(
        config_path=config_path,
        output_dir=tmp_path / "out",
        data_csv=data_csv,
        as_of_date=pd.Timestamp("2020-03-01"),
        generated_at="2026-01-01T00:00:00+00:00",
    )

    status = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert status["status"] == "warning"
    assert status["steps"]["data_freshness_check"]["status"] == "warning"
    log = pd.read_csv(result.log_path)
    assert log["data_freshness_status"].iloc[-1] == "warning"


def test_daily_log_appends_without_duplicate_date_rows(tmp_path: Path) -> None:
    config_path, data_csv, _ = _write_daily_fixture(tmp_path)
    output_dir = tmp_path / "out"

    for _ in range(2):
        run_daily_voltarget_monitor(
            config_path=config_path,
            output_dir=output_dir,
            data_csv=data_csv,
            as_of_date=pd.Timestamp("2020-02-14"),
            generated_at="2026-01-01T00:00:00+00:00",
        )

    log = pd.read_csv(output_dir / "daily_run_log.csv")
    assert log["run_date"].is_unique
    assert len(log[log["run_date"] == "2020-02-14"]) == 1


def test_failed_signal_generation_writes_failed_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, data_csv, _ = _write_daily_fixture(tmp_path)

    def fail_signal(**_: object) -> object:
        raise ValueError("signal generation failed")

    monkeypatch.setattr(voltarget_daily_monitor, "generate_voltarget_signal", fail_signal)
    result = run_daily_voltarget_monitor(
        config_path=config_path,
        output_dir=tmp_path / "out",
        data_csv=data_csv,
        as_of_date=pd.Timestamp("2020-02-14"),
    )

    status = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "signal generation failed" in status["error"]
    log = pd.read_csv(result.log_path)
    assert log["status"].iloc[-1] == "failed"


def test_no_broker_or_order_code_exists() -> None:
    source = Path(voltarget_daily_monitor.__file__).read_text(encoding="utf-8")
    forbidden = {"broker_api", "submit_order", "place_order", "auto_trade", "rebalance_account"}
    assert forbidden.isdisjoint(source)


def _refresh_result(*, status: str, success: bool, stale_days: int) -> MarketDataRefreshResult:
    return MarketDataRefreshResult(
        attempted=True,
        refresh_success=success,
        source="refresh" if success else "failed_refresh",
        status=status,
        cache_path=Path("price_cache/tqqq_qqq_ohlc.csv"),
        latest_price_date="2020-02-14",
        stale_days=stale_days,
        max_allowed_stale_days=1,
        refreshed_rows=1 if success else 0,
        error="" if success else "network unavailable",
        generated_at="2026-01-01T00:00:00+00:00",
    )


def test_stale_cache_triggers_refresh_attempt_and_blocks_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, data_csv, _ = _write_daily_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["max_allowed_stale_days"] = 1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = {"refresh": 0, "signal": 0}

    def fake_refresh(*_: object, **__: object) -> MarketDataRefreshResult:
        calls["refresh"] += 1
        return _refresh_result(status="stale_data", success=False, stale_days=16)

    def fake_signal(**_: object) -> object:
        calls["signal"] += 1
        raise AssertionError("stale data should block accepted signal generation")

    monkeypatch.setattr(voltarget_daily_monitor, "refresh_recent_market_data", fake_refresh)
    monkeypatch.setattr(voltarget_daily_monitor, "generate_voltarget_signal", fake_signal)

    result = run_daily_voltarget_monitor(
        config_path=config_path,
        output_dir=tmp_path / "out",
        data_csv=data_csv,
        as_of_date=pd.Timestamp("2020-03-01"),
    )

    status = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert result.status["status"] == "warning"
    assert status["steps"]["market_data_refresh"]["attempted"] is True
    assert status["steps"]["generate_voltarget_signal"]["status"] == "stale_data"
    assert status["steps"]["generate_voltarget_signal"]["accepted_signal"] is False
    assert calls == {"refresh": 1, "signal": 0}


def test_fresh_refresh_clears_stale_warning_and_accepts_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, data_csv, _ = _write_daily_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["max_allowed_stale_days"] = 1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def fake_refresh(*_: object, **__: object) -> MarketDataRefreshResult:
        return _refresh_result(status="ok", success=True, stale_days=0)

    monkeypatch.setattr(voltarget_daily_monitor, "refresh_recent_market_data", fake_refresh)

    result = run_daily_voltarget_monitor(
        config_path=config_path,
        output_dir=tmp_path / "out",
        data_csv=data_csv,
        as_of_date=pd.Timestamp("2020-02-14"),
    )

    status = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert result.status["status"] == "ok"
    assert status["steps"]["market_data_refresh"]["refresh_success"] is True
    assert status["steps"]["generate_voltarget_signal"]["accepted_signal"] is True
    assert "Market Data Refresh" in (tmp_path / "out" / "signal_report.md").read_text(encoding="utf-8")
