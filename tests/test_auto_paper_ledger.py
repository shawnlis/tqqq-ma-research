from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.auto_paper_ledger import initialize_auto_paper_ledger, update_auto_paper_ledger
from research.cli import main


def _write_config(path: Path, **overrides: object) -> Path:
    config = {
        "input_dir": "unused",
        "cache_dir": "./price_cache",
        "execution_assumption": "next_open_to_next_open",
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
        "classification": "crash_control_candidate",
        "production_ready": False,
        "paper_trading_only": True,
        "auto_paper_ledger_enabled": True,
        "manual_ledger_required": False,
        "paper_ledger_start_mode": "historical_backfill",
        "paper_ledger_start_date": None,
        "paper_ledger_allow_historical_backfill": True,
        "paper_ledger_reset_allowed": False,
    }
    config.update(overrides)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_signals(signal_dir: Path, rows: list[dict[str, object]]) -> None:
    signal_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(signal_dir / "signal_history.csv", index=False)
    latest = rows[-1]
    (signal_dir / "signal_today.json").write_text(
        (
            "{"
            f"\"latest_price_date\": \"{latest['latest_price_date']}\", "
            "\"data_quality_status\": \"ok\""
            "}"
        ),
        encoding="utf-8",
    )


def _write_prices(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _base_signal_rows() -> list[dict[str, object]]:
    return [
        {
            "latest_price_date": "2020-01-01",
            "target_symbol": "TQQQ",
            "target_exposure_next_session": 1.0,
            "previous_target_exposure": 0.0,
        },
        {
            "latest_price_date": "2020-01-02",
            "target_symbol": "TQQQ",
            "target_exposure_next_session": 1.5,
            "previous_target_exposure": 1.0,
        },
    ]


def _base_prices() -> list[dict[str, object]]:
    return [
        {"Date": "2020-01-01", "TQQQ": 100.0, "TQQQ_OPEN": 99.0, "TQQQ_CLOSE": 100.0},
        {"Date": "2020-01-02", "TQQQ": 102.0, "TQQQ_OPEN": 100.0, "TQQQ_CLOSE": 102.0},
        {"Date": "2020-01-03", "TQQQ": 104.0, "TQQQ_OPEN": 104.0, "TQQQ_CLOSE": 104.0},
    ]


def test_auto_paper_ledger_writes_one_row_per_signal_date(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir, _base_signal_rows())
    prices = _write_prices(tmp_path / "prices.csv", _base_prices())

    result = update_auto_paper_ledger(
        config_path=config,
        signal_dir=signal_dir,
        output_dir=tmp_path / "out",
        data_csv=prices,
    )

    assert result.ledger_path.exists()
    assert result.summary_path.exists()
    assert result.report_path.exists()
    assert len(result.ledger) == 2
    assert result.ledger["signal_date"].is_unique
    assert (tmp_path / "out" / "paper_equity_vs_tqqq.png").exists()
    assert (tmp_path / "out" / "relative_equity_vs_tqqq.png").exists()
    assert (tmp_path / "out" / "paper_exposure_history.png").exists()
    assert (tmp_path / "out" / "paper_trade_delta_history.png").exists()


def test_duplicate_signal_date_is_skipped_not_duplicated(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    signal_dir = tmp_path / "signals"
    rows = _base_signal_rows() + [_base_signal_rows()[-1]]
    _write_signals(signal_dir, rows)
    prices = _write_prices(tmp_path / "prices.csv", _base_prices())

    first = update_auto_paper_ledger(config_path=config, signal_dir=signal_dir, output_dir=tmp_path / "out", data_csv=prices)
    second = update_auto_paper_ledger(config_path=config, signal_dir=signal_dir, output_dir=tmp_path / "out", data_csv=prices)

    assert len(first.ledger) == 2
    assert len(second.ledger) == 2
    assert int(second.summary["duplicate_skipped_count"].iloc[0]) == 1


def test_missing_next_open_creates_pending_fill_then_resolves(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir, [_base_signal_rows()[0]])
    missing_prices = _write_prices(
        tmp_path / "prices_missing.csv",
        [{"Date": "2020-01-01", "TQQQ": 100.0, "TQQQ_OPEN": 99.0, "TQQQ_CLOSE": 100.0}],
    )

    pending = update_auto_paper_ledger(
        config_path=config,
        signal_dir=signal_dir,
        output_dir=tmp_path / "out",
        data_csv=missing_prices,
    )

    assert pending.latest_status == "pending_fill"

    resolved_prices = _write_prices(tmp_path / "prices_resolved.csv", _base_prices()[:2])
    resolved = update_auto_paper_ledger(
        config_path=config,
        signal_dir=signal_dir,
        output_dir=tmp_path / "out",
        data_csv=resolved_prices,
    )

    assert resolved.latest_status == "ok"
    assert resolved.ledger["status"].tolist() == ["ok"]


def test_transaction_cost_slippage_and_financing_are_applied(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "config.yaml",
        annual_financing_rate_assumption=0.252,
        financing_annual_cost=0.252,
        assumed_transaction_cost_bps=10.0,
        assumed_slippage_bps=5.0,
    )
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir, [_base_signal_rows()[1]])
    prices = _write_prices(tmp_path / "prices.csv", _base_prices())

    result = update_auto_paper_ledger(config_path=config, signal_dir=signal_dir, output_dir=tmp_path / "out", data_csv=prices)
    row = result.ledger.iloc[0]

    assert row["effective_fill_price"] > row["raw_fill_price"]
    assert row["transaction_cost"] == pytest.approx(150.0)
    assert row["slippage_cost"] == pytest.approx(75.0)
    assert row["financing_cost"] == pytest.approx(50.0)
    assert row["paper_equity"] == pytest.approx(99725.0)


def test_financing_cost_applies_only_above_threshold(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "config.yaml",
        annual_financing_rate_assumption=0.252,
        financing_annual_cost=0.252,
        assumed_transaction_cost_bps=0.0,
        assumed_slippage_bps=0.0,
    )
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir, _base_signal_rows())
    prices = _write_prices(tmp_path / "prices.csv", _base_prices())

    result = update_auto_paper_ledger(config_path=config, signal_dir=signal_dir, output_dir=tmp_path / "out", data_csv=prices)

    assert result.ledger["financing_cost"].iloc[0] == pytest.approx(0.0)
    assert result.ledger["financing_cost"].iloc[1] > 0.0


def test_benchmark_equity_uses_same_execution_dates(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml", assumed_transaction_cost_bps=0.0, assumed_slippage_bps=0.0)
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir, _base_signal_rows())
    prices = _write_prices(tmp_path / "prices.csv", _base_prices())

    result = update_auto_paper_ledger(config_path=config, signal_dir=signal_dir, output_dir=tmp_path / "out", data_csv=prices)

    assert result.ledger["execution_date"].tolist() == ["2020-01-02", "2020-01-03"]
    assert result.ledger["tqqq_buy_hold_equity"].iloc[0] == pytest.approx(100000.0)
    assert result.ledger["tqqq_buy_hold_equity"].iloc[1] == pytest.approx(104000.0)


def test_update_auto_paper_ledger_cli_writes_outputs(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    signal_dir = tmp_path / "signals"
    output_dir = tmp_path / "out"
    _write_signals(signal_dir, _base_signal_rows())
    prices = _write_prices(tmp_path / "prices.csv", _base_prices())

    assert (
        main(
            [
                "update-auto-paper-ledger",
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
    assert (output_dir / "auto_paper_ledger.csv").exists()
    assert (output_dir / "auto_paper_summary.csv").exists()
    assert (output_dir / "auto_paper_report.md").exists()


def test_auto_paper_ledger_has_no_broker_or_order_code() -> None:
    source = Path("research/auto_paper_ledger.py").read_text(encoding="utf-8")
    forbidden = {"broker_api", "submit_order", "place_order", "ib_insync", "create_order"}
    assert forbidden.isdisjoint(source)


def test_auto_paper_ledger_does_not_backfill_when_disabled(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "config.yaml",
        paper_ledger_start_mode="live_from_config_date",
        paper_ledger_start_date="2020-01-02",
        paper_ledger_allow_historical_backfill=False,
    )
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir, _base_signal_rows())
    prices = _write_prices(tmp_path / "prices.csv", _base_prices())

    result = update_auto_paper_ledger(
        config_path=config,
        signal_dir=signal_dir,
        output_dir=tmp_path / "out",
        data_csv=prices,
    )

    assert result.ledger["signal_date"].tolist() == ["2020-01-02"]
    assert set(result.ledger["ledger_mode"]) == {"live_monitor"}
    assert not result.ledger["historical_backfill"].astype(bool).any()


def test_missing_paper_ledger_start_date_gives_needs_start_date(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "config.yaml",
        paper_ledger_start_mode="live_from_config_date",
        paper_ledger_start_date=None,
        paper_ledger_allow_historical_backfill=False,
    )
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir, _base_signal_rows())
    prices = _write_prices(tmp_path / "prices.csv", _base_prices())

    result = update_auto_paper_ledger(
        config_path=config,
        signal_dir=signal_dir,
        output_dir=tmp_path / "out",
        data_csv=prices,
    )

    assert result.latest_status == "needs_start_date"
    assert int(result.summary["rows"].iloc[0]) == 1
    assert bool(result.summary["live_ledger_initialized"].iloc[0]) is False


def test_initialize_auto_paper_ledger_creates_clean_live_state(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml", paper_ledger_allow_historical_backfill=False)

    result = initialize_auto_paper_ledger(
        config_path=config,
        start_date="2026-06-18",
        starting_equity=100000,
        output_dir=tmp_path / "out",
        confirm_reset=True,
    )

    assert result.state_path.exists()
    assert result.ledger_path.exists()
    ledger = pd.read_csv(result.ledger_path)
    assert ledger["status"].iloc[0] == "initialized"
    assert ledger["ledger_mode"].iloc[0] == "live_monitor"
    assert ledger["paper_equity"].iloc[0] == pytest.approx(100000)
    assert result.state["paper_ledger_start_date"] == "2026-06-18"


def test_initialize_auto_paper_ledger_reset_requires_confirmation(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "auto_paper_ledger.csv").write_text("status\nok\n", encoding="utf-8")

    with pytest.raises(ValueError, match="confirm-reset"):
        initialize_auto_paper_ledger(
            config_path=config,
            start_date="2026-06-18",
            starting_equity=100000,
            output_dir=output_dir,
            confirm_reset=False,
        )


def test_initialize_auto_paper_ledger_cli_creates_state(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "config.yaml", paper_ledger_allow_historical_backfill=False)
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "initialize-auto-paper-ledger",
                "--config",
                str(config),
                "--start-date",
                "2026-06-18",
                "--starting-equity",
                "100000",
                "--output-dir",
                str(output_dir),
                "--confirm-reset",
            ]
        )
        == 0
    )
    assert (output_dir / "auto_paper_ledger_state.json").exists()
    assert (output_dir / "auto_paper_ledger.csv").exists()
