from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.cli import main
from research.paper_trade_reconciliation import LEDGER_COLUMNS, reconcile_paper_trades


def _write_signals(signal_dir: Path) -> None:
    signal_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "latest_price_date": "2026-06-12",
                "target_symbol": "TQQQ",
                "target_exposure_next_session": 1.20,
                "previous_target_exposure": 1.10,
                "trade_delta": 0.10,
                "estimated_transaction_cost": 0.00010,
                "estimated_financing_cost": 0.00004,
            },
            {
                "latest_price_date": "2026-06-13",
                "target_symbol": "TQQQ",
                "target_exposure_next_session": 0.80,
                "previous_target_exposure": 1.20,
                "trade_delta": -0.40,
                "estimated_transaction_cost": 0.00040,
                "estimated_financing_cost": 0.0,
            },
        ]
    ).to_csv(signal_dir / "signal_history.csv", index=False)


def _write_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2026-06-13",
                "signal_date": "2026-06-12",
                "intended_execution_date": "2026-06-15",
                "symbol": "TQQQ",
                "target_exposure": 1.25,
                "previous_exposure": 1.10,
                "trade_delta": 0.15,
                "reference_close": 99.0,
                "next_open": 100.0,
                "assumed_fill_price": 101.0,
                "shares_paper": 12.0,
                "notional_paper": 1212.0,
                "estimated_transaction_cost": 0.00011,
                "estimated_slippage": 0.00002,
                "estimated_financing_cost": 0.00005,
                "notes": "manual fixture",
            },
            {
                "date": "2026-06-15",
                "signal_date": "2026-06-14",
                "intended_execution_date": "2026-06-16",
                "symbol": "TQQQ",
                "target_exposure": 1.00,
                "previous_exposure": 1.25,
                "trade_delta": -0.25,
                "reference_close": 102.0,
                "next_open": 103.0,
                "assumed_fill_price": 103.0,
                "shares_paper": 10.0,
                "notional_paper": 1030.0,
                "estimated_transaction_cost": 0.00025,
                "estimated_slippage": 0.0,
                "estimated_financing_cost": 0.0,
                "notes": "extra ledger row",
            },
        ]
    ).to_csv(path, index=False)


def test_template_contains_required_ledger_columns() -> None:
    template = Path("data/paper_trading/voltarget_paper_trades_template.csv")
    columns = pd.read_csv(template).columns.tolist()
    assert columns == list(LEDGER_COLUMNS)


def test_missing_ledger_file_marks_all_signal_dates_missing(tmp_path: Path) -> None:
    signal_dir = tmp_path / "signals"
    _write_signals(signal_dir)

    result = reconcile_paper_trades(
        signal_dir=signal_dir,
        ledger_path=tmp_path / "missing_ledger.csv",
        output_dir=tmp_path / "out",
    )

    assert result.reconciliation_path.exists()
    assert result.report_path.exists()
    assert set(result.reconciliation["warnings"]) == {"missing_ledger_row"}
    assert result.reconciliation["has_signal"].all()
    assert not result.reconciliation["has_ledger"].any()


def test_reconciliation_reports_missing_dates_and_drift(tmp_path: Path) -> None:
    signal_dir = tmp_path / "signals"
    ledger_path = tmp_path / "ledger.csv"
    _write_signals(signal_dir)
    _write_ledger(ledger_path)

    result = reconcile_paper_trades(
        signal_dir=signal_dir,
        ledger_path=ledger_path,
        output_dir=tmp_path / "out",
    )

    by_date = result.reconciliation.set_index("signal_date")
    assert "realized_paper_exposure_drift" in by_date.loc["2026-06-12", "warnings"]
    assert "execution_price_drift" in by_date.loc["2026-06-12", "warnings"]
    assert "trade_delta_drift" in by_date.loc["2026-06-12", "warnings"]
    assert "cost_estimate_drift" in by_date.loc["2026-06-12", "warnings"]
    assert by_date.loc["2026-06-12", "realized_paper_exposure_drift"] == pytest.approx(0.05)
    assert by_date.loc["2026-06-12", "execution_price_drift"] == pytest.approx(0.01)
    assert by_date.loc["2026-06-13", "warnings"] == "missing_ledger_row"
    assert by_date.loc["2026-06-14", "warnings"] == "missing_signal_date"
    report = result.report_path.read_text(encoding="utf-8")
    assert "Missing signal dates" in report
    assert "Missing ledger rows" in report


def test_reconcile_paper_trades_cli_writes_outputs(tmp_path: Path) -> None:
    signal_dir = tmp_path / "signals"
    ledger_path = tmp_path / "ledger.csv"
    output_dir = tmp_path / "out"
    _write_signals(signal_dir)
    _write_ledger(ledger_path)

    assert (
        main(
            [
                "reconcile-paper-trades",
                "--signal-dir",
                str(signal_dir),
                "--ledger",
                str(ledger_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert (output_dir / "paper_trade_reconciliation.csv").exists()
    assert (output_dir / "paper_trade_reconciliation_report.md").exists()


def test_no_broker_or_order_integration_code_exists() -> None:
    source = Path("research/paper_trade_reconciliation.py").read_text(encoding="utf-8")
    forbidden = {"broker_api", "submit_order", "place_order", "ib_insync", "create_order"}
    assert forbidden.isdisjoint(source)
