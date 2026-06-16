from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


LEDGER_COLUMNS = (
    "date",
    "signal_date",
    "intended_execution_date",
    "symbol",
    "target_exposure",
    "previous_exposure",
    "trade_delta",
    "reference_close",
    "next_open",
    "assumed_fill_price",
    "shares_paper",
    "notional_paper",
    "estimated_transaction_cost",
    "estimated_slippage",
    "estimated_financing_cost",
    "notes",
)

REQUIRED_SIGNAL_COLUMNS = (
    "latest_price_date",
    "target_symbol",
    "target_exposure_next_session",
    "previous_target_exposure",
    "trade_delta",
    "estimated_transaction_cost",
    "estimated_financing_cost",
)


@dataclass(frozen=True)
class PaperTradeReconciliationResult:
    output_dir: Path
    reconciliation_path: Path
    report_path: Path
    reconciliation: pd.DataFrame


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=list(LEDGER_COLUMNS))


def _read_signal_history(signal_dir: Path) -> pd.DataFrame:
    path = Path(signal_dir) / "signal_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing signal history: {path}")
    signals = pd.read_csv(path)
    missing = [column for column in REQUIRED_SIGNAL_COLUMNS if column not in signals.columns]
    if missing:
        raise ValueError(f"signal_history.csv missing required columns: {', '.join(missing)}")
    signals = signals.copy()
    signals["signal_date"] = pd.to_datetime(signals["latest_price_date"]).dt.date.astype(str)
    return signals.sort_values("signal_date").drop_duplicates(subset=["signal_date"], keep="last")


def _read_ledger(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return _empty_ledger()
    ledger = pd.read_csv(path)
    missing = [column for column in LEDGER_COLUMNS if column not in ledger.columns]
    if missing:
        raise ValueError(f"paper trade ledger missing required columns: {', '.join(missing)}")
    ledger = ledger.copy()
    if ledger.empty:
        return ledger
    ledger["signal_date"] = pd.to_datetime(ledger["signal_date"]).dt.date.astype(str)
    ledger["date"] = pd.to_datetime(ledger["date"]).dt.date.astype(str)
    ledger["intended_execution_date"] = pd.to_datetime(ledger["intended_execution_date"]).dt.date.astype(str)
    return ledger.sort_values(["signal_date", "date"]).drop_duplicates(subset=["signal_date"], keep="last")


def _as_float(row: pd.Series, column: str, default: float = np.nan) -> float:
    try:
        value = row.get(column, default)
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct_drift(numerator: float, denominator: float) -> float:
    if np.isnan(numerator) or np.isnan(denominator) or denominator == 0.0:
        return np.nan
    return numerator / denominator - 1.0


def _join_warnings(parts: Iterable[str]) -> str:
    return ";".join(part for part in parts if part)


def _reconcile_row(signal: Optional[pd.Series], ledger: Optional[pd.Series], signal_date: str) -> Dict[str, object]:
    warnings: List[str] = []
    has_signal = signal is not None
    has_ledger = ledger is not None
    if not has_signal:
        warnings.append("missing_signal_date")
    if not has_ledger:
        warnings.append("missing_ledger_row")

    signal_target = _as_float(signal, "target_exposure_next_session") if has_signal else np.nan
    signal_previous = _as_float(signal, "previous_target_exposure") if has_signal else np.nan
    signal_delta = _as_float(signal, "trade_delta") if has_signal else np.nan
    signal_transaction_cost = _as_float(signal, "estimated_transaction_cost") if has_signal else np.nan
    signal_financing_cost = _as_float(signal, "estimated_financing_cost") if has_signal else np.nan

    ledger_target = _as_float(ledger, "target_exposure") if has_ledger else np.nan
    ledger_previous = _as_float(ledger, "previous_exposure") if has_ledger else np.nan
    ledger_delta = _as_float(ledger, "trade_delta") if has_ledger else np.nan
    next_open = _as_float(ledger, "next_open") if has_ledger else np.nan
    fill_price = _as_float(ledger, "assumed_fill_price") if has_ledger else np.nan
    ledger_transaction_cost = _as_float(ledger, "estimated_transaction_cost") if has_ledger else np.nan
    ledger_slippage = _as_float(ledger, "estimated_slippage", 0.0) if has_ledger else np.nan
    ledger_financing_cost = _as_float(ledger, "estimated_financing_cost") if has_ledger else np.nan

    exposure_drift = ledger_target - signal_target if has_signal and has_ledger else np.nan
    previous_exposure_drift = ledger_previous - signal_previous if has_signal and has_ledger else np.nan
    trade_delta_drift = ledger_delta - signal_delta if has_signal and has_ledger else np.nan
    execution_price_drift = _pct_drift(fill_price, next_open)
    signal_total_cost = signal_transaction_cost + signal_financing_cost if has_signal else np.nan
    ledger_total_cost = (
        ledger_transaction_cost + ledger_slippage + ledger_financing_cost if has_ledger else np.nan
    )
    cost_estimate_drift = ledger_total_cost - signal_total_cost if has_signal and has_ledger else np.nan

    if has_signal and has_ledger:
        if not np.isnan(execution_price_drift) and abs(execution_price_drift) > 0.001:
            warnings.append("execution_price_drift")
        if not np.isnan(exposure_drift) and abs(exposure_drift) > 1e-6:
            warnings.append("realized_paper_exposure_drift")
        if not np.isnan(trade_delta_drift) and abs(trade_delta_drift) > 1e-6:
            warnings.append("trade_delta_drift")
        if not np.isnan(cost_estimate_drift) and abs(cost_estimate_drift) > 1e-8:
            warnings.append("cost_estimate_drift")

    if has_ledger:
        symbol = ledger.get("symbol", "")
    elif has_signal:
        symbol = signal.get("target_symbol", "")
    else:
        symbol = ""

    return {
        "signal_date": signal_date,
        "ledger_date": ledger.get("date", "") if has_ledger else "",
        "intended_execution_date": ledger.get("intended_execution_date", "") if has_ledger else "",
        "symbol": symbol,
        "has_signal": has_signal,
        "has_ledger": has_ledger,
        "signal_target_exposure": signal_target,
        "ledger_target_exposure": ledger_target,
        "realized_paper_exposure_drift": exposure_drift,
        "signal_previous_exposure": signal_previous,
        "ledger_previous_exposure": ledger_previous,
        "previous_exposure_drift": previous_exposure_drift,
        "signal_trade_delta": signal_delta,
        "ledger_trade_delta": ledger_delta,
        "trade_delta_drift": trade_delta_drift,
        "reference_close": _as_float(ledger, "reference_close") if has_ledger else np.nan,
        "next_open": next_open,
        "assumed_fill_price": fill_price,
        "execution_price_drift": execution_price_drift,
        "signal_estimated_transaction_cost": signal_transaction_cost,
        "ledger_estimated_transaction_cost": ledger_transaction_cost,
        "ledger_estimated_slippage": ledger_slippage,
        "signal_estimated_financing_cost": signal_financing_cost,
        "ledger_estimated_financing_cost": ledger_financing_cost,
        "signal_total_cost_estimate": signal_total_cost,
        "ledger_total_cost_estimate": ledger_total_cost,
        "cost_estimate_drift": cost_estimate_drift,
        "shares_paper": _as_float(ledger, "shares_paper") if has_ledger else np.nan,
        "notional_paper": _as_float(ledger, "notional_paper") if has_ledger else np.nan,
        "warnings": _join_warnings(warnings),
        "notes": ledger.get("notes", "") if has_ledger else "",
    }


def reconcile_paper_trades(
    *,
    signal_dir: Path,
    ledger_path: Path,
    output_dir: Path,
) -> PaperTradeReconciliationResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals = _read_signal_history(Path(signal_dir))
    ledger = _read_ledger(Path(ledger_path))

    signal_by_date = {str(row["signal_date"]): row for _, row in signals.iterrows()}
    ledger_by_date = {str(row["signal_date"]): row for _, row in ledger.iterrows()} if not ledger.empty else {}
    all_dates = sorted(set(signal_by_date) | set(ledger_by_date))
    rows = [
        _reconcile_row(signal_by_date.get(date), ledger_by_date.get(date), date)
        for date in all_dates
    ]
    reconciliation = pd.DataFrame(rows)
    reconciliation_path = output_dir / "paper_trade_reconciliation.csv"
    report_path = output_dir / "paper_trade_reconciliation_report.md"
    reconciliation.to_csv(reconciliation_path, index=False)
    _write_report(report_path, reconciliation, Path(ledger_path), Path(signal_dir), reconciliation_path)
    return PaperTradeReconciliationResult(
        output_dir=output_dir,
        reconciliation_path=reconciliation_path,
        report_path=report_path,
        reconciliation=reconciliation,
    )


def _count_warning(frame: pd.DataFrame, warning: str) -> int:
    if frame.empty or "warnings" not in frame.columns:
        return 0
    return int(frame["warnings"].fillna("").astype(str).str.contains(warning, regex=False).sum())


def _write_report(
    path: Path,
    reconciliation: pd.DataFrame,
    ledger_path: Path,
    signal_dir: Path,
    reconciliation_path: Path,
) -> None:
    missing_signal = _count_warning(reconciliation, "missing_signal_date")
    missing_ledger = _count_warning(reconciliation, "missing_ledger_row")
    execution_drift = _count_warning(reconciliation, "execution_price_drift")
    exposure_drift = _count_warning(reconciliation, "realized_paper_exposure_drift")
    cost_drift = _count_warning(reconciliation, "cost_estimate_drift")
    warning_rows = int(reconciliation["warnings"].fillna("").astype(str).ne("").sum()) if not reconciliation.empty else 0
    lines = [
        "# VolTarget Paper Trade Reconciliation",
        "",
        "**paper trading only**",
        "",
        "This report reconciles manually-entered paper-trade ledger rows against generated VolTarget paper signals.",
        "It does not place trades, create orders, call broker APIs, or provide investment advice.",
        "",
        "## Inputs",
        f"- Signal directory: `{signal_dir}`",
        f"- Ledger path: `{ledger_path}`",
        "",
        "## Outputs",
        f"- Reconciliation CSV: `{reconciliation_path}`",
        "",
        "## Summary",
        f"- Rows reviewed: `{len(reconciliation)}`",
        f"- Missing signal dates: `{missing_signal}`",
        f"- Missing ledger rows: `{missing_ledger}`",
        f"- Execution price drift warnings: `{execution_drift}`",
        f"- Realized paper exposure drift warnings: `{exposure_drift}`",
        f"- Cost estimate drift warnings: `{cost_drift}`",
        f"- Rows with any warning: `{warning_rows}`",
        "",
        "## Warning Types",
        "- `missing_signal_date`: ledger row exists for a signal date not present in signal history.",
        "- `missing_ledger_row`: signal history has a date without a manual ledger row.",
        "- `execution_price_drift`: assumed fill price differs from next open by more than 0.10%.",
        "- `realized_paper_exposure_drift`: manual target exposure differs from generated target exposure.",
        "- `trade_delta_drift`: manual trade delta differs from generated trade delta.",
        "- `cost_estimate_drift`: manual cost estimate differs from generated transaction plus financing estimate.",
        "",
        "## Boundary",
        "Manual ledger reconciliation is audit-only. It must not be used as auto-trading or broker execution infrastructure.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
