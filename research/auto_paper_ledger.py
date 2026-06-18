from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_prices
from .execution import ohlc_column
from .voltarget_live_monitor import load_monitor_config


LEDGER_STATUSES = (
    "ok",
    "pending_fill",
    "stale_data",
    "missing_price_data",
    "duplicate_skipped",
    "initialized",
    "needs_start_date",
    "failed",
)

STATE_FILENAME = "auto_paper_ledger_state.json"


@dataclass(frozen=True)
class AutoPaperLedgerResult:
    output_dir: Path
    ledger_path: Path
    summary_path: Path
    report_path: Path
    chart_paths: Dict[str, Path]
    ledger: pd.DataFrame
    summary: pd.DataFrame
    latest_status: str
    latest_row: Dict[str, Any]


@dataclass(frozen=True)
class AutoPaperLedgerInitResult:
    output_dir: Path
    state_path: Path
    ledger_path: Path
    state: Dict[str, Any]
    reset_performed: bool


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _read_signal_history(signal_dir: Path) -> pd.DataFrame:
    path = Path(signal_dir) / "signal_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing signal history: {path}")
    signals = pd.read_csv(path)
    required = ["latest_price_date", "target_symbol", "target_exposure_next_session"]
    missing = [column for column in required if column not in signals.columns]
    if missing:
        raise ValueError(f"signal_history.csv missing required columns: {', '.join(missing)}")
    out = signals.copy()
    out["signal_date"] = pd.to_datetime(out["latest_price_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["signal_date"])
    out = out.sort_values("signal_date").drop_duplicates(subset=["signal_date"], keep="last")
    return out.reset_index(drop=True)


def _load_price_frame(
    *,
    config: Dict[str, Any],
    signals: pd.DataFrame,
    data_csv: Optional[Path],
) -> pd.DataFrame:
    start = pd.Timestamp(signals["signal_date"].min()).date().isoformat()
    target_symbol = str(config.get("target_symbol", "TQQQ")).upper()
    benchmark_symbol = str(config.get("benchmark_symbol", target_symbol)).upper()
    if data_csv is not None:
        prices = pd.read_csv(data_csv, parse_dates=["Date"]).set_index("Date").sort_index()
        prices = prices.loc[prices.index >= pd.Timestamp(start)]
    else:
        symbols = tuple(dict.fromkeys([target_symbol, benchmark_symbol]))
        prices = load_prices(
            start=start,
            symbols=symbols,
            cache_dir=str(config.get("cache_dir", "./price_cache")),
            include_ohlc=True,
            use_csv_if_exists=True,
            dropna=False,
            allow_missing_symbols=False,
        )
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    return prices.sort_index()


def _price_column(prices: pd.DataFrame, symbol: str, field: str) -> str:
    ohlc = ohlc_column(symbol, field)
    if ohlc in prices.columns:
        return ohlc
    if field.upper() == "CLOSE" and symbol in prices.columns:
        return symbol
    raise ValueError(f"Price data missing required {field.upper()} column for {symbol}: {ohlc}")


def _next_index(index: pd.DatetimeIndex, date: pd.Timestamp) -> Optional[pd.Timestamp]:
    later = index[index > pd.Timestamp(date)]
    if len(later) == 0:
        return None
    return pd.Timestamp(later[0])


def _price_at(prices: pd.DataFrame, date: Optional[pd.Timestamp], column: str) -> float:
    if date is None or date not in prices.index:
        return np.nan
    value = prices.at[date, column]
    return _safe_float(value)


def _target_status(signal: pd.Series, signal_today: Dict[str, Any]) -> str:
    latest_today = str(signal_today.get("latest_price_date", ""))
    quality = str(signal_today.get("data_quality_status", ""))
    if latest_today and latest_today == pd.Timestamp(signal["signal_date"]).date().isoformat() and quality != "ok":
        return "stale_data"
    return "ok"


def _effective_fill_price(raw_fill: float, trade_shares: float, slippage_bps: float) -> float:
    if np.isnan(raw_fill):
        return np.nan
    if trade_shares == 0.0:
        return float(raw_fill)
    direction = 1.0 if trade_shares > 0.0 else -1.0
    return float(raw_fill) * (1.0 + direction * float(slippage_bps) / 10000.0)


def _build_ledger(
    *,
    config: Dict[str, Any],
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    signal_today: Dict[str, Any],
) -> pd.DataFrame:
    symbol = str(config.get("target_symbol", "TQQQ")).upper()
    benchmark_symbol = str(config.get("benchmark_symbol", symbol)).upper()
    open_col = _price_column(prices, symbol, "OPEN")
    close_col = _price_column(prices, symbol, "CLOSE")
    benchmark_open_col = _price_column(prices, benchmark_symbol, "OPEN")

    starting_equity = float(config.get("paper_starting_equity", 100000.0))
    slippage_bps = float(config.get("assumed_slippage_bps", config.get("slippage_bps", 0.0)))
    transaction_cost_bps = float(config.get("assumed_transaction_cost_bps", config.get("transaction_cost_bps", 0.0)))
    financing_rate = float(
        config.get("annual_financing_rate_assumption", config.get("financing_annual_cost", 0.0))
    )
    financing_threshold = float(config.get("financing_applies_above_exposure", 1.0))
    day_count = float(config.get("financing_day_count_basis", 252))
    fill_source = str(config.get("paper_fill_price_source", "next_open"))
    fallback_source = str(config.get("fallback_fill_price_source", "latest_close"))

    cash = starting_equity
    shares = 0.0
    benchmark_shares: Optional[float] = None
    previous_exposure = 0.0
    rows: list[Dict[str, Any]] = []

    for _, signal in signals.iterrows():
        signal_date = pd.Timestamp(signal["signal_date"])
        execution_date = _next_index(prices.index, signal_date)
        reference_close = _price_at(prices, signal_date, close_col)
        next_open = _price_at(prices, execution_date, open_col)
        fallback_price = reference_close if fallback_source == "latest_close" else np.nan
        target_exposure = _safe_float(signal.get("target_exposure_next_session"), 0.0)
        previous_signal_exposure = _safe_float(signal.get("previous_target_exposure"), previous_exposure)

        status = _target_status(signal, signal_today)
        warning = ""
        raw_fill_price = next_open if fill_source == "next_open" else np.nan
        if prices.empty:
            status = "missing_price_data"
            warning = "price data is empty"
        elif execution_date is None or np.isnan(raw_fill_price):
            status = "pending_fill"
            warning = "next open is not available yet"

        if status in {"pending_fill", "missing_price_data"}:
            mark_price = fallback_price if not np.isnan(fallback_price) else 0.0
            paper_equity = cash + shares * mark_price if mark_price else cash
            benchmark_equity = (
                benchmark_shares * mark_price
                if benchmark_shares is not None and mark_price
                else starting_equity
            )
            rows.append(
                {
                    "signal_date": signal_date.date().isoformat(),
                    "execution_date": "",
                    "status": status,
                    "warning": warning,
                    "symbol": symbol,
                    "target_exposure": target_exposure,
                    "previous_target_exposure": previous_signal_exposure,
                    "previous_actual_exposure": previous_exposure,
                    "trade_delta_exposure": target_exposure - previous_exposure,
                    "paper_execution_model": str(config.get("paper_execution_model", "next_open_to_next_open")),
                    "fill_price_source": fill_source,
                    "fallback_fill_price_source": fallback_source,
                    "reference_close": reference_close,
                    "raw_fill_price": np.nan,
                    "effective_fill_price": np.nan,
                    "fallback_price": fallback_price,
                    "paper_shares": shares,
                    "paper_cash": cash,
                    "paper_equity": paper_equity,
                    "target_notional": np.nan,
                    "trade_shares": 0.0,
                    "trade_notional": 0.0,
                    "slippage_cost": 0.0,
                    "transaction_cost": 0.0,
                    "financing_cost": 0.0,
                    "benchmark_symbol": benchmark_symbol,
                    "benchmark_price": np.nan,
                    "tqqq_buy_hold_equity": benchmark_equity,
                    "relative_equity_vs_tqqq": paper_equity / benchmark_equity if benchmark_equity else np.nan,
                    "paper_trading_only": True,
                    "no_broker_integration": True,
                    "no_auto_trading": True,
                }
            )
            continue

        actual_exposure_before_trade = previous_exposure
        equity_before_trade = cash + shares * raw_fill_price
        target_notional = target_exposure * equity_before_trade
        target_shares = target_notional / raw_fill_price if raw_fill_price else 0.0
        trade_shares = target_shares - shares
        trade_notional = abs(trade_shares * raw_fill_price)
        effective_fill = _effective_fill_price(raw_fill_price, trade_shares, slippage_bps)
        slippage_cost = abs(trade_shares * (effective_fill - raw_fill_price))
        transaction_cost = trade_notional * (transaction_cost_bps / 10000.0)
        financing_cost = max(abs(target_exposure) - financing_threshold, 0.0) * equity_before_trade * (financing_rate / day_count)

        cash = cash - trade_shares * effective_fill - transaction_cost - financing_cost
        shares = target_shares
        paper_equity = cash + shares * raw_fill_price
        previous_exposure = target_exposure

        benchmark_price = _price_at(prices, execution_date, benchmark_open_col)
        if benchmark_shares is None and not np.isnan(benchmark_price) and benchmark_price != 0.0:
            benchmark_shares = starting_equity / benchmark_price
        benchmark_equity = benchmark_shares * benchmark_price if benchmark_shares is not None and not np.isnan(benchmark_price) else np.nan

        rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "execution_date": execution_date.date().isoformat() if execution_date is not None else "",
                "status": status,
                "warning": warning,
                "symbol": symbol,
                "target_exposure": target_exposure,
                "previous_target_exposure": previous_signal_exposure,
                "previous_actual_exposure": actual_exposure_before_trade,
                "trade_delta_exposure": target_exposure - actual_exposure_before_trade,
                "paper_execution_model": str(config.get("paper_execution_model", "next_open_to_next_open")),
                "fill_price_source": fill_source,
                "fallback_fill_price_source": fallback_source,
                "reference_close": reference_close,
                "raw_fill_price": raw_fill_price,
                "effective_fill_price": effective_fill,
                "fallback_price": fallback_price,
                "paper_shares": shares,
                "paper_cash": cash,
                "paper_equity": paper_equity,
                "target_notional": target_notional,
                "trade_shares": trade_shares,
                "trade_notional": trade_notional,
                "slippage_cost": slippage_cost,
                "transaction_cost": transaction_cost,
                "financing_cost": financing_cost,
                "benchmark_symbol": benchmark_symbol,
                "benchmark_price": benchmark_price,
                "tqqq_buy_hold_equity": benchmark_equity,
                "relative_equity_vs_tqqq": paper_equity / benchmark_equity if benchmark_equity else np.nan,
                "paper_trading_only": True,
                "no_broker_integration": True,
                "no_auto_trading": True,
            }
        )

    return pd.DataFrame(rows)


def _empty_status_ledger(config: Dict[str, Any], status: str, start_date: str = "", starting_equity: Optional[float] = None) -> pd.DataFrame:
    equity = float(starting_equity if starting_equity is not None else config.get("paper_starting_equity", 100000.0))
    return pd.DataFrame(
        [
            {
                "signal_date": start_date,
                "execution_date": "",
                "status": status,
                "warning": "paper_ledger_start_date is required" if status == "needs_start_date" else "",
                "symbol": str(config.get("target_symbol", "TQQQ")).upper(),
                "target_exposure": 0.0,
                "previous_target_exposure": 0.0,
                "previous_actual_exposure": 0.0,
                "trade_delta_exposure": 0.0,
                "paper_execution_model": str(config.get("paper_execution_model", "next_open_to_next_open")),
                "fill_price_source": str(config.get("paper_fill_price_source", "next_open")),
                "fallback_fill_price_source": str(config.get("fallback_fill_price_source", "latest_close")),
                "reference_close": np.nan,
                "raw_fill_price": np.nan,
                "effective_fill_price": np.nan,
                "fallback_price": np.nan,
                "paper_shares": 0.0,
                "paper_cash": equity,
                "paper_equity": equity,
                "target_notional": 0.0,
                "trade_shares": 0.0,
                "trade_notional": 0.0,
                "slippage_cost": 0.0,
                "transaction_cost": 0.0,
                "financing_cost": 0.0,
                "benchmark_symbol": str(config.get("benchmark_symbol", "TQQQ")).upper(),
                "benchmark_price": np.nan,
                "tqqq_buy_hold_equity": equity,
                "relative_equity_vs_tqqq": 1.0,
                "ledger_mode": "live_monitor",
                "paper_ledger_start_date": start_date,
                "historical_backfill": False,
                "live_ledger_initialized": status == "initialized",
                "paper_trading_only": True,
                "no_broker_integration": True,
                "no_auto_trading": True,
            }
        ]
    )


def _summary(ledger: pd.DataFrame, *, duplicate_skipped: int, ledger_mode: str, start_date: str, initialized: bool) -> pd.DataFrame:
    latest = ledger.iloc[-1].to_dict() if not ledger.empty else {}
    return pd.DataFrame(
        [
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "latest_signal_date": latest.get("signal_date", ""),
                "latest_execution_date": latest.get("execution_date", ""),
                "latest_status": latest.get("status", "failed"),
                "latest_target_exposure": latest.get("target_exposure", np.nan),
                "latest_paper_equity": latest.get("paper_equity", np.nan),
                "latest_relative_equity_vs_tqqq": latest.get("relative_equity_vs_tqqq", np.nan),
                "rows": int(len(ledger)),
                "pending_fill_count": int(ledger["status"].eq("pending_fill").sum()) if not ledger.empty else 0,
                "stale_data_count": int(ledger["status"].eq("stale_data").sum()) if not ledger.empty else 0,
                "missing_price_data_count": int(ledger["status"].eq("missing_price_data").sum()) if not ledger.empty else 0,
                "duplicate_skipped_count": int(duplicate_skipped),
                "auto_paper_ledger_mode": ledger_mode,
                "paper_ledger_start_date": start_date,
                "current_ledger_kind": latest.get("ledger_mode", ledger_mode),
                "historical_backfill": bool(latest.get("historical_backfill", ledger_mode == "historical_backfill")),
                "live_ledger_initialized": bool(initialized),
                "paper_trading_only": True,
                "no_broker_integration": True,
                "no_auto_trading": True,
            }
        ]
    )


def _write_chart(ledger: pd.DataFrame, output_dir: Path, filename: str, columns: list[str], title: str, ylabel: str) -> Path:
    path = output_dir / filename
    fig, ax = plt.subplots(figsize=(10, 5))
    plot = ledger.copy()
    plot["plot_date"] = pd.to_datetime(plot["execution_date"].where(plot["execution_date"].astype(str).ne(""), plot["signal_date"]))
    plot = plot.set_index("plot_date")
    if plot.empty:
        ax.text(0.5, 0.5, "No auto paper ledger rows", ha="center", va="center")
    else:
        plot[columns].astype(float).plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_report(path: Path, summary: pd.DataFrame, chart_paths: Dict[str, Path], ledger_path: Path) -> None:
    latest = summary.iloc[0]
    lines = [
        "# VolTarget Automated Paper Ledger Report",
        "",
        "**paper monitoring only**",
        "",
        "This automated ledger simulates paper fills from generated signals and observable market prices. It does not place trades, connect to brokers, or create live orders.",
        "",
        "## Latest Status",
        f"- Latest signal date: `{latest['latest_signal_date']}`",
        f"- Latest execution date: `{latest['latest_execution_date']}`",
        f"- Latest status: `{latest['latest_status']}`",
        f"- Latest target exposure: `{latest['latest_target_exposure']}`",
        f"- Latest paper equity: `{latest['latest_paper_equity']}`",
        f"- Latest relative equity vs TQQQ: `{latest['latest_relative_equity_vs_tqqq']}`",
        f"- Auto paper ledger mode: `{latest['auto_paper_ledger_mode']}`",
        f"- Paper ledger start date: `{latest['paper_ledger_start_date']}`",
        f"- Current ledger kind: `{latest['current_ledger_kind']}`",
        f"- Live ledger initialized: `{latest['live_ledger_initialized']}`",
        f"- Pending fills: `{latest['pending_fill_count']}`",
        f"- Stale data rows: `{latest['stale_data_count']}`",
        "",
        "## Outputs",
        f"- Auto paper ledger: `{ledger_path}`",
        f"- Paper equity vs TQQQ: `{chart_paths['paper_equity_vs_tqqq']}`",
        f"- Relative equity vs TQQQ: `{chart_paths['relative_equity_vs_tqqq']}`",
        f"- Paper exposure history: `{chart_paths['paper_exposure_history']}`",
        f"- Paper trade delta history: `{chart_paths['paper_trade_delta_history']}`",
        "",
        "## Status Meanings",
        "- `ok`: the signal has an observable fill under the configured paper execution assumption.",
        "- `pending_fill`: the next open needed for the simulated fill is not available yet.",
        "- `stale_data`: the latest generated signal was already flagged stale by the signal monitor.",
        "- `missing_price_data`: required price data was unavailable.",
        "- `duplicate_skipped`: duplicate signal dates were ignored during normalization.",
        "- `failed`: the ledger update failed before producing a valid row.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def update_auto_paper_ledger(
    *,
    config_path: Path,
    signal_dir: Path,
    output_dir: Path,
    data_csv: Optional[Path] = None,
) -> AutoPaperLedgerResult:
    config = load_monitor_config(config_path)
    signal_dir = Path(signal_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _read_json(output_dir / STATE_FILENAME)
    allow_backfill = bool(config.get("paper_ledger_allow_historical_backfill", True))
    state_start_date = str(state.get("paper_ledger_start_date", "") or "")
    config_start_date = str(config.get("paper_ledger_start_date") or "")
    start_date = state_start_date or config_start_date
    initialized = bool(state.get("ledger_mode") == "live_monitor" and state_start_date)
    if state.get("starting_equity") not in {None, ""}:
        config["paper_starting_equity"] = float(state["starting_equity"])
    ledger_mode = "historical_backfill" if allow_backfill else "live_monitor"

    if not allow_backfill and not start_date:
        ledger = _empty_status_ledger(config, "needs_start_date")
        ledger_path = output_dir / "auto_paper_ledger.csv"
        summary_path = output_dir / "auto_paper_summary.csv"
        report_path = output_dir / "auto_paper_report.md"
        ledger.to_csv(ledger_path, index=False)
        summary = _summary(ledger, duplicate_skipped=0, ledger_mode=ledger_mode, start_date="", initialized=False)
        summary.to_csv(summary_path, index=False)
        chart_paths = {
            "paper_equity_vs_tqqq": _write_chart(ledger, output_dir, "paper_equity_vs_tqqq.png", ["paper_equity", "tqqq_buy_hold_equity"], "Auto Paper Equity vs TQQQ", "Equity"),
            "relative_equity_vs_tqqq": _write_chart(ledger, output_dir, "relative_equity_vs_tqqq.png", ["relative_equity_vs_tqqq"], "Auto Paper Relative Equity vs TQQQ", "Relative equity"),
            "paper_exposure_history": _write_chart(ledger, output_dir, "paper_exposure_history.png", ["target_exposure"], "Auto Paper Exposure History", "Exposure"),
            "paper_trade_delta_history": _write_chart(ledger, output_dir, "paper_trade_delta_history.png", ["trade_delta_exposure"], "Auto Paper Trade Delta History", "Exposure delta"),
        }
        _write_report(report_path, summary, chart_paths, ledger_path)
        latest_row = ledger.iloc[-1].to_dict()
        return AutoPaperLedgerResult(output_dir, ledger_path, summary_path, report_path, chart_paths, ledger, summary, "needs_start_date", latest_row)

    signal_today = _read_json(signal_dir / "signal_today.json")
    raw_signals = pd.read_csv(signal_dir / "signal_history.csv") if (signal_dir / "signal_history.csv").exists() else pd.DataFrame()
    signals = _read_signal_history(signal_dir)
    if not allow_backfill:
        signals = signals.loc[signals["signal_date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
        if signals.empty:
            ledger = _empty_status_ledger(config, "initialized", start_date=start_date, starting_equity=float(config["paper_starting_equity"]))
            ledger_path = output_dir / "auto_paper_ledger.csv"
            summary_path = output_dir / "auto_paper_summary.csv"
            report_path = output_dir / "auto_paper_report.md"
            ledger.to_csv(ledger_path, index=False)
            summary = _summary(ledger, duplicate_skipped=0, ledger_mode=ledger_mode, start_date=start_date, initialized=initialized)
            summary.to_csv(summary_path, index=False)
            chart_paths = {
                "paper_equity_vs_tqqq": _write_chart(ledger, output_dir, "paper_equity_vs_tqqq.png", ["paper_equity", "tqqq_buy_hold_equity"], "Auto Paper Equity vs TQQQ", "Equity"),
                "relative_equity_vs_tqqq": _write_chart(ledger, output_dir, "relative_equity_vs_tqqq.png", ["relative_equity_vs_tqqq"], "Auto Paper Relative Equity vs TQQQ", "Relative equity"),
                "paper_exposure_history": _write_chart(ledger, output_dir, "paper_exposure_history.png", ["target_exposure"], "Auto Paper Exposure History", "Exposure"),
                "paper_trade_delta_history": _write_chart(ledger, output_dir, "paper_trade_delta_history.png", ["trade_delta_exposure"], "Auto Paper Trade Delta History", "Exposure delta"),
            }
            _write_report(report_path, summary, chart_paths, ledger_path)
            latest_row = ledger.iloc[-1].to_dict()
            return AutoPaperLedgerResult(output_dir, ledger_path, summary_path, report_path, chart_paths, ledger, summary, "initialized", latest_row)
    duplicate_skipped = max(int(len(raw_signals) - len(signals)), 0)
    prices = _load_price_frame(config=config, signals=signals, data_csv=data_csv)
    ledger = _build_ledger(config=config, signals=signals, prices=prices, signal_today=signal_today)
    ledger["ledger_mode"] = ledger_mode
    ledger["paper_ledger_start_date"] = start_date
    ledger["historical_backfill"] = bool(allow_backfill)
    ledger["live_ledger_initialized"] = bool(initialized or allow_backfill)

    ledger_path = output_dir / "auto_paper_ledger.csv"
    summary_path = output_dir / "auto_paper_summary.csv"
    report_path = output_dir / "auto_paper_report.md"
    ledger.to_csv(ledger_path, index=False)
    summary = _summary(ledger, duplicate_skipped=duplicate_skipped, ledger_mode=ledger_mode, start_date=start_date, initialized=bool(initialized or allow_backfill))
    summary.to_csv(summary_path, index=False)
    chart_paths = {
        "paper_equity_vs_tqqq": _write_chart(
            ledger,
            output_dir,
            "paper_equity_vs_tqqq.png",
            ["paper_equity", "tqqq_buy_hold_equity"],
            "Auto Paper Equity vs TQQQ",
            "Equity",
        ),
        "relative_equity_vs_tqqq": _write_chart(
            ledger,
            output_dir,
            "relative_equity_vs_tqqq.png",
            ["relative_equity_vs_tqqq"],
            "Auto Paper Relative Equity vs TQQQ",
            "Relative equity",
        ),
        "paper_exposure_history": _write_chart(
            ledger,
            output_dir,
            "paper_exposure_history.png",
            ["target_exposure"],
            "Auto Paper Exposure History",
            "Exposure",
        ),
        "paper_trade_delta_history": _write_chart(
            ledger,
            output_dir,
            "paper_trade_delta_history.png",
            ["trade_delta_exposure"],
            "Auto Paper Trade Delta History",
            "Exposure delta",
        ),
    }
    _write_report(report_path, summary, chart_paths, ledger_path)
    latest_row = ledger.iloc[-1].to_dict() if not ledger.empty else {"status": "failed"}
    return AutoPaperLedgerResult(
        output_dir=output_dir,
        ledger_path=ledger_path,
        summary_path=summary_path,
        report_path=report_path,
        chart_paths=chart_paths,
        ledger=ledger,
        summary=summary,
        latest_status=str(latest_row.get("status", "failed")),
        latest_row=latest_row,
    )


def initialize_auto_paper_ledger(
    *,
    config_path: Path,
    start_date: str,
    starting_equity: float,
    output_dir: Path,
    confirm_reset: bool = False,
) -> AutoPaperLedgerInitResult:
    config = load_monitor_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STATE_FILENAME
    ledger_path = output_dir / "auto_paper_ledger.csv"
    if ledger_path.exists() and not confirm_reset:
        raise ValueError("auto_paper_ledger.csv exists; pass --confirm-reset to create/reset it")
    state = {
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "ledger_mode": "live_monitor",
        "paper_ledger_start_date": pd.Timestamp(start_date).date().isoformat(),
        "starting_equity": float(starting_equity),
        "historical_backfill": False,
        "paper_trading_only": True,
        "no_broker_integration": True,
        "no_auto_trading": True,
    }
    _write_json(state_path, state)
    reset_performed = False
    if confirm_reset:
        ledger = _empty_status_ledger(
            config,
            "initialized",
            start_date=state["paper_ledger_start_date"],
            starting_equity=float(starting_equity),
        )
        ledger.to_csv(ledger_path, index=False)
        reset_performed = True
    return AutoPaperLedgerInitResult(
        output_dir=output_dir,
        state_path=state_path,
        ledger_path=ledger_path,
        state=state,
        reset_performed=reset_performed,
    )
