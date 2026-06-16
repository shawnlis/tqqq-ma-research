from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_prices
from .execution import ohlc_column
from .voltarget_live_monitor import load_monitor_config


DEFAULT_DRIFT_THRESHOLD = 0.01

REQUIRED_SIGNAL_COLUMNS = (
    "latest_price_date",
    "target_symbol",
    "target_exposure_next_session",
    "estimated_transaction_cost",
    "estimated_financing_cost",
    "current_paper_equity",
)


@dataclass(frozen=True)
class ExecutionDriftResult:
    output_dir: Path
    summary_path: Path
    report_path: Path
    chart_paths: Dict[str, Path]
    summary: pd.DataFrame
    metrics: Dict[str, float | bool | str]


def _read_signal_history(signal_dir: Path) -> pd.DataFrame:
    path = Path(signal_dir) / "signal_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing signal history: {path}")
    signals = pd.read_csv(path)
    missing = [column for column in REQUIRED_SIGNAL_COLUMNS if column not in signals.columns]
    if missing:
        raise ValueError(f"signal_history.csv missing required columns: {', '.join(missing)}")
    out = signals.copy()
    out["signal_date"] = pd.to_datetime(out["latest_price_date"]).dt.normalize()
    out = out.sort_values("signal_date").drop_duplicates(subset=["signal_date"], keep="last")
    out["monitor_model_return"] = out["current_paper_equity"].astype(float).pct_change(fill_method=None)
    return out.reset_index(drop=True)


def _price_column(data: pd.DataFrame, symbol: str, field: str) -> str:
    column = ohlc_column(symbol, field)
    if column in data.columns:
        return column
    if field.upper() == "CLOSE" and symbol in data.columns:
        return symbol
    raise ValueError(f"Price data missing required {field.upper()} column for {symbol}: {column}")


def _next_index(index: pd.DatetimeIndex, date: pd.Timestamp) -> Optional[pd.Timestamp]:
    later = index[index > pd.Timestamp(date)]
    return pd.Timestamp(later[0]) if len(later) else None


def _safe_price(data: pd.DataFrame, date: Optional[pd.Timestamp], column: str) -> float:
    if date is None or date not in data.index:
        return np.nan
    value = data.at[date, column]
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _pct_return(end: float, start: float) -> float:
    if np.isnan(start) or np.isnan(end) or start == 0.0:
        return np.nan
    return end / start - 1.0


def _build_drift_frame(signals: pd.DataFrame, prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices.sort_index()
    close_col = _price_column(prices, symbol, "CLOSE")
    open_col = _price_column(prices, symbol, "OPEN")
    rows = []
    for _, signal in signals.iterrows():
        signal_date = pd.Timestamp(signal["signal_date"])
        next_date = _next_index(prices.index, signal_date)
        following_date = _next_index(prices.index, next_date) if next_date is not None else None
        previous_close = _safe_price(prices, signal_date, close_col)
        next_open = _safe_price(prices, next_date, open_col)
        next_close = _safe_price(prices, next_date, close_col)
        following_open = _safe_price(prices, following_date, open_col)
        close_to_close_return = _pct_return(next_close, previous_close)
        open_to_close_return = _pct_return(next_close, next_open)
        open_to_open_return = _pct_return(following_open, next_open)
        exposure = float(signal["target_exposure_next_session"])
        transaction_cost = float(signal.get("estimated_transaction_cost", 0.0) or 0.0)
        financing_cost = float(signal.get("estimated_financing_cost", 0.0) or 0.0)
        close_model_return = exposure * close_to_close_return if not np.isnan(close_to_close_return) else np.nan
        next_open_model_return = exposure * open_to_open_return if not np.isnan(open_to_open_return) else np.nan
        paper_execution_return = (
            next_open_model_return - transaction_cost - financing_cost
            if not np.isnan(next_open_model_return)
            else np.nan
        )
        monitor_model_return = float(signal["monitor_model_return"]) if not pd.isna(signal["monitor_model_return"]) else np.nan
        warnings = []
        if np.isnan(previous_close):
            warnings.append("missing_previous_close")
        if np.isnan(next_open):
            warnings.append("missing_next_open")
        if np.isnan(next_close):
            warnings.append("missing_next_close")
        if np.isnan(following_open):
            warnings.append("missing_following_open")
        rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "target_symbol": symbol,
                "target_exposure": exposure,
                "previous_close": previous_close,
                "next_price_date": next_date.date().isoformat() if next_date is not None else "",
                "next_open": next_open,
                "next_close": next_close,
                "following_open_date": following_date.date().isoformat() if following_date is not None else "",
                "following_open": following_open,
                "close_to_close_return": close_to_close_return,
                "open_to_close_return": open_to_close_return,
                "open_to_open_return": open_to_open_return,
                "model_return": monitor_model_return,
                "close_to_close_research_model_return": close_model_return,
                "next_open_to_next_open_model_return": next_open_model_return,
                "paper_execution_return": paper_execution_return,
                "daily_drift_vs_close_to_close_research_model": paper_execution_return - close_model_return
                if not np.isnan(paper_execution_return) and not np.isnan(close_model_return)
                else np.nan,
                "daily_drift_vs_next_open_to_next_open_model": paper_execution_return - next_open_model_return
                if not np.isnan(paper_execution_return) and not np.isnan(next_open_model_return)
                else np.nan,
                "daily_drift_vs_monitor_model": paper_execution_return - monitor_model_return
                if not np.isnan(paper_execution_return) and not np.isnan(monitor_model_return)
                else np.nan,
                "estimated_transaction_cost": transaction_cost,
                "estimated_financing_cost": financing_cost,
                "warnings": ";".join(warnings),
            }
        )
    frame = pd.DataFrame(rows)
    for column in (
        "paper_execution_return",
        "close_to_close_research_model_return",
        "next_open_to_next_open_model_return",
        "model_return",
    ):
        frame[f"{column}_equity"] = (1.0 + frame[column].fillna(0.0).astype(float)).cumprod()
    frame["cumulative_drift_vs_close_to_close_research_model"] = (
        frame["paper_execution_return_equity"] / frame["close_to_close_research_model_return_equity"] - 1.0
    )
    frame["cumulative_drift_vs_next_open_to_next_open_model"] = (
        frame["paper_execution_return_equity"] / frame["next_open_to_next_open_model_return_equity"] - 1.0
    )
    frame["cumulative_drift_vs_monitor_model"] = (
        frame["paper_execution_return_equity"] / frame["model_return_equity"] - 1.0
    )
    return frame


def _metrics(frame: pd.DataFrame, threshold: float) -> Dict[str, float | bool | str]:
    valid_close = frame["daily_drift_vs_close_to_close_research_model"].dropna()
    valid_open = frame["daily_drift_vs_next_open_to_next_open_model"].dropna()
    cumulative_close = float(frame["cumulative_drift_vs_close_to_close_research_model"].iloc[-1]) if not frame.empty else np.nan
    cumulative_open = float(frame["cumulative_drift_vs_next_open_to_next_open_model"].iloc[-1]) if not frame.empty else np.nan
    worst_daily = float(valid_close.abs().max()) if not valid_close.empty else np.nan
    average_daily = float(valid_close.mean()) if not valid_close.empty else np.nan
    exceeds = bool(
        (not np.isnan(cumulative_close) and abs(cumulative_close) > threshold)
        or (not np.isnan(cumulative_open) and abs(cumulative_open) > threshold)
        or (not np.isnan(worst_daily) and abs(worst_daily) > threshold)
    )
    return {
        "cumulative_drift_vs_close_to_close_research_model": cumulative_close,
        "cumulative_drift_vs_next_open_to_next_open_model": cumulative_open,
        "average_daily_drift": average_daily,
        "worst_daily_drift": worst_daily,
        "average_daily_drift_vs_next_open_to_next_open_model": float(valid_open.mean()) if not valid_open.empty else np.nan,
        "policy_threshold": float(threshold),
        "drift_exceeds_policy_threshold": exceeds,
        "rows_with_warnings": int(frame["warnings"].fillna("").astype(str).ne("").sum()) if not frame.empty else 0,
    }


def _write_model_chart(frame: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "model_vs_execution_equity.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot = frame.copy()
    plot["signal_date"] = pd.to_datetime(plot["signal_date"])
    for column, label in (
        ("paper_execution_return_equity", "Paper execution"),
        ("close_to_close_research_model_return_equity", "Close-to-close research model"),
        ("next_open_to_next_open_model_return_equity", "Next-open-to-next-open model"),
        ("model_return_equity", "Monitor model equity"),
    ):
        ax.plot(plot["signal_date"], plot[column].astype(float), label=label)
    ax.set_title("VolTarget Model vs Observable Execution Equity")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_daily_drift_chart(frame: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "daily_drift.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    plot = frame.copy()
    plot["signal_date"] = pd.to_datetime(plot["signal_date"])
    ax.plot(
        plot["signal_date"],
        plot["daily_drift_vs_close_to_close_research_model"].astype(float),
        label="Daily drift vs close-to-close model",
    )
    ax.plot(
        plot["signal_date"],
        plot["daily_drift_vs_next_open_to_next_open_model"].astype(float),
        label="Daily drift vs next-open-to-next-open model",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("VolTarget Daily Execution Drift")
    ax.set_ylabel("Daily drift")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_report(path: Path, metrics: Dict[str, float | bool | str], chart_paths: Dict[str, Path], summary_path: Path) -> None:
    lines = [
        "# VolTarget Execution Drift Report",
        "",
        "**paper trading only**",
        "",
        "This report compares paper-monitor model assumptions against observable TQQQ prices. It does not place trades or create orders.",
        "",
        "## Summary",
        f"- Cumulative drift vs close-to-close research model: `{metrics['cumulative_drift_vs_close_to_close_research_model']}`",
        f"- Cumulative drift vs next-open-to-next-open model: `{metrics['cumulative_drift_vs_next_open_to_next_open_model']}`",
        f"- Average daily drift: `{metrics['average_daily_drift']}`",
        f"- Worst daily drift: `{metrics['worst_daily_drift']}`",
        f"- Policy threshold: `{metrics['policy_threshold']}`",
        f"- Drift exceeds policy threshold: `{metrics['drift_exceeds_policy_threshold']}`",
        f"- Rows with warnings: `{metrics['rows_with_warnings']}`",
        "",
        "## Outputs",
        f"- Execution drift summary: `{summary_path}`",
        f"- Model vs execution equity chart: `{chart_paths['model_vs_execution_equity']}`",
        f"- Daily drift chart: `{chart_paths['daily_drift']}`",
        "",
        "## Warning Notes",
        "- Missing open prices are recorded as warnings and excluded from return/drift calculations for the affected row.",
        "- The open-to-open execution row for a signal uses the next session open and the following session open; it does not use later prices.",
        "",
        "## Boundary",
        "Execution drift tracking is audit-only. It does not change strategy logic, connect to a broker, or enable auto-trading.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_execution_drift_tracking(
    *,
    config_path: Path,
    signal_dir: Path,
    output_dir: Path,
    data_csv: Optional[Path] = None,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> ExecutionDriftResult:
    config = load_monitor_config(Path(config_path))
    signal_dir = Path(signal_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals = _read_signal_history(signal_dir)
    symbol = str(signals["target_symbol"].dropna().iloc[-1]).upper()
    start_date = pd.Timestamp(signals["signal_date"].min()).date().isoformat()
    if data_csv is not None:
        prices = pd.read_csv(data_csv, parse_dates=["Date"]).set_index("Date").sort_index()
    else:
        benchmark_symbol = str(config.get("benchmark_symbol", "")).upper()
        symbols = [symbol]
        if benchmark_symbol and benchmark_symbol != symbol:
            symbols.append(benchmark_symbol)
        elif symbol == "TQQQ":
            symbols.append("QQQ")
        prices = load_prices(
            start=start_date,
            symbols=tuple(symbols),
            cache_dir=str(config.get("cache_dir", "./price_cache")),
            include_ohlc=True,
            use_csv_if_exists=True,
            dropna=False,
            allow_missing_symbols=False,
        )
    summary = _build_drift_frame(signals, prices, symbol)
    metrics = _metrics(summary, float(drift_threshold))
    summary_path = output_dir / "execution_drift_summary.csv"
    report_path = output_dir / "execution_drift_report.md"
    summary.to_csv(summary_path, index=False)
    chart_paths = {
        "model_vs_execution_equity": _write_model_chart(summary, output_dir),
        "daily_drift": _write_daily_drift_chart(summary, output_dir),
    }
    _write_report(report_path, metrics, chart_paths, summary_path)
    return ExecutionDriftResult(
        output_dir=output_dir,
        summary_path=summary_path,
        report_path=report_path,
        chart_paths=chart_paths,
        summary=summary,
        metrics=metrics,
    )
