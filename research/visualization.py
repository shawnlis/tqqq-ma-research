from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import annualized_return


EXPOSURE_COLUMNS = (
    "target_exposure",
    "position",
    "trade_weight",
    "tqqq_weight",
    "weight",
    "exposure",
)


@dataclass(frozen=True)
class VisualizationResult:
    summary: pd.DataFrame
    chart_paths: Dict[str, Path]
    report_path: Path
    summary_path: Path


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _as_float(value: object, default: float = np.nan) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_drawdown(equity: pd.Series) -> pd.Series:
    equity = pd.Series(equity, copy=True).astype(float)
    if equity.empty:
        return equity
    return equity / equity.cummax() - 1.0


def _normalize_to_first_common(strategy_equity: pd.Series, benchmark_equity: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "strategy_equity": pd.Series(strategy_equity, copy=True).astype(float),
            "benchmark_equity": pd.Series(benchmark_equity, copy=True).astype(float),
        }
    ).dropna()
    if frame.empty:
        raise ValueError("No common strategy and benchmark equity dates are available.")
    first = frame.iloc[0]
    if first["strategy_equity"] == 0.0 or first["benchmark_equity"] == 0.0:
        raise ValueError("Cannot normalize equity curves from a zero first equity value.")
    frame["strategy_equity"] = frame["strategy_equity"] / first["strategy_equity"]
    frame["benchmark_equity"] = frame["benchmark_equity"] / first["benchmark_equity"]
    return frame


def prepare_equity_curves(stitched: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in stitched.columns:
        raise ValueError("stitched_equity.csv must include a Date column.")
    if "daily_ret_tqqq" not in stitched.columns:
        raise ValueError("stitched_equity.csv must include daily_ret_tqqq for same-period TQQQ.")

    data = stitched.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").set_index("Date")

    if "equity" in data.columns:
        strategy_equity = data["equity"].astype(float)
    elif "ret" in data.columns:
        strategy_equity = (1.0 + data["ret"].astype(float).fillna(0.0)).cumprod()
    else:
        raise ValueError("stitched_equity.csv must include equity or ret.")

    benchmark_equity = (1.0 + data["daily_ret_tqqq"].astype(float).fillna(0.0)).cumprod()
    curves = _normalize_to_first_common(strategy_equity, benchmark_equity)
    curves["strategy_drawdown"] = calculate_drawdown(curves["strategy_equity"])
    curves["benchmark_drawdown"] = calculate_drawdown(curves["benchmark_equity"])
    curves["relative_equity"] = curves["strategy_equity"] / curves["benchmark_equity"]
    return curves


def locate_stage3_candidate(input_dir: Path) -> pd.Series:
    input_dir = Path(input_dir)
    decision_path = input_dir / "candidate_decision_table.csv"
    summary_path = input_dir / "voltarget_stage3_summary.csv"
    robust_path = input_dir / "robust_alpha_audit.csv"

    family = "VolTargetTQQQStrategy"
    category = "tqqq_voltarget"

    if decision_path.exists():
        decision = pd.read_csv(decision_path)
        rows = decision[
            (decision.get("family", pd.Series(dtype=object)).astype(str) == family)
            | (decision.get("category", pd.Series(dtype=object)).astype(str) == category)
        ]
        primary = rows[rows.get("is_primary_candidate", pd.Series(dtype=object)).map(_as_bool)] if not rows.empty else rows
        selected = primary.iloc[0] if not primary.empty else rows.iloc[0] if not rows.empty else None
        if selected is not None:
            family = str(selected.get("family", family))
            category = str(selected.get("category", category))

    for path in (summary_path, robust_path):
        if not path.exists():
            continue
        table = pd.read_csv(path)
        rows = table[
            (table.get("family", pd.Series(dtype=object)).astype(str) == family)
            | (table.get("category", pd.Series(dtype=object)).astype(str) == category)
        ]
        primary = rows[rows.get("is_primary_candidate", pd.Series(dtype=object)).map(_as_bool)] if not rows.empty else rows
        if not primary.empty:
            return primary.iloc[0]
        if not rows.empty:
            return rows.iloc[0]

    raise FileNotFoundError(
        "Could not locate a plain VolTargetTQQQStrategy candidate from "
        "candidate_decision_table.csv, voltarget_stage3_summary.csv, or robust_alpha_audit.csv."
    )


def _resolve_candidate_output_dir(input_dir: Path, candidate: pd.Series) -> Path:
    output_value = str(candidate.get("baseline_output_dir", "")).strip()
    if not output_value:
        raise ValueError("Selected candidate does not include baseline_output_dir.")
    output_dir = Path(output_value)
    if not output_dir.is_absolute():
        parts = output_dir.parts
        if parts and parts[0].lower() == "outputs":
            output_dir = Path(output_value)
        else:
            output_dir = input_dir / output_dir
    if not output_dir.exists():
        raise FileNotFoundError(f"Candidate output directory does not exist: {output_dir}")
    return output_dir


def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def _find_exposure_column(stitched: pd.DataFrame) -> Optional[str]:
    for column in EXPOSURE_COLUMNS:
        if column in stitched.columns:
            return column
    return None


def _save_equity_chart(curves: pd.DataFrame, output_path: Path, *, log_scale: bool, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(curves.index, curves["strategy_equity"], label="VolTargetTQQQStrategy", linewidth=1.8)
    ax.plot(curves.index, curves["benchmark_equity"], label="TQQQ buy-and-hold", linewidth=1.6)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Normalized equity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_drawdown_chart(curves: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(curves.index, curves["strategy_drawdown"], label="VolTargetTQQQStrategy", linewidth=1.6)
    ax.plot(curves.index, curves["benchmark_drawdown"], label="TQQQ buy-and-hold", linewidth=1.6)
    ax.set_title("Drawdown: VolTargetTQQQStrategy vs TQQQ buy-and-hold")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_relative_chart(curves: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(curves.index, curves["relative_equity"], label="VolTarget / TQQQ", linewidth=1.7)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Parity")
    ax.set_title("Relative equity: VolTargetTQQQStrategy divided by TQQQ buy-and-hold")
    ax.set_ylabel("Relative equity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_yearly_chart(yearly: pd.DataFrame, output_path: Path) -> None:
    required = {"year", "strategy_return", "benchmark_return"}
    if not required.issubset(yearly.columns):
        raise ValueError("yearly_returns.csv must include year, strategy_return, and benchmark_return.")
    data = yearly.copy()
    data["relative_difference"] = data["strategy_return"].astype(float) - data["benchmark_return"].astype(float)
    x = np.arange(len(data))
    width = 0.28
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, data["strategy_return"].astype(float), width, label="VolTargetTQQQStrategy")
    ax.bar(x, data["benchmark_return"].astype(float), width, label="TQQQ buy-and-hold")
    ax.bar(x + width, data["relative_difference"], width, label="Relative difference")
    ax.set_xticks(x)
    ax.set_xticklabels(data["year"].astype(str), rotation=45)
    ax.set_title("Yearly return comparison")
    ax.set_ylabel("Calendar-year return")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_exposure_chart(stitched: pd.DataFrame, column: str, output_path: Path) -> None:
    data = stitched.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(data["Date"], data[column].astype(float), label=column, linewidth=1.4)
    ax.set_title(f"VolTarget exposure over time ({column})")
    ax.set_ylabel("Exposure")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_report(
    report_path: Path,
    *,
    chart_paths: Dict[str, Path],
    summary_row: Dict[str, object],
    exposure_note: str,
) -> None:
    chart_lines = "\n".join(f"- `{name}`: `{path}`" for name, path in chart_paths.items())
    report = f"""# VolTarget Stage 3 Equity Visualization

This visualization is an empirical review artifact. It is not an investment recommendation, not proven persistent alpha, and not production-ready.

## Final Classification

```text
VolTargetTQQQStrategy = {summary_row["classification"]}
Recommendation = {summary_row["recommendation"]}
```

## Summary Metrics

- Start date: {summary_row["start_date"]}
- End date: {summary_row["end_date"]}
- Strategy final equity: {summary_row["strategy_final_equity"]}
- TQQQ final equity: {summary_row["tqqq_final_equity"]}
- Final equity ratio: {summary_row["final_equity_ratio"]}
- Strategy CAGR: {summary_row["strategy_cagr"]}
- TQQQ CAGR: {summary_row["tqqq_cagr"]}
- Strategy max drawdown: {summary_row["strategy_max_drawdown"]}
- TQQQ max drawdown: {summary_row["tqqq_max_drawdown"]}

## Charts

{chart_lines}

## Warnings

- One-year dominance warning: {summary_row["one_year_dominance_warning"]}
- Stage 3 identified a dominant contribution year; review 2022 explicitly because prior gates treated 2022 concentration as a core risk.
- This is classified as a crash-control / risk-budget overlay, not proven persistent alpha.
- This is not production-ready.
- Full tournament and deep VolTarget tournament remain deferred.

## Exposure

{exposure_note}
"""
    report_path.write_text(report, encoding="utf-8")


def generate_equity_visualizations(input_dir: Path, output_dir: Path) -> VisualizationResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = locate_stage3_candidate(input_dir)
    candidate_output_dir = _resolve_candidate_output_dir(input_dir, candidate)

    stitched_path = candidate_output_dir / "stitched_equity.csv"
    if not stitched_path.exists():
        raise FileNotFoundError(f"Missing stitched equity file: {stitched_path}")
    benchmark_summary_path = candidate_output_dir / "same_period_benchmark_summary.csv"
    if not benchmark_summary_path.exists():
        raise FileNotFoundError(f"Missing benchmark summary file: {benchmark_summary_path}")

    stitched = pd.read_csv(stitched_path)
    benchmark_summary = pd.read_csv(benchmark_summary_path)
    yearly = _read_optional_csv(candidate_output_dir / "yearly_returns.csv")
    run_config_path = candidate_output_dir / "run_config.json"

    curves = prepare_equity_curves(stitched)
    strategy_final = float(curves["strategy_equity"].iloc[-1])
    tqqq_final = float(curves["benchmark_equity"].iloc[-1])
    final_ratio = strategy_final / tqqq_final if tqqq_final != 0.0 else np.nan
    strategy_cagr = annualized_return(curves["strategy_equity"])
    tqqq_cagr = annualized_return(curves["benchmark_equity"])
    strategy_max_dd = float(curves["strategy_drawdown"].min())
    tqqq_max_dd = float(curves["benchmark_drawdown"].min())

    title_suffix = f"final equity: {strategy_final:.3f} vs {tqqq_final:.3f}"
    chart_paths: Dict[str, Path] = {}
    chart_paths["equity_curve_linear"] = output_dir / "equity_curve_linear.png"
    _save_equity_chart(
        curves,
        chart_paths["equity_curve_linear"],
        log_scale=False,
        title=f"VolTargetTQQQStrategy vs TQQQ buy-and-hold\n{title_suffix}",
    )
    chart_paths["equity_curve_log"] = output_dir / "equity_curve_log.png"
    _save_equity_chart(
        curves,
        chart_paths["equity_curve_log"],
        log_scale=True,
        title=f"VolTargetTQQQStrategy vs TQQQ buy-and-hold (log scale)\n{title_suffix}",
    )
    chart_paths["drawdown_curve"] = output_dir / "drawdown_curve.png"
    _save_drawdown_chart(curves, chart_paths["drawdown_curve"])
    chart_paths["relative_equity_curve"] = output_dir / "relative_equity_curve.png"
    _save_relative_chart(curves, chart_paths["relative_equity_curve"])

    if yearly is not None:
        chart_paths["yearly_return_comparison"] = output_dir / "yearly_return_comparison.png"
        _save_yearly_chart(yearly, chart_paths["yearly_return_comparison"])

    exposure_column = _find_exposure_column(stitched)
    exposure_note = "Exposure column unavailable; exposure chart was skipped."
    if exposure_column is not None:
        chart_paths["exposure_over_time"] = output_dir / "exposure_over_time.png"
        _save_exposure_chart(stitched, exposure_column, chart_paths["exposure_over_time"])
        exposure_note = f"Exposure chart written using `{exposure_column}`."

    one_year_share = _as_float(candidate.get("one_year_dominance_share"))
    dominant_year = str(candidate.get("dominant_year", "")).strip()
    one_year_warning = (
        f"dominant year {dominant_year} contributed {one_year_share:.3f} of audited contribution"
        if not np.isnan(one_year_share)
        else "one-year dominance data unavailable"
    )

    summary_row = {
        "strategy_name": str(candidate.get("family", "VolTargetTQQQStrategy")),
        "classification": str(candidate.get("classification", "crash_control_candidate")),
        "recommendation": str(candidate.get("run_deep_tournament_recommendation", "defer_deep_tournament")),
        "candidate_output_dir": str(candidate_output_dir),
        "stitched_equity_path": str(stitched_path),
        "same_period_benchmark_summary_path": str(benchmark_summary_path),
        "yearly_returns_path": str(candidate_output_dir / "yearly_returns.csv") if yearly is not None else "",
        "run_config_path": str(run_config_path) if run_config_path.exists() else "",
        "start_date": curves.index.min().date().isoformat(),
        "end_date": curves.index.max().date().isoformat(),
        "strategy_final_equity": strategy_final,
        "tqqq_final_equity": tqqq_final,
        "final_equity_ratio": final_ratio,
        "strategy_cagr": strategy_cagr,
        "tqqq_cagr": tqqq_cagr,
        "strategy_max_drawdown": strategy_max_dd,
        "tqqq_max_drawdown": tqqq_max_dd,
        "stage3_full_period_ratio_vs_tqqq": _as_float(candidate.get("full_period_ratio_vs_tqqq")),
        "stage3_ex_2022_ratio_vs_tqqq": _as_float(candidate.get("ex_2022_ratio_vs_tqqq")),
        "stage3_ex_2022_ratio_vs_simple_voltarget_no_trend": _as_float(
            candidate.get("ex_2022_ratio_vs_simple_voltarget_no_trend")
        ),
        "one_year_dominance_share": one_year_share,
        "one_year_dominance_warning": one_year_warning,
        "exposure_column": exposure_column or "",
        "exposure_status": "written" if exposure_column else "exposure column unavailable",
        "benchmark_symbol": str(benchmark_summary.iloc[0].get("benchmark_symbol", "TQQQ"))
        if not benchmark_summary.empty
        else "TQQQ",
        "warning": "not proven persistent alpha; not production-ready",
    }

    summary = pd.DataFrame([summary_row])
    summary_path = output_dir / "visualization_summary.csv"
    summary.to_csv(summary_path, index=False)

    report_path = output_dir / "visualization_report.md"
    _write_report(report_path, chart_paths=chart_paths, summary_row=summary_row, exposure_note=exposure_note)

    return VisualizationResult(
        summary=summary,
        chart_paths=chart_paths,
        report_path=report_path,
        summary_path=summary_path,
    )
