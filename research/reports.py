from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_objective_scores,
    max_drawdown,
    sharpe_ratio,
)


def summarize_stitched(stitched_df: pd.DataFrame) -> Dict[str, float]:
    if stitched_df.empty:
        return {}

    cagr = annualized_return(stitched_df["equity"])
    vol = annualized_volatility(stitched_df["ret"])
    sharpe = sharpe_ratio(stitched_df["ret"])
    mdd = max_drawdown(stitched_df["equity"])
    calmar = calmar_ratio(cagr, mdd)

    return {
        "wf_final_equity": float(stitched_df["equity"].iloc[-1]),
        "wf_cagr": cagr,
        "wf_vol": vol,
        "wf_sharpe": sharpe,
        "wf_max_dd": mdd,
        "wf_calmar": calmar,
        "wf_avg_exposure": float(stitched_df["position"].mean()),
        "wf_trades": int((stitched_df["turnover"] > 0).sum()),
    }


def summarize_param_stability(
    results: pd.DataFrame,
    metric: str = "sharpe",
    short_radius: int = 3,
    long_radius: int = 15,
    score_mode: str = "difference",   # "difference" or "ratio"
) -> pd.DataFrame:
    """
    Vectorized neighborhood stability analysis.

    For each parameter set, evaluate its local neighborhood within the same
    strategy family:
      short in [short-short_radius, short+short_radius]
      long  in [long-long_radius, long+long_radius]

    score_mode:
      - "difference": robustness = neigh_mean - neigh_std
      - "ratio":      robustness = neigh_mean / (neigh_std + 1e-9)

    Note:
      robustness_score is metric-specific and not comparable across different metrics.
    """
    if results.empty:
        return pd.DataFrame()

    family_cols = [
        "ma_type",
        "signal_asset",
        "threshold",
        "cooldown_days",
        "qqq_filter",
        "qqq_filter_window",
    ]
    required_cols = set(family_cols + ["short", "long", metric])
    missing = required_cols - set(results.columns)
    if missing:
        raise ValueError(f"Missing columns for stability analysis: {missing}")

    left = results[family_cols + ["short", "long", metric]].copy()
    right = left.rename(
        columns={
            "short": "short_r",
            "long": "long_r",
            metric: f"{metric}_r",
        }
    )

    merged = left.merge(right, on=family_cols, how="inner")

    in_hood = (
        (merged["short_r"] - merged["short"]).abs() <= short_radius
    ) & (
        (merged["long_r"] - merged["long"]).abs() <= long_radius
    )

    merged = merged.loc[in_hood].copy()
    if merged.empty:
        return pd.DataFrame()

    # Avoid named-agg + lambda compatibility risk
    merged[f"{metric}_r_sq"] = merged[f"{metric}_r"] ** 2

    group_cols = family_cols + ["short", "long"]

    agg = (
        merged.groupby(group_cols, as_index=False)
        .agg(
            neigh_mean=(f"{metric}_r", "mean"),
            neigh_min=(f"{metric}_r", "min"),
            neigh_max=(f"{metric}_r", "max"),
            neigh_count=(f"{metric}_r", "count"),
            neigh_sq_mean=(f"{metric}_r_sq", "mean"),
        )
    )

    # ddof=0 std = sqrt(E[x^2] - E[x]^2)
    agg["neigh_std"] = np.sqrt(
        np.maximum(agg["neigh_sq_mean"] - agg["neigh_mean"] ** 2, 0.0)
    )
    agg = agg.drop(columns=["neigh_sq_mean"])

    out = left.merge(agg, on=group_cols, how="left")
    out = out.rename(columns={metric: f"{metric}_point"})
    out["neigh_std"] = out["neigh_std"].fillna(0.0)

    if score_mode == "difference":
        out["robustness_score"] = out["neigh_mean"] - out["neigh_std"]
    elif score_mode == "ratio":
        out["robustness_score"] = out["neigh_mean"] / (out["neigh_std"] + 1e-9)
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    return out.sort_values(
        ["robustness_score", "neigh_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)


def make_heatmap_table(
    results: pd.DataFrame,
    metric: str = "sharpe",
    ma_type: str = "ema",
    signal_asset: str = "QQQ",
    threshold: float = 0.0,
    cooldown_days: int = 0,
    qqq_filter: bool = True,
    qqq_filter_window: int = 200,
) -> pd.DataFrame:
    subset = results[
        (results["ma_type"] == ma_type)
        & (results["signal_asset"] == signal_asset)
        & (np.isclose(results["threshold"], threshold))
        & (results["cooldown_days"] == cooldown_days)
        & (results["qqq_filter"] == int(qqq_filter))
        & (results["qqq_filter_window"] == qqq_filter_window)
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    pivot = subset.pivot_table(
        index="short",
        columns="long",
        values=metric,
        aggfunc="mean",
    )
    return pivot.sort_index().sort_index(axis=1)


def compare_to_benchmark(
    stitched_df: pd.DataFrame,
    price_data: pd.DataFrame,
    benchmark_symbol: str = "TQQQ",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if stitched_df.empty:
        raise ValueError("stitched_df is empty; no benchmark comparison can be computed.")
    if "ret" not in stitched_df.columns:
        raise ValueError("stitched_df must contain a ret column.")
    if benchmark_symbol not in price_data.columns:
        raise ValueError(f"price_data is missing benchmark column: {benchmark_symbol}")
    if stitched_df.index.has_duplicates:
        raise ValueError("stitched_df index contains duplicate dates.")

    stitched = stitched_df.sort_index().copy()
    missing_dates = stitched.index.difference(price_data.index)
    if len(missing_dates) > 0:
        first_missing = missing_dates[0]
        raise ValueError(f"price_data is missing stitched benchmark date: {first_missing}")

    benchmark_prices = price_data.loc[stitched.index, benchmark_symbol].astype(float)
    if benchmark_prices.isna().any():
        raise ValueError("Benchmark prices contain missing values on stitched dates.")

    strategy_ret = stitched["ret"].astype(float)
    strategy_equity = (
        stitched["equity"].astype(float)
        if "equity" in stitched.columns
        else (1.0 + strategy_ret).cumprod()
    )

    benchmark_ret = benchmark_prices.pct_change().fillna(0.0)
    benchmark_equity = (1.0 + benchmark_ret).cumprod()

    strategy_final_equity = float(strategy_equity.iloc[-1])
    benchmark_final_equity = float(benchmark_equity.iloc[-1])
    strategy_cagr = annualized_return(strategy_equity)
    benchmark_cagr = annualized_return(benchmark_equity)
    strategy_sharpe = sharpe_ratio(strategy_ret)
    benchmark_sharpe = sharpe_ratio(benchmark_ret)
    strategy_max_dd = max_drawdown(strategy_equity)
    benchmark_max_dd = max_drawdown(benchmark_equity)
    strategy_calmar = calmar_ratio(strategy_cagr, strategy_max_dd)
    benchmark_calmar = calmar_ratio(benchmark_cagr, benchmark_max_dd)
    final_equity_ratio = (
        strategy_final_equity / benchmark_final_equity
        if benchmark_final_equity != 0.0
        else np.nan
    )
    excess_cagr = (
        strategy_cagr - benchmark_cagr
        if not (np.isnan(strategy_cagr) or np.isnan(benchmark_cagr))
        else np.nan
    )

    yearly = pd.DataFrame(
        {
            "strategy_return": strategy_ret,
            "benchmark_return": benchmark_ret,
        },
        index=stitched.index,
    )
    yearly_returns = yearly.groupby(yearly.index.year).agg(
        lambda returns: float((1.0 + returns).prod() - 1.0)
    )
    yearly_returns.index.name = "year"
    yearly_returns = yearly_returns.reset_index()
    yearly_returns["strategy_beats_benchmark"] = (
        yearly_returns["strategy_return"] > yearly_returns["benchmark_return"]
    )

    years_strategy_beats_benchmark = int(yearly_returns["strategy_beats_benchmark"].sum())
    total_years = int(len(yearly_returns))
    turnover = stitched["turnover"].astype(float) if "turnover" in stitched.columns else None
    objective_scores = compute_objective_scores(
        strategy_ret=strategy_ret,
        benchmark_ret=benchmark_ret,
        strategy_equity=strategy_equity,
        benchmark_equity=benchmark_equity,
        turnover=turnover,
    )

    summary = pd.DataFrame(
        [
            {
                "benchmark_symbol": benchmark_symbol,
                "start_date": stitched.index[0].date().isoformat(),
                "end_date": stitched.index[-1].date().isoformat(),
                "stitched_rows": int(len(stitched)),
                "strategy_final_equity": strategy_final_equity,
                "benchmark_final_equity": benchmark_final_equity,
                "strategy_cagr": strategy_cagr,
                "benchmark_cagr": benchmark_cagr,
                "strategy_sharpe": strategy_sharpe,
                "benchmark_sharpe": benchmark_sharpe,
                "strategy_max_dd": strategy_max_dd,
                "benchmark_max_dd": benchmark_max_dd,
                "strategy_calmar": strategy_calmar,
                "benchmark_calmar": benchmark_calmar,
                "final_equity_ratio": final_equity_ratio,
                "excess_cagr": excess_cagr,
                "years_strategy_beats_benchmark": years_strategy_beats_benchmark,
                "total_years": total_years,
                "raw_outperformance_pass": bool(final_equity_ratio > 1.0),
                **objective_scores,
            }
        ]
    )

    return summary, yearly_returns


def save_benchmark_comparison(
    out_dir: Path,
    summary: pd.DataFrame,
    yearly_returns: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "same_period_benchmark_summary.csv", index=False)
    yearly_returns.to_csv(out_dir / "same_period_yearly_returns.csv", index=False)


def compare_real_to_synthetic(
    real_prices: pd.Series,
    synthetic_prices: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aligned = pd.concat(
        [
            pd.Series(real_prices, copy=True).rename("real_price"),
            pd.Series(synthetic_prices, copy=True).rename("synthetic_price"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        raise ValueError("No overlapping real/synthetic price dates available.")

    real_return = aligned["real_price"].pct_change().fillna(0.0)
    synthetic_return = aligned["synthetic_price"].pct_change().fillna(0.0)
    real_equity = (1.0 + real_return).cumprod()
    synthetic_equity = (1.0 + synthetic_return).cumprod()

    real_cagr = annualized_return(real_equity)
    synthetic_cagr = annualized_return(synthetic_equity)
    real_max_dd = max_drawdown(real_equity)
    synthetic_max_dd = max_drawdown(synthetic_equity)
    tracking_error = annualized_volatility(synthetic_return - real_return)

    daily = pd.DataFrame(
        {
            "real_return": real_return,
            "synthetic_return": synthetic_return,
            "return_difference": synthetic_return - real_return,
            "real_equity": real_equity,
            "synthetic_equity": synthetic_equity,
        },
        index=aligned.index,
    )

    summary = pd.DataFrame(
        [
            {
                "start_date": aligned.index[0].date().isoformat(),
                "end_date": aligned.index[-1].date().isoformat(),
                "overlap_rows": int(len(aligned)),
                "real_final_equity": float(real_equity.iloc[-1]),
                "synthetic_final_equity": float(synthetic_equity.iloc[-1]),
                "tracking_error": tracking_error,
                "real_cagr": real_cagr,
                "synthetic_cagr": synthetic_cagr,
                "cagr_difference": synthetic_cagr - real_cagr
                if not (np.isnan(synthetic_cagr) or np.isnan(real_cagr))
                else np.nan,
                "real_max_dd": real_max_dd,
                "synthetic_max_dd": synthetic_max_dd,
                "max_drawdown_difference": synthetic_max_dd - real_max_dd
                if not (np.isnan(synthetic_max_dd) or np.isnan(real_max_dd))
                else np.nan,
            }
        ]
    )

    return summary, daily


def save_synthetic_tracking(
    out_dir: Path,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "synthetic_tracking_summary.csv", index=False)
    daily.to_csv(out_dir / "synthetic_tracking_equity.csv")


def print_benchmark_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    row = summary.iloc[0]
    print("\n=== Same-period benchmark ===")
    print(f"STRATEGY FINAL: {row['strategy_final_equity']:.6f}")
    print(f"BENCHMARK FINAL: {row['benchmark_final_equity']:.6f}")
    print(f"STRATEGY / BENCHMARK: {row['final_equity_ratio']:.6f}")
    print(f"RAW OUTPERFORMANCE PASS: {bool(row['raw_outperformance_pass'])}")



def _json_default(value: Any) -> Any:
    if isinstance(value, range):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_run_config(out_dir: Path, config: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **config,
    }
    path = out_dir / "run_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return path


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Any, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value) * 100.0:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def _yes_no(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return bool(value)


def _plot_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_chart(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _placeholder_chart(output_dir: Path, filename: str, title: str, message: str = "No data available") -> Path:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_axis_off()
    path = output_dir / "charts" / filename
    _save_chart(fig, path)
    return path


def _markdown_table(df: pd.DataFrame, columns: Optional[Iterable[str]] = None, max_rows: int = 10) -> str:
    if df.empty:
        return "_No data available._"
    out = df.copy()
    if columns is not None:
        out = out[[column for column in columns if column in out.columns]]
    if out.empty:
        return "_No requested columns available._"
    out = out.head(max_rows)

    def cell(value: Any) -> str:
        if pd.isna(value):
            text = ""
        elif isinstance(value, float):
            text = _fmt(value, 6)
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    headers = [str(column).replace("|", "\\|") for column in out.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(cell(row[column]) for column in out.columns) + " |")
    return "\n".join(lines)


def _read_stitched_equity(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "stitched_equity.csv"
    if not path.exists():
        return pd.DataFrame()
    stitched = pd.read_csv(path, index_col=0, parse_dates=True)
    stitched.index.name = "Date"
    return stitched


def _benchmark_equity_for_stitched(
    stitched: pd.DataFrame,
    price_data: Optional[pd.DataFrame],
    benchmark_symbol: str,
) -> pd.Series:
    if stitched.empty or price_data is None or benchmark_symbol not in price_data.columns:
        return pd.Series(dtype=float)
    benchmark_prices = price_data[benchmark_symbol].reindex(stitched.index).astype(float)
    if benchmark_prices.isna().any():
        return pd.Series(dtype=float)
    benchmark_ret = benchmark_prices.pct_change().fillna(0.0)
    benchmark_equity = (1.0 + benchmark_ret).cumprod()
    benchmark_equity.name = f"{benchmark_symbol}_equity"
    return benchmark_equity


def _chart_equity_vs_benchmark(
    output_dir: Path,
    stitched: pd.DataFrame,
    benchmark_equity: pd.Series,
) -> Optional[Path]:
    if stitched.empty or "equity" not in stitched.columns:
        return _placeholder_chart(output_dir, "equity_vs_benchmark_log.png", "Strategy Equity vs Benchmark")
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    stitched["equity"].astype(float).plot(ax=ax, label="Strategy")
    if not benchmark_equity.empty:
        benchmark_equity.plot(ax=ax, label=benchmark_equity.name.replace("_equity", ""))
    ax.set_yscale("log")
    ax.set_title("Strategy Equity vs Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity, log scale")
    ax.legend()
    path = output_dir / "charts" / "equity_vs_benchmark_log.png"
    _save_chart(fig, path)
    return path


def _chart_drawdown(
    output_dir: Path,
    stitched: pd.DataFrame,
    benchmark_equity: pd.Series,
) -> Optional[Path]:
    if stitched.empty or "equity" not in stitched.columns:
        return _placeholder_chart(output_dir, "drawdown.png", "Drawdown")
    plt = _plot_setup()
    strategy_equity = stitched["equity"].astype(float)
    strategy_dd = strategy_equity / strategy_equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(9, 4.8))
    strategy_dd.plot(ax=ax, label="Strategy")
    if not benchmark_equity.empty:
        benchmark_dd = benchmark_equity / benchmark_equity.cummax() - 1.0
        benchmark_dd.plot(ax=ax, label=benchmark_equity.name.replace("_equity", ""))
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    path = output_dir / "charts" / "drawdown.png"
    _save_chart(fig, path)
    return path


def _chart_yearly_relative_returns(output_dir: Path, yearly: pd.DataFrame) -> Optional[Path]:
    if yearly.empty or not {"year", "strategy_return", "benchmark_return"}.issubset(yearly.columns):
        return _placeholder_chart(output_dir, "yearly_relative_returns.png", "Yearly Relative Returns")
    data = yearly.copy()
    data["relative_return"] = data["strategy_return"].astype(float) - data["benchmark_return"].astype(float)
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(data["year"].astype(str), data["relative_return"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Yearly Relative Returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Strategy minus benchmark")
    ax.tick_params(axis="x", rotation=45)
    path = output_dir / "charts" / "yearly_relative_returns.png"
    _save_chart(fig, path)
    return path


def _chart_exposure(output_dir: Path, stitched: pd.DataFrame) -> Optional[Path]:
    exposure_cols = [
        column
        for column in [
            "position",
            "trade_weight",
            "tqqq_weight",
            "leveraged_exposure",
            "risk_off_weight",
            "defensive_weight",
            "target_exposure",
        ]
        if column in stitched.columns
    ]
    if stitched.empty or not exposure_cols:
        return _placeholder_chart(output_dir, "exposure_over_time.png", "Exposure Over Time")
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    stitched[exposure_cols].astype(float).plot(ax=ax)
    ax.set_title("Exposure Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Exposure / weight")
    ax.legend()
    path = output_dir / "charts" / "exposure_over_time.png"
    _save_chart(fig, path)
    return path


def _chart_cost_sensitivity(output_dir: Path, costs: pd.DataFrame) -> Optional[Path]:
    if costs.empty or not {"transaction_cost_bps", "final_equity_ratio"}.issubset(costs.columns):
        return _placeholder_chart(output_dir, "cost_sensitivity.png", "Cost Sensitivity")
    data = costs.dropna(subset=["transaction_cost_bps", "final_equity_ratio"]).copy()
    if data.empty:
        return _placeholder_chart(output_dir, "cost_sensitivity.png", "Cost Sensitivity")
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(data["transaction_cost_bps"].astype(str), data["final_equity_ratio"].astype(float))
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("Cost Sensitivity")
    ax.set_xlabel("Transaction cost, bps")
    ax.set_ylabel("Final equity ratio")
    path = output_dir / "charts" / "cost_sensitivity.png"
    _save_chart(fig, path)
    return path


def _chart_validation_methods(output_dir: Path, variants: pd.DataFrame) -> Optional[Path]:
    if variants.empty or not {"variant", "final_equity_ratio"}.issubset(variants.columns):
        return _placeholder_chart(output_dir, "validation_final_equity_ratio.png", "Final Equity Ratio by Validation Method")
    data = variants.dropna(subset=["final_equity_ratio"]).copy()
    if data.empty:
        return _placeholder_chart(output_dir, "validation_final_equity_ratio.png", "Final Equity Ratio by Validation Method")
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(data["variant"].astype(str), data["final_equity_ratio"].astype(float))
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("Final Equity Ratio by Validation Method")
    ax.set_xlabel("Validation method")
    ax.set_ylabel("Final equity ratio")
    ax.tick_params(axis="x", rotation=45)
    path = output_dir / "charts" / "validation_final_equity_ratio.png"
    _save_chart(fig, path)
    return path


def _relative_chart_path(report_path: Path, chart_path: Optional[Path]) -> str:
    if chart_path is None:
        return "_Chart not available for this run._"
    return f"![{chart_path.stem}]({chart_path.relative_to(report_path.parent).as_posix()})"


def _strategy_description(config: Dict[str, Any]) -> str:
    strategy = str(config.get("strategy_name", "unknown"))
    asset_config = config.get("asset_config", {})
    trade_asset = asset_config.get("trade_asset", "TQQQ") if isinstance(asset_config, dict) else "TQQQ"
    signal_asset = asset_config.get("primary_signal_asset", "QQQ") if isinstance(asset_config, dict) else "QQQ"
    description = f"`{strategy}` strategy trading `{trade_asset}` using `{signal_asset}` as the primary signal asset."
    if bool(config.get("use_ensemble_wf", False)):
        description += " Walk-forward selection uses the configured MA ensemble."
    return description


def generate_experiment_report(
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    price_data: Optional[pd.DataFrame] = None,
) -> Path:
    output_dir = Path(output_dir)
    config = dict(config or {})
    if not config and (output_dir / "run_config.json").exists():
        config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))

    stitched = _read_stitched_equity(output_dir)
    summary = _load_csv(output_dir / "same_period_benchmark_summary.csv")
    yearly = _load_csv(output_dir / "yearly_returns.csv")
    wf = _load_csv(output_dir / "walk_forward_windows.csv")
    subperiod = _load_csv(output_dir / "subperiod_summary.csv")
    validation = _load_csv(output_dir / "validation_summary.csv")
    variants = _load_csv(output_dir / "walk_forward_variant_summary.csv")
    costs = _load_csv(output_dir / "cost_sensitivity.csv")
    bootstrap = _load_csv(output_dir / "bootstrap_summary.csv")
    stability = _load_csv(output_dir / "parameter_stability.csv")

    row = summary.iloc[0].to_dict() if not summary.empty else {}
    benchmark_symbol = str(row.get("benchmark_symbol", config.get("benchmark_symbol", "TQQQ"))).upper()
    benchmark_equity = _benchmark_equity_for_stitched(stitched, price_data, benchmark_symbol)

    chart_paths = {
        "equity": _chart_equity_vs_benchmark(output_dir, stitched, benchmark_equity),
        "drawdown": _chart_drawdown(output_dir, stitched, benchmark_equity),
        "yearly": _chart_yearly_relative_returns(output_dir, yearly),
        "exposure": _chart_exposure(output_dir, stitched),
        "cost": _chart_cost_sensitivity(output_dir, costs),
        "validation": _chart_validation_methods(output_dir, variants),
    }

    raw_wealth = _bool_value(row.get("raw_outperformance_pass", False))
    risk_adjusted = bool(
        pd.notna(row.get("strategy_calmar", np.nan))
        and pd.notna(row.get("benchmark_calmar", np.nan))
        and float(row.get("strategy_calmar")) > float(row.get("benchmark_calmar"))
    )
    cost_25 = np.nan
    if not costs.empty and "transaction_cost_bps" in costs.columns:
        match = costs.loc[costs["transaction_cost_bps"].astype(float) == 25.0, "final_equity_ratio"]
        if not match.empty:
            cost_25 = float(match.iloc[0])
    candidate_viable = bool(
        raw_wealth
        and (pd.notna(cost_25) and cost_25 > 1.0)
        and (
            validation.empty
            or _bool_value(validation.iloc[0].get("candidate_viable", False))
        )
    )

    report_summary = pd.DataFrame(
        [
            {
                "experiment_name": config.get("experiment_name", ""),
                "strategy_name": config.get("strategy_name", ""),
                "benchmark_symbol": benchmark_symbol,
                "start_date": row.get("start_date", ""),
                "end_date": row.get("end_date", ""),
                "final_equity_ratio": row.get("final_equity_ratio", np.nan),
                "raw_wealth_beats_benchmark": raw_wealth,
                "better_risk_adjusted_profile": risk_adjusted,
                "candidate_for_further_research": candidate_viable,
                "cost_25_final_equity_ratio": cost_25,
            }
        ]
    )
    report_summary.to_csv(output_dir / "report_summary.csv", index=False)

    report_path = output_dir / "report.md"
    lines = [
        f"# Experiment Report: {config.get('experiment_name', output_dir.name)}",
        "",
        "This report summarizes empirical backtest outputs only. It is not an investment recommendation.",
        "",
        "## 1. Objective",
        f"Evaluate `{config.get('strategy_name', 'unknown')}` against same-period `{benchmark_symbol}` buy-and-hold using the configured walk-forward framework.",
        "",
        "## 2. Strategy description",
        _strategy_description(config),
        "",
        "## 3. Data period",
        f"Start: `{row.get('start_date', config.get('start_date', 'n/a'))}`. End: `{row.get('end_date', config.get('end_date', 'n/a'))}`. Stitched rows: `{row.get('stitched_rows', 'n/a')}`.",
        "",
        "## 4. Benchmark",
        f"Benchmark: `{benchmark_symbol}` over the exact stitched strategy dates.",
        "",
        "## 5. Same-period performance",
        _markdown_table(summary),
        "",
        _relative_chart_path(report_path, chart_paths["equity"]),
        "",
        "## 6. Year-by-year returns",
        _markdown_table(yearly),
        "",
        _relative_chart_path(report_path, chart_paths["yearly"]),
        "",
        "## 7. Drawdowns",
        f"Strategy max drawdown: `{_pct(row.get('strategy_max_dd'))}`. Benchmark max drawdown: `{_pct(row.get('benchmark_max_dd'))}`.",
        "",
        _relative_chart_path(report_path, chart_paths["drawdown"]),
        "",
        "## 8. Exposure over time",
        f"Average exposure: `{_fmt(stitched['position'].mean()) if not stitched.empty and 'position' in stitched.columns else 'n/a'}`.",
        "",
        _relative_chart_path(report_path, chart_paths["exposure"]),
        "",
        "## 9. Turnover and cost sensitivity",
        f"Total turnover: `{_fmt(stitched['turnover'].sum()) if not stitched.empty and 'turnover' in stitched.columns else 'n/a'}`. Total cost: `{_fmt(stitched['cost'].sum()) if not stitched.empty and 'cost' in stitched.columns else 'n/a'}`.",
        "",
        _markdown_table(costs),
        "",
        _relative_chart_path(report_path, chart_paths["cost"]),
        "",
        "## 10. Walk-forward parameter choices",
        _markdown_table(wf, max_rows=20),
        "",
        "## 11. Subperiod results",
        _markdown_table(subperiod),
        "",
        "## 12. Robustness checks",
        "Validation summary:",
        "",
        _markdown_table(validation),
        "",
        "Bootstrap summary:",
        "",
        _markdown_table(bootstrap),
        "",
        "Parameter stability:",
        "",
        _markdown_table(stability),
        "",
        _relative_chart_path(report_path, chart_paths["validation"]),
        "",
        "## 13. Verdict",
        f"- Beats {benchmark_symbol} raw wealth? {_yes_no(raw_wealth)}.",
        f"- Better risk-adjusted profile? {_yes_no(risk_adjusted)}.",
        f"- Candidate for further research? {_yes_no(candidate_viable)}.",
        "",
        "Limitations: results are historical, parameter-selected, sensitive to execution and cost assumptions, and may not persist out of sample. No investment recommendation is made.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _bar_chart(
    output_dir: Path,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    filename: str,
    hline: Optional[float] = None,
) -> Optional[Path]:
    if data.empty or not {x_col, y_col}.issubset(data.columns):
        return _placeholder_chart(output_dir, filename, title)
    plot_data = data.dropna(subset=[y_col]).copy()
    if plot_data.empty:
        return _placeholder_chart(output_dir, filename, title)
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_data[x_col].astype(str), plot_data[y_col].astype(float))
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=45)
    path = output_dir / "charts" / filename
    _save_chart(fig, path)
    return path


def _chart_tournament_validation_methods(output_dir: Path, variants: pd.DataFrame) -> Optional[Path]:
    if variants.empty or not {"walk_forward_variant", "final_equity_ratio"}.issubset(variants.columns):
        return None
    data = (
        variants[variants["status"] == "ok"]
        .groupby("walk_forward_variant", as_index=False)["final_equity_ratio"]
        .median()
    )
    return _bar_chart(
        output_dir,
        data,
        "walk_forward_variant",
        "final_equity_ratio",
        "Median Final Equity Ratio by Validation Method",
        "validation_final_equity_ratio.png",
        hline=1.0,
    )


def _chart_tournament_cost(output_dir: Path, variants: pd.DataFrame) -> Optional[Path]:
    if variants.empty or not {"transaction_cost_bps", "final_equity_ratio"}.issubset(variants.columns):
        return None
    data = (
        variants[variants["status"] == "ok"]
        .groupby("transaction_cost_bps", as_index=False)["final_equity_ratio"]
        .median()
    )
    return _bar_chart(
        output_dir,
        data,
        "transaction_cost_bps",
        "final_equity_ratio",
        "Median Cost Sensitivity",
        "cost_sensitivity.png",
        hline=1.0,
    )


def generate_tournament_report(output_dir: Path, config: Optional[Dict[str, Any]] = None) -> Path:
    output_dir = Path(output_dir)
    config = dict(config or {})
    summary = _load_csv(output_dir / "tournament_summary.csv")
    variants = _load_csv(output_dir / "tournament_variant_results.csv")
    failures = _load_csv(output_dir / "tournament_failures.csv")

    chart_paths = {
        "equity": _bar_chart(
            output_dir,
            summary,
            "family",
            "baseline_final_equity_ratio",
            "Tournament Final Equity Ratio by Family",
            "equity_vs_benchmark_log.png",
            hline=1.0,
        ),
        "drawdown": _bar_chart(
            output_dir,
            summary,
            "family",
            "worst_strategy_max_dd",
            "Worst Strategy Drawdown by Family",
            "drawdown.png",
        ),
        "yearly": None,
        "exposure": _bar_chart(
            output_dir,
            summary,
            "family",
            "variant_win_rate",
            "Variant Win Rate by Family",
            "exposure_over_time.png",
        ),
        "cost": _chart_tournament_cost(output_dir, variants),
        "validation": _chart_tournament_validation_methods(output_dir, variants),
        "robustness": _bar_chart(
            output_dir,
            summary,
            "family",
            "robustness_score",
            "Tournament Robustness Score",
            "robustness_score.png",
        ),
    }

    if summary.empty:
        report_summary = pd.DataFrame()
        best_raw = best_risk = best_soxl = None
    else:
        best_raw = summary.sort_values("baseline_final_equity_ratio", ascending=False).iloc[0]
        best_risk = summary.sort_values("robustness_score", ascending=False).iloc[0]
        soxl = summary[summary["category"].astype(str).str.lower().str.contains("soxl")]
        best_soxl = soxl.sort_values("baseline_final_equity_ratio", ascending=False).iloc[0] if not soxl.empty else None
        report_summary = pd.DataFrame(
            [
                {
                    "best_raw_outperformer": best_raw["family"],
                    "best_raw_final_equity_ratio": best_raw["baseline_final_equity_ratio"],
                    "best_risk_adjusted_candidate": best_risk["family"],
                    "best_risk_robustness_score": best_risk["robustness_score"],
                    "best_soxl_candidate": "" if best_soxl is None else best_soxl["family"],
                    "candidate_for_further_research_count": int(summary["accepted_candidate"].map(_bool_value).sum())
                    if "accepted_candidate" in summary.columns
                    else 0,
                }
            ]
        )
    report_summary.to_csv(output_dir / "report_summary.csv", index=False)

    report_path = output_dir / "report.md"
    benchmark = str(config.get("benchmark_symbol", "TQQQ")).upper()
    lines = [
        f"# Tournament Report: {config.get('tournament_name', output_dir.name)}",
        "",
        "This report summarizes empirical backtest outputs only. It is not an investment recommendation.",
        "",
        "## 1. Objective",
        f"Compare configured candidate strategy families against same-period `{benchmark}` across tournament variants.",
        "",
        "## 2. Strategy description",
        "The tournament runs the strategy configs listed in the tournament YAML and records every variant, including failures.",
        "",
        "## 3. Data period",
        "Tournament rows may span different available data periods by family. See `tournament_variant_results.csv` for per-variant dates.",
        "",
        "## 4. Benchmark",
        f"Tournament benchmark: `{benchmark}`. SOXL-family strategies may also have native strategy-selection benchmarks in their individual configs.",
        "",
        "## 5. Same-period performance",
        _markdown_table(summary, max_rows=20),
        "",
        _relative_chart_path(report_path, chart_paths["equity"]),
        "",
        "## 6. Year-by-year returns",
        "Year-by-year returns are not retained at tournament aggregation level. Inspect each experiment output directory for `yearly_returns.csv`.",
        "",
        "## 7. Drawdowns",
        _markdown_table(summary, ["family", "baseline_strategy_max_dd", "worst_strategy_max_dd"], max_rows=20),
        "",
        _relative_chart_path(report_path, chart_paths["drawdown"]),
        "",
        "## 8. Exposure over time",
        "Daily exposure series are not retained at tournament aggregation level. The chart below uses variant win rate as a robustness proxy, not exposure.",
        "",
        _relative_chart_path(report_path, chart_paths["exposure"]),
        "",
        "## 9. Turnover and cost sensitivity",
        _markdown_table(summary, ["family", "cost_10_final_equity_ratio", "cost_25_final_equity_ratio", "cost_50_final_equity_ratio", "cost_sensitivity_25_vs_10"], max_rows=20),
        "",
        _relative_chart_path(report_path, chart_paths["cost"]),
        "",
        "## 10. Walk-forward parameter choices",
        "Tournament output keeps family-level and variant-level performance. Inspect individual experiment outputs for per-window parameter choices.",
        "",
        "## 11. Subperiod results",
        "Subperiod reports are generated by individual experiment runs, not by tournament aggregation.",
        "",
        "## 12. Robustness checks",
        _markdown_table(summary, ["family", "median_final_equity_ratio", "worst_final_equity_ratio", "validation_variants_beating_tqqq", "variant_win_rate", "robustness_score"], max_rows=20),
        "",
        _relative_chart_path(report_path, chart_paths["validation"]),
        "",
        _relative_chart_path(report_path, chart_paths["robustness"]),
        "",
        "Failures and rejections:",
        "",
        _markdown_table(failures, ["record_type", "family", "variant", "status", "error", "rejection_reason"], max_rows=30),
        "",
        "## 13. Verdict",
    ]
    if summary.empty:
        lines.extend(
            [
                "- Beats TQQQ raw wealth? no.",
                "- Better risk-adjusted profile? no.",
                "- Candidate for further research? no.",
            ]
        )
    else:
        accepted_count = int(summary["accepted_candidate"].map(_bool_value).sum()) if "accepted_candidate" in summary.columns else 0
        lines.extend(
            [
                f"- Beats TQQQ raw wealth? {_yes_no(bool((summary['baseline_final_equity_ratio'].astype(float) > 1.0).any()))}.",
                f"- Better risk-adjusted profile? {_yes_no(bool((summary['robustness_score'].astype(float) > 1.0).any()))}.",
                f"- Candidate for further research? {_yes_no(accepted_count > 0)}.",
                f"- Best raw outperformer: `{best_raw['family']}` with baseline final equity ratio `{_fmt(best_raw['baseline_final_equity_ratio'])}`.",
                f"- Best risk-adjusted candidate by robustness score: `{best_risk['family']}` with score `{_fmt(best_risk['robustness_score'])}`.",
                f"- Best SOXL candidate: `{best_soxl['family'] if best_soxl is not None else 'none'}`.",
            ]
        )
    lines.extend(
        [
            "",
            "Limitations: tournament aggregation does not preserve daily equity or exposure for every variant. Results are historical, parameter-selected, sensitive to execution and cost assumptions, and may not persist out of sample. No investment recommendation is made.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
