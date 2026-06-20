from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .assets import asset_config_from_row
from .backtest import make_vol_target_eval_piece
from .data import add_cash_series, load_prices
from .execution import (
    CLOSE_TO_CLOSE_SHIFTED,
    NEXT_OPEN_TO_CLOSE,
    NEXT_OPEN_TO_NEXT_OPEN,
    normalize_execution_model,
)
from .final_voltarget_audit import locate_plain_voltarget_candidate
from .metrics import annualized_return, annualized_volatility, calmar_ratio, max_drawdown, sharpe_ratio
from .strategies.drawdown_governor import DrawdownGovernorParams
from .strategies.market_internals import MarketInternalsParams
from .strategies.rebound import ReboundParams
from .strategies.vol_target_strategy import VolTargetParams


EXECUTION_MODELS = (CLOSE_TO_CLOSE_SHIFTED, NEXT_OPEN_TO_CLOSE, NEXT_OPEN_TO_NEXT_OPEN)
SLIPPAGE_BPS_SCENARIOS = (0.0, 5.0, 10.0, 25.0)
TRANSACTION_COST_BPS_SCENARIOS = (10.0, 25.0, 50.0, 100.0)
FINANCING_ANNUAL_COST_SCENARIOS = (0.0, 0.03, 0.06, 0.09, 0.12)
PERIODS_PER_YEAR = 252


@dataclass(frozen=True)
class ExecutionFinancingResult:
    output_dir: Path
    summary: pd.DataFrame
    summary_path: Path
    report_path: Path
    heatmap_path: Path
    still_beats_tqqq_after_realistic_financing: bool
    breaking_assumption: str


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _resolve_output_dir(input_dir: Path, output_value: Any) -> Path:
    text = str(output_value).strip()
    if not text:
        raise ValueError("Selected candidate does not include baseline_output_dir.")
    path = Path(text)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "outputs":
        return path
    return input_dir / path


def _load_price_data(
    *,
    run_config: Dict[str, Any],
    windows: pd.DataFrame,
    data_csv: Optional[Path],
    cache_dir: str,
) -> pd.DataFrame:
    start = str(run_config.get("start_date") or pd.to_datetime(windows["train_start"]).min().date())
    end_value = run_config.get("end_date")
    end = str(end_value) if end_value else None
    symbols = tuple(str(symbol).upper() for symbol in run_config.get("symbols", ["TQQQ", "QQQ"]))

    if data_csv is not None:
        data = pd.read_csv(data_csv, parse_dates=["Date"]).set_index("Date").sort_index()
        data = data.loc[data.index >= pd.Timestamp(start)]
        if end is not None:
            data = data.loc[data.index <= pd.Timestamp(end)]
    else:
        data = load_prices(
            start=start,
            end=end,
            symbols=symbols,
            use_csv_if_exists=True,
            cache_dir=cache_dir,
            include_ohlc=True,
        )
    if "CASH" not in data.columns:
        data = add_cash_series(data, include_ohlc=True)
    return data.sort_index()


def voltarget_params_from_window(row: pd.Series) -> VolTargetParams:
    asset_config = asset_config_from_row(row.to_dict())
    rebound = ReboundParams(
        use_rebound_module=_truthy(row.get("use_rebound_module", False)),
        rolling_high_window=_safe_int(row.get("rolling_high_window"), 126),
        drawdown_trigger=_safe_float(row.get("drawdown_trigger"), -0.20),
        rebound_momentum_window=_safe_int(row.get("rebound_momentum_window"), 10),
        rebound_momentum_threshold=_safe_float(row.get("rebound_momentum_threshold"), 0.05),
        reclaim_ma_window=_safe_int(row.get("reclaim_ma_window"), 20),
        rebound_position=_safe_float(row.get("rebound_position"), 1.0),
        rebound_hold_days=_safe_int(row.get("rebound_hold_days"), 10),
        extreme_crash_vol_window=_safe_int(row.get("extreme_crash_vol_window"), 20),
        extreme_crash_vol_threshold=_safe_float(row.get("extreme_crash_vol_threshold"), 0.05),
    )
    governor = DrawdownGovernorParams(
        use_drawdown_governor=_truthy(row.get("use_drawdown_governor", False)),
        portfolio_dd_trigger=_safe_float(row.get("portfolio_dd_trigger"), -0.30),
        qqq_dd_trigger=_safe_float(row.get("qqq_dd_trigger"), -0.15),
        reduced_exposure=_safe_float(row.get("reduced_exposure"), 0.50),
        recovery_ma_window=_safe_int(row.get("recovery_ma_window"), 20),
        recovery_momentum_window=_safe_int(row.get("recovery_momentum_window"), 10),
        recovery_momentum_threshold=_safe_float(row.get("recovery_momentum_threshold"), 0.05),
        max_days_reduced=_safe_int(row.get("max_days_reduced"), 40),
    )
    market_internals = MarketInternalsParams(
        use_market_internals=_truthy(row.get("use_market_internals", False)),
        signal_window=_safe_int(row.get("market_internals_signal_window"), 100),
        risk_on_threshold=_safe_float(row.get("market_internals_risk_on_threshold"), 0.60),
        risk_off_threshold=_safe_float(row.get("market_internals_risk_off_threshold"), 0.40),
        score_method=str(row.get("market_internals_score_method", "binary") or "binary"),
        min_components=_safe_int(row.get("market_internals_min_components"), 1),
        vix_risk_on_threshold=_safe_float(row.get("vix_risk_on_threshold"), 25.0),
        vix_risk_off_threshold=_safe_float(row.get("vix_risk_off_threshold"), 35.0),
    )
    return VolTargetParams(
        target_ann_vol=_safe_float(row.get("target_ann_vol")),
        realized_vol_window=_safe_int(row.get("realized_vol_window"), 20),
        min_exposure=_safe_float(row.get("min_exposure"), 0.0),
        max_exposure=_safe_float(row.get("max_exposure"), 1.0),
        trend_window=_safe_int(row.get("trend_window"), 150),
        momentum_window=_safe_int(row.get("momentum_window"), 20),
        trend_multiplier_below_ma=_safe_float(row.get("trend_multiplier_below_ma"), 1.0),
        momentum_boost=_safe_float(row.get("momentum_boost"), 1.0),
        crash_vol_cutoff=_safe_float(row.get("crash_vol_cutoff"), 1.0e9),
        crash_exposure=_safe_float(row.get("crash_exposure"), 0.0),
        rebound=rebound,
        governor=governor,
        risk_off_symbol=str(row.get("risk_off_symbol", "CASH") or "CASH").upper(),
        risk_off_weight=_safe_float(row.get("risk_off_weight"), 0.0),
        market_internals=market_internals,
        asset_config=asset_config,
    )


def apply_slippage_and_financing(
    stitched: pd.DataFrame,
    *,
    slippage_bps: float,
    financing_annual_cost: float,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.DataFrame:
    out = stitched.copy()
    turnover = out["turnover"].astype(float) if "turnover" in out.columns else pd.Series(0.0, index=out.index)
    if "trade_weight" in out.columns:
        exposure = out["trade_weight"].astype(float)
    elif "tqqq_weight" in out.columns:
        exposure = out["tqqq_weight"].astype(float)
    else:
        exposure = out["position"].astype(float)

    out["slippage_cost"] = turnover * (float(slippage_bps) / 10000.0)
    out["financing_exposure"] = (exposure - 1.0).clip(lower=0.0)
    out["financing_cost"] = out["financing_exposure"] * (float(financing_annual_cost) / float(periods_per_year))
    out["ret_before_slippage_financing"] = out["ret"].astype(float)
    out["ret"] = out["ret_before_slippage_financing"] - out["slippage_cost"] - out["financing_cost"]
    out["total_cost"] = out.get("cost", 0.0) + out["slippage_cost"] + out["financing_cost"]
    out["equity"] = (1.0 + out["ret"].fillna(0.0)).cumprod()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    return out


def replay_locked_voltarget_candidate(
    *,
    data: pd.DataFrame,
    windows: pd.DataFrame,
    transaction_cost_bps: float,
    execution_model: str,
) -> pd.DataFrame:
    execution_model = normalize_execution_model(execution_model)
    pieces: list[pd.DataFrame] = []
    prev_position = 0.0
    prev_risk_off_weight = 0.0

    for _, row in windows.iterrows():
        params = voltarget_params_from_window(row)
        eval_start = pd.Timestamp(row["test_start"])
        eval_end = pd.Timestamp(row["test_end"])
        piece = make_vol_target_eval_piece(
            full_data=data,
            params=params,
            eval_start=eval_start,
            eval_end=eval_end,
            transaction_cost_bps=float(transaction_cost_bps),
            prev_position=prev_position,
            prev_risk_off_weight=prev_risk_off_weight,
            execution_model=execution_model,
        )
        if piece.empty:
            continue
        prev_position = float(piece["tqqq_weight"].iloc[-1]) if "tqqq_weight" in piece.columns else float(piece["position"].iloc[-1])
        prev_risk_off_weight = float(piece["risk_off_weight"].iloc[-1]) if "risk_off_weight" in piece.columns else 0.0
        pieces.append(piece)

    if not pieces:
        raise ValueError("No stitched VolTarget pieces were produced.")
    stitched = pd.concat(pieces).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    stitched["equity"] = (1.0 + stitched["ret"].fillna(0.0)).cumprod()
    stitched["drawdown"] = stitched["equity"] / stitched["equity"].cummax() - 1.0
    return stitched


def _years(index: pd.Index) -> float:
    if len(index) < 2:
        return np.nan
    return max((pd.Timestamp(index[-1]) - pd.Timestamp(index[0])).days / 365.25, 1.0 / PERIODS_PER_YEAR)


def summarize_scenario(
    stitched: pd.DataFrame,
    *,
    execution_model: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    financing_annual_cost: float,
) -> Dict[str, Any]:
    strategy_equity = stitched["equity"].astype(float)
    strategy_ret = stitched["ret"].astype(float)
    benchmark_ret = stitched["daily_ret_tqqq"].astype(float)
    benchmark_equity = (1.0 + benchmark_ret.fillna(0.0)).cumprod()
    final_ratio = float(strategy_equity.iloc[-1] / benchmark_equity.iloc[-1]) if benchmark_equity.iloc[-1] != 0 else np.nan
    cagr = annualized_return(strategy_equity)
    mdd = max_drawdown(strategy_equity)
    bench_cagr = annualized_return(benchmark_equity)
    bench_mdd = max_drawdown(benchmark_equity)
    years = _years(stitched.index)
    effective_models = ";".join(sorted(str(v) for v in stitched.get("execution_model", pd.Series(dtype=object)).dropna().unique()))
    warnings = ";".join(sorted(str(v) for v in stitched.get("execution_model_warning", pd.Series(dtype=object)).dropna().unique() if str(v)))
    return {
        "execution_model": normalize_execution_model(execution_model),
        "effective_execution_model": effective_models or normalize_execution_model(execution_model),
        "execution_model_warning": warnings,
        "transaction_cost_bps": float(transaction_cost_bps),
        "slippage_bps": float(slippage_bps),
        "financing_annual_cost": float(financing_annual_cost),
        "start_date": str(pd.Timestamp(stitched.index.min()).date()),
        "end_date": str(pd.Timestamp(stitched.index.max()).date()),
        "date_count": int(len(stitched)),
        "strategy_final_equity": float(strategy_equity.iloc[-1]),
        "tqqq_final_equity": float(benchmark_equity.iloc[-1]),
        "final_equity_ratio_vs_tqqq": final_ratio,
        "beats_tqqq": bool(final_ratio > 1.0) if pd.notna(final_ratio) else False,
        "strategy_cagr": cagr,
        "tqqq_cagr": bench_cagr,
        "strategy_vol": annualized_volatility(strategy_ret),
        "tqqq_vol": annualized_volatility(benchmark_ret),
        "strategy_sharpe": sharpe_ratio(strategy_ret),
        "tqqq_sharpe": sharpe_ratio(benchmark_ret),
        "strategy_max_drawdown": mdd,
        "tqqq_max_drawdown": bench_mdd,
        "strategy_calmar": calmar_ratio(cagr, mdd),
        "tqqq_calmar": calmar_ratio(bench_cagr, bench_mdd),
        "avg_exposure": float(stitched["position"].astype(float).mean()) if "position" in stitched.columns else np.nan,
        "max_exposure": float(stitched["position"].astype(float).max()) if "position" in stitched.columns else np.nan,
        "annual_turnover": float(stitched["turnover"].astype(float).sum() / years) if "turnover" in stitched.columns and years > 0 else np.nan,
        "total_transaction_cost": float(stitched["cost"].astype(float).sum()) if "cost" in stitched.columns else np.nan,
        "total_slippage_cost": float(stitched["slippage_cost"].astype(float).sum()) if "slippage_cost" in stitched.columns else 0.0,
        "total_financing_cost": float(stitched["financing_cost"].astype(float).sum()) if "financing_cost" in stitched.columns else 0.0,
        "total_cost": float(stitched["total_cost"].astype(float).sum()) if "total_cost" in stitched.columns else np.nan,
    }


def _severity_columns(df: pd.DataFrame) -> pd.DataFrame:
    order = {model: idx for idx, model in enumerate(EXECUTION_MODELS)}
    out = df.copy()
    out["_execution_order"] = out["execution_model"].map(order).fillna(99).astype(int)
    return out.sort_values(["financing_annual_cost", "transaction_cost_bps", "slippage_bps", "_execution_order"])


def identify_breaking_assumption(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "no scenarios were produced"
    failed = _severity_columns(summary.loc[~summary["beats_tqqq"].astype(bool)])
    if failed.empty:
        return "No tested execution, slippage, transaction-cost, or financing scenario breaks final equity ratio below 1.0."
    row = failed.iloc[0]
    return (
        f"first failing scenario: execution_model={row['execution_model']}, "
        f"transaction_cost_bps={row['transaction_cost_bps']}, slippage_bps={row['slippage_bps']}, "
        f"financing_annual_cost={row['financing_annual_cost']}, "
        f"final_equity_ratio_vs_tqqq={row['final_equity_ratio_vs_tqqq']}"
    )


def write_sensitivity_heatmap(summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap = (
        summary.groupby(["financing_annual_cost", "transaction_cost_bps"], as_index=False)["final_equity_ratio_vs_tqqq"]
        .min()
        .pivot(index="financing_annual_cost", columns="transaction_cost_bps", values="final_equity_ratio_vs_tqqq")
        .sort_index(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    values = heatmap.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=max(2.0, np.nanmax(values)))
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels([f"{float(v):g}" for v in heatmap.columns])
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels([f"{float(v):.0%}" for v in heatmap.index])
    ax.set_xlabel("Transaction cost, bps")
    ax.set_ylabel("Annual financing cost")
    ax.set_title("Worst final equity ratio vs TQQQ by cost and financing")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if np.isfinite(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Min final equity ratio vs TQQQ")
    fig.tight_layout()
    path = output_dir / "sensitivity_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_report(path: Path, summary: pd.DataFrame, heatmap_path: Path, breaking_assumption: str) -> None:
    best = summary.sort_values("final_equity_ratio_vs_tqqq", ascending=False).iloc[0]
    worst = summary.sort_values("final_equity_ratio_vs_tqqq", ascending=True).iloc[0]
    all_pass = bool(summary["beats_tqqq"].astype(bool).all())
    failed_count = int((~summary["beats_tqqq"].astype(bool)).sum())
    answer = (
        "Yes across all tested scenarios."
        if all_pass
        else f"No. {failed_count} of {len(summary)} tested scenarios failed to beat same-period TQQQ."
    )
    lines = [
        "# VolTarget Execution and Financing Realism Audit",
        "",
        "This audit replays the locked Stage 3 VolTarget candidate parameters under execution, slippage, transaction-cost, and financing assumptions. It does not change strategy signal logic and does not run a parameter search.",
        "",
        "## Direct Answers",
        f"- Does VolTarget still beat TQQQ after realistic financing? {answer}",
        f"- Which assumption breaks the result? {breaking_assumption}",
        "",
        "## Tested Assumptions",
        "- Execution models: close_to_close_shifted, next_open_to_close, next_open_to_next_open.",
        "- Slippage bps: 0, 5, 10, 25.",
        "- Transaction cost bps: 10, 25, 50, 100.",
        "- Margin/financing annual cost: 0%, 3%, 6%, 9%, 12%.",
        "- Extra financing cost is applied only to exposure above 1.0x.",
        "",
        "## Best Scenario",
        f"- Execution model: {best['execution_model']}",
        f"- Transaction cost bps: {best['transaction_cost_bps']}",
        f"- Slippage bps: {best['slippage_bps']}",
        f"- Financing annual cost: {best['financing_annual_cost']}",
        f"- Final equity ratio vs TQQQ: {best['final_equity_ratio_vs_tqqq']}",
        "",
        "## Worst Scenario",
        f"- Execution model: {worst['execution_model']}",
        f"- Transaction cost bps: {worst['transaction_cost_bps']}",
        f"- Slippage bps: {worst['slippage_bps']}",
        f"- Financing annual cost: {worst['financing_annual_cost']}",
        f"- Final equity ratio vs TQQQ: {worst['final_equity_ratio_vs_tqqq']}",
        "",
        "## Files",
        "- execution_financing_summary.csv",
        f"- {heatmap_path.name}",
        "",
        "## Boundary",
        "This is empirical backtest sensitivity analysis only. It is not an investment recommendation, not production readiness, and not an auto-trading workflow.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_execution_financing_audit(
    *,
    input_dir: Path,
    output_dir: Path,
    data_csv: Optional[Path] = None,
    cache_dir: str = "./price_cache",
    execution_models: Iterable[str] = EXECUTION_MODELS,
    slippage_bps_scenarios: Iterable[float] = SLIPPAGE_BPS_SCENARIOS,
    transaction_cost_bps_scenarios: Iterable[float] = TRANSACTION_COST_BPS_SCENARIOS,
    financing_annual_cost_scenarios: Iterable[float] = FINANCING_ANNUAL_COST_SCENARIOS,
) -> ExecutionFinancingResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = locate_plain_voltarget_candidate(input_dir)
    candidate_dir = _resolve_output_dir(input_dir, candidate.get("baseline_output_dir"))
    windows_path = candidate_dir / "walk_forward_windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing walk_forward_windows.csv: {windows_path}")
    windows = pd.read_csv(windows_path)
    if windows.empty:
        raise ValueError(f"Window file is empty: {windows_path}")
    run_config = _read_json(candidate_dir / "run_config.json")
    data = _load_price_data(run_config=run_config, windows=windows, data_csv=data_csv, cache_dir=cache_dir)

    rows: list[Dict[str, Any]] = []
    base_cache: Dict[tuple[str, float], pd.DataFrame] = {}
    for execution_model in execution_models:
        normalized_model = normalize_execution_model(execution_model)
        for transaction_cost_bps in transaction_cost_bps_scenarios:
            key = (normalized_model, float(transaction_cost_bps))
            base = base_cache.get(key)
            if base is None:
                base = replay_locked_voltarget_candidate(
                    data=data,
                    windows=windows,
                    transaction_cost_bps=float(transaction_cost_bps),
                    execution_model=normalized_model,
                )
                base_cache[key] = base
            for slippage_bps in slippage_bps_scenarios:
                for financing_annual_cost in financing_annual_cost_scenarios:
                    adjusted = apply_slippage_and_financing(
                        base,
                        slippage_bps=float(slippage_bps),
                        financing_annual_cost=float(financing_annual_cost),
                    )
                    rows.append(
                        summarize_scenario(
                            adjusted,
                            execution_model=normalized_model,
                            transaction_cost_bps=float(transaction_cost_bps),
                            slippage_bps=float(slippage_bps),
                            financing_annual_cost=float(financing_annual_cost),
                        )
                    )

    summary = pd.DataFrame(rows).sort_values(
        ["execution_model", "transaction_cost_bps", "slippage_bps", "financing_annual_cost"]
    )
    summary_path = output_dir / "execution_financing_summary.csv"
    report_path = output_dir / "execution_financing_report.md"
    heatmap_path = write_sensitivity_heatmap(summary, output_dir)
    breaking_assumption = identify_breaking_assumption(summary)
    still_beats = bool(summary["beats_tqqq"].astype(bool).all())

    summary.to_csv(summary_path, index=False)
    _write_report(report_path, summary, heatmap_path, breaking_assumption)
    return ExecutionFinancingResult(
        output_dir=output_dir,
        summary=summary,
        summary_path=summary_path,
        report_path=report_path,
        heatmap_path=heatmap_path,
        still_beats_tqqq_after_realistic_financing=still_beats,
        breaking_assumption=breaking_assumption,
    )