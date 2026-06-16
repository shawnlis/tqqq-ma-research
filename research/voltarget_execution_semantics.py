from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import add_cash_series
from .execution import CLOSE_TO_CLOSE_SHIFTED, NEXT_OPEN_TO_CLOSE, NEXT_OPEN_TO_NEXT_OPEN, ohlc_column
from .execution_financing import _load_price_data, _read_json, _resolve_output_dir, apply_slippage_and_financing, replay_locked_voltarget_candidate
from .fairness import constant_exposure_backtest
from .final_voltarget_audit import locate_plain_voltarget_candidate
from .metrics import annualized_return, annualized_volatility, calmar_ratio, max_drawdown, sharpe_ratio


BASE_EXECUTION_MODELS = (CLOSE_TO_CLOSE_SHIFTED, NEXT_OPEN_TO_NEXT_OPEN, NEXT_OPEN_TO_CLOSE)
TRANSACTION_COST_BPS_SCENARIOS = (10.0, 25.0, 50.0)
SLIPPAGE_BPS_SCENARIOS = (0.0, 5.0, 10.0)
FINANCING_ANNUAL_COST_SCENARIOS = (0.0, 0.03, 0.06, 0.09, 0.12)


@dataclass(frozen=True)
class ExecutionSemanticsResult:
    output_dir: Path
    summary: pd.DataFrame
    decomposition: pd.DataFrame
    summary_path: Path
    decomposition_path: Path
    report_path: Path
    chart_paths: Dict[str, Path]
    classification: str


def compute_return_decomposition(data: pd.DataFrame, symbol: str = "TQQQ") -> pd.DataFrame:
    symbol = str(symbol).upper()
    close = data[symbol].astype(float)
    open_ = data[ohlc_column(symbol, "OPEN")].astype(float)
    out = pd.DataFrame(index=data.index)
    out["overnight_return"] = open_ / close.shift(1) - 1.0
    out["intraday_return"] = close / open_ - 1.0
    out["close_to_close_return"] = close / close.shift(1) - 1.0
    out["open_to_open_return"] = open_.shift(-1) / open_ - 1.0
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def is_valid_default_execution_model_for_overnight_strategy(execution_model: str) -> bool:
    return execution_model in {CLOSE_TO_CLOSE_SHIFTED, NEXT_OPEN_TO_NEXT_OPEN}


def _period_mask(index: pd.DatetimeIndex, period: str) -> pd.Series:
    years = pd.Series(index.year, index=index)
    if period == "full_period":
        return pd.Series(True, index=index)
    if period == "ex_2022":
        return years != 2022
    if period == "post_2022":
        return years > 2022
    raise ValueError(f"Unsupported period: {period}")


def _multiple(frame: pd.DataFrame, period: str) -> float:
    ret = frame.loc[_period_mask(frame.index, period), "ret"].astype(float).fillna(0.0)
    if ret.empty:
        return np.nan
    return float((1.0 + ret).cumprod().iloc[-1])


def _max_dd(frame: pd.DataFrame, period: str) -> float:
    ret = frame.loc[_period_mask(frame.index, period), "ret"].astype(float).fillna(0.0)
    if ret.empty:
        return np.nan
    equity = (1.0 + ret).cumprod()
    return max_drawdown(equity)


def _performance_row(
    *,
    model: str,
    scenario_type: str,
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    transaction_cost_bps: float,
    slippage_bps: float,
    financing_annual_cost: float,
    valid_for_overnight_strategy: bool,
    notes: str = "",
) -> Dict[str, Any]:
    strategy_ret = strategy["ret"].astype(float).fillna(0.0)
    benchmark_ret = benchmark["ret"].astype(float).fillna(0.0).reindex(strategy.index)
    strategy_equity = (1.0 + strategy_ret).cumprod()
    benchmark_equity = (1.0 + benchmark_ret).cumprod()
    full_strategy = _multiple(strategy, "full_period")
    full_benchmark = _multiple(benchmark, "full_period")
    ex_strategy = _multiple(strategy, "ex_2022")
    ex_benchmark = _multiple(benchmark, "ex_2022")
    post_strategy = _multiple(strategy, "post_2022")
    post_benchmark = _multiple(benchmark, "post_2022")
    strategy_dd = max_drawdown(strategy_equity)
    benchmark_dd = max_drawdown(benchmark_equity)
    return {
        "execution_model": model,
        "scenario_type": scenario_type,
        "transaction_cost_bps": float(transaction_cost_bps),
        "slippage_bps": float(slippage_bps),
        "financing_annual_cost": float(financing_annual_cost),
        "valid_for_overnight_strategy": bool(valid_for_overnight_strategy),
        "start_date": str(pd.Timestamp(strategy.index.min()).date()),
        "end_date": str(pd.Timestamp(strategy.index.max()).date()),
        "date_count": int(len(strategy)),
        "strategy_final_equity": float(strategy_equity.iloc[-1]),
        "tqqq_final_equity": float(benchmark_equity.iloc[-1]),
        "full_period_ratio_vs_tqqq": full_strategy / full_benchmark if full_benchmark else np.nan,
        "ex_2022_ratio_vs_tqqq": ex_strategy / ex_benchmark if ex_benchmark else np.nan,
        "post_2022_ratio_vs_tqqq": post_strategy / post_benchmark if post_benchmark else np.nan,
        "strategy_cagr": annualized_return(strategy_equity),
        "tqqq_cagr": annualized_return(benchmark_equity),
        "strategy_vol": annualized_volatility(strategy_ret),
        "tqqq_vol": annualized_volatility(benchmark_ret),
        "strategy_sharpe": sharpe_ratio(strategy_ret),
        "tqqq_sharpe": sharpe_ratio(benchmark_ret),
        "strategy_max_drawdown": strategy_dd,
        "tqqq_max_drawdown": benchmark_dd,
        "max_drawdown_improvement": abs(benchmark_dd) - abs(strategy_dd),
        "strategy_calmar": calmar_ratio(annualized_return(strategy_equity), strategy_dd),
        "tqqq_calmar": calmar_ratio(annualized_return(benchmark_equity), benchmark_dd),
        "notes": notes,
    }


def _same_dates(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = left.index.intersection(right.index)
    if len(common) == 0:
        raise ValueError("No common dates are available for execution semantics comparison.")
    return left.loc[common].copy(), right.loc[common].copy()


def _benchmark_frame(data: pd.DataFrame, dates: pd.Index, execution_model: str, transaction_cost_bps: float) -> pd.DataFrame:
    benchmark = constant_exposure_backtest(
        data,
        dates=dates,
        exposure=1.0,
        trade_asset="TQQQ",
        transaction_cost_bps=float(transaction_cost_bps),
        execution_model=execution_model,
    )
    return benchmark


def _decomposition_contribution(
    *,
    name: str,
    decomposition: pd.DataFrame,
    position: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    if position is None:
        overnight = decomposition["overnight_return"]
        intraday = decomposition["intraday_return"]
    else:
        aligned_position = pd.Series(position, copy=True).astype(float).reindex(decomposition.index).fillna(0.0)
        overnight = aligned_position * decomposition["overnight_return"]
        intraday = aligned_position * decomposition["intraday_return"]
    overnight_log = np.log1p(overnight.clip(lower=-0.999999)).sum()
    intraday_log = np.log1p(intraday.clip(lower=-0.999999)).sum()
    total = overnight_log + intraday_log
    return {
        "series": name,
        "start_date": str(pd.Timestamp(decomposition.index.min()).date()),
        "end_date": str(pd.Timestamp(decomposition.index.max()).date()),
        "overnight_log_return": float(overnight_log),
        "intraday_log_return": float(intraday_log),
        "total_log_return": float(total),
        "overnight_share_of_total": float(overnight_log / total) if total else np.nan,
        "intraday_share_of_total": float(intraday_log / total) if total else np.nan,
        "close_to_close_multiple": float((1.0 + decomposition["close_to_close_return"].fillna(0.0)).cumprod().iloc[-1]),
        "open_to_open_multiple": float((1.0 + decomposition["open_to_open_return"].fillna(0.0)).cumprod().iloc[-1]),
    }


def classify_execution_semantics(summary: pd.DataFrame) -> str:
    base = summary[summary["scenario_type"].eq("base_execution")]
    c2c = base[base["execution_model"].eq(CLOSE_TO_CLOSE_SHIFTED)]
    n2n = base[base["execution_model"].eq(NEXT_OPEN_TO_NEXT_OPEN)]
    if n2n.empty:
        return "execution_fragile_needs_revision"
    n2n_row = n2n.iloc[0]
    n2n_ratio = float(n2n_row["full_period_ratio_vs_tqqq"])
    n2n_ex = float(n2n_row["ex_2022_ratio_vs_tqqq"])
    dd_improvement = float(n2n_row["max_drawdown_improvement"])
    financing_6 = summary[
        summary["scenario_type"].eq("financing_sensitivity")
        & summary["execution_model"].eq(NEXT_OPEN_TO_NEXT_OPEN)
        & np.isclose(pd.to_numeric(summary["financing_annual_cost"], errors="coerce"), 0.06)
        & np.isclose(pd.to_numeric(summary["transaction_cost_bps"], errors="coerce"), 10.0)
        & np.isclose(pd.to_numeric(summary["slippage_bps"], errors="coerce"), 0.0)
    ]
    financing_ok = not financing_6.empty and float(financing_6.iloc[0]["full_period_ratio_vs_tqqq"]) > 1.0
    ex_ok = n2n_ex > 1.0 or (n2n_ex > 0.90 and dd_improvement >= 0.10)
    if n2n_ratio > 1.0 and ex_ok and financing_ok:
        return "execution_ok_for_paper_monitor"
    c2c_ok = (not c2c.empty) and float(c2c.iloc[0]["full_period_ratio_vs_tqqq"]) > 1.0
    if c2c_ok and n2n_ratio < 0.80:
        return "execution_invalidates_strategy"
    return "execution_fragile_needs_revision"


def _write_equity_chart(base_frames: Dict[str, tuple[pd.DataFrame, pd.DataFrame]], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, (strategy, benchmark) in base_frames.items():
        strategy_equity = (1.0 + strategy["ret"].astype(float).fillna(0.0)).cumprod()
        strategy_equity.plot(ax=ax, label=f"VolTarget {model}")
        if model == CLOSE_TO_CLOSE_SHIFTED:
            benchmark_equity = (1.0 + benchmark["ret"].astype(float).fillna(0.0)).cumprod()
            benchmark_equity.plot(ax=ax, label="TQQQ buy-and-hold")
    ax.set_yscale("log")
    ax.set_title("VolTarget Equity by Execution Model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity, log scale")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "execution_model_equity_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_decomposition_chart(decomposition: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot = decomposition.set_index("series")[["overnight_log_return", "intraday_log_return"]]
    plot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Overnight vs Intraday Log Return Contribution")
    ax.set_xlabel("")
    ax.set_ylabel("Log return contribution")
    fig.tight_layout()
    path = output_dir / "overnight_vs_intraday_contribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    decomposition: pd.DataFrame,
    classification: str,
    chart_paths: Dict[str, Path],
) -> None:
    base = summary[summary["scenario_type"].eq("base_execution")]
    n2n = base[base["execution_model"].eq(NEXT_OPEN_TO_NEXT_OPEN)].iloc[0]
    n2c = base[base["execution_model"].eq(NEXT_OPEN_TO_CLOSE)].iloc[0]
    financing_6 = summary[
        summary["scenario_type"].eq("financing_sensitivity")
        & summary["execution_model"].eq(NEXT_OPEN_TO_NEXT_OPEN)
        & np.isclose(pd.to_numeric(summary["financing_annual_cost"], errors="coerce"), 0.06)
        & np.isclose(pd.to_numeric(summary["transaction_cost_bps"], errors="coerce"), 10.0)
        & np.isclose(pd.to_numeric(summary["slippage_bps"], errors="coerce"), 0.0)
    ]
    financing_6_ratio = float(financing_6.iloc[0]["full_period_ratio_vs_tqqq"]) if not financing_6.empty else np.nan
    voltarget_decomp = decomposition[decomposition["series"].eq("VolTarget close_to_close_weighted")]
    tqqq_decomp = decomposition[decomposition["series"].eq("TQQQ buy-and-hold")]
    overnight_share = float(voltarget_decomp.iloc[0]["overnight_share_of_total"]) if not voltarget_decomp.empty else np.nan
    overnight_advantage_share = np.nan
    intraday_advantage_log = np.nan
    if not voltarget_decomp.empty and not tqqq_decomp.empty:
        overnight_advantage_log = (
            float(voltarget_decomp.iloc[0]["overnight_log_return"])
            - float(tqqq_decomp.iloc[0]["overnight_log_return"])
        )
        intraday_advantage_log = (
            float(voltarget_decomp.iloc[0]["intraday_log_return"])
            - float(tqqq_decomp.iloc[0]["intraday_log_return"])
        )
        total_advantage_log = overnight_advantage_log + intraday_advantage_log
        overnight_advantage_share = overnight_advantage_log / total_advantage_log if total_advantage_log else np.nan
    paper_answer = "yes, as paper monitor only" if classification == "execution_ok_for_paper_monitor" else "not yet; resolve execution fragility first"
    lines = [
        "# VolTarget Execution Semantics Audit",
        "",
        "## Intended Trade Lifecycle",
        "- Signal is calculated after close t from information available at that close.",
        "- A realistic implementation cannot trade at close t after seeing that close unless a separate close-auction assumption is explicitly validated.",
        "- For an overnight holding strategy, the closest implemented execution model is trade at open t+1 and hold until the next rebalance open: `next_open_to_next_open`.",
        "- `next_open_to_close` is an intraday-only open-to-close exposure model. It excludes the overnight leg and is not a valid default for a strategy that remains invested overnight.",
        "- `next_close_to_next_close` is not implemented in the current execution engine.",
        "- next_close_to_next_close is not implemented.",
        "",
        "## Direct Answers",
        f"- Is next_open_to_close a valid model for this strategy? No. It is useful as a stress test, but not as the default for overnight holdings. Its full-period ratio is `{n2c['full_period_ratio_vs_tqqq']}`.",
        f"- Does next_open_to_next_open preserve the VolTarget edge? {'Yes' if float(n2n['full_period_ratio_vs_tqqq']) > 1.0 else 'No'}; full-period ratio `{n2n['full_period_ratio_vs_tqqq']}`, ex-2022 ratio `{n2n['ex_2022_ratio_vs_tqqq']}`.",
        f"- How much of the strategy's advantage comes from overnight returns? Overnight advantage share versus TQQQ is `{overnight_advantage_share}`; weighted strategy total-return overnight share is `{overnight_share}`; intraday advantage log contribution is `{intraday_advantage_log}`.",
        f"- Does realistic financing destroy the edge? {'No at the 6% baseline financing check' if financing_6_ratio > 1.0 else 'Yes or materially weakens it at the 6% baseline financing check'}; 6% ratio `{financing_6_ratio}`.",
        f"- Should we proceed to paper-trading monitor? {paper_answer}.",
        f"- Classification: `{classification}`.",
        "",
        "## Base Execution Summary",
        "```text",
        base.to_string(index=False),
        "```",
        "",
        "## Return Decomposition",
        "```text",
        decomposition.to_string(index=False),
        "```",
        "",
        "## Charts",
        f"- Execution model equity curves: `{chart_paths['execution_model_equity_curves']}`",
        f"- Overnight vs intraday contribution: `{chart_paths['overnight_vs_intraday_contribution']}`",
        "",
        "## Boundary",
        "This is an execution-semantics audit only. It does not change strategy alpha logic, add strategy families, run full/deep tournaments, place trades, or make investment recommendations.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_execution_semantics_audit(
    *,
    input_dir: Path,
    output_dir: Path,
    data_csv: Optional[Path] = None,
    cache_dir: str = "./price_cache",
    transaction_cost_bps_scenarios: Iterable[float] = TRANSACTION_COST_BPS_SCENARIOS,
    slippage_bps_scenarios: Iterable[float] = SLIPPAGE_BPS_SCENARIOS,
    financing_annual_cost_scenarios: Iterable[float] = FINANCING_ANNUAL_COST_SCENARIOS,
) -> ExecutionSemanticsResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = locate_plain_voltarget_candidate(input_dir)
    candidate_dir = _resolve_output_dir(input_dir, candidate.get("baseline_output_dir"))
    windows_path = candidate_dir / "walk_forward_windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing walk_forward_windows.csv: {windows_path}")
    windows = pd.read_csv(windows_path)
    run_config = _read_json(candidate_dir / "run_config.json")
    data = _load_price_data(run_config=run_config, windows=windows, data_csv=data_csv, cache_dir=cache_dir)
    if "CASH" not in data.columns:
        data = add_cash_series(data, include_ohlc=True)

    base_cost = float(run_config.get("transaction_cost_bps", 10.0))
    rows: list[Dict[str, Any]] = []
    base_frames: Dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for model in BASE_EXECUTION_MODELS:
        strategy = replay_locked_voltarget_candidate(
            data=data,
            windows=windows,
            transaction_cost_bps=base_cost,
            execution_model=model,
        )
        benchmark = _benchmark_frame(data, strategy.index, model, base_cost)
        strategy, benchmark = _same_dates(strategy, benchmark)
        base_frames[model] = (strategy, benchmark)
        rows.append(
            _performance_row(
                model=model,
                scenario_type="base_execution",
                strategy=strategy,
                benchmark=benchmark,
                transaction_cost_bps=base_cost,
                slippage_bps=0.0,
                financing_annual_cost=0.0,
                valid_for_overnight_strategy=is_valid_default_execution_model_for_overnight_strategy(model),
            )
        )

    rows.append(
        {
            "execution_model": "next_close_to_next_close",
            "scenario_type": "not_implemented",
            "transaction_cost_bps": base_cost,
            "slippage_bps": 0.0,
            "financing_annual_cost": 0.0,
            "valid_for_overnight_strategy": False,
            "notes": "not implemented in research.execution",
        }
    )

    for transaction_cost_bps in transaction_cost_bps_scenarios:
        base_strategy = replay_locked_voltarget_candidate(
            data=data,
            windows=windows,
            transaction_cost_bps=float(transaction_cost_bps),
            execution_model=NEXT_OPEN_TO_NEXT_OPEN,
        )
        base_benchmark = _benchmark_frame(data, base_strategy.index, NEXT_OPEN_TO_NEXT_OPEN, float(transaction_cost_bps))
        for slippage_bps in slippage_bps_scenarios:
            for financing in financing_annual_cost_scenarios:
                strategy = apply_slippage_and_financing(
                    base_strategy,
                    slippage_bps=float(slippage_bps),
                    financing_annual_cost=float(financing),
                )
                benchmark = apply_slippage_and_financing(
                    base_benchmark,
                    slippage_bps=float(slippage_bps),
                    financing_annual_cost=float(financing),
                )
                strategy, benchmark = _same_dates(strategy, benchmark)
                rows.append(
                    _performance_row(
                        model=NEXT_OPEN_TO_NEXT_OPEN,
                        scenario_type="financing_sensitivity",
                        strategy=strategy,
                        benchmark=benchmark,
                        transaction_cost_bps=float(transaction_cost_bps),
                        slippage_bps=float(slippage_bps),
                        financing_annual_cost=float(financing),
                        valid_for_overnight_strategy=True,
                    )
                )

    summary = pd.DataFrame(rows)
    decomposition_source = compute_return_decomposition(data.loc[base_frames[CLOSE_TO_CLOSE_SHIFTED][0].index], "TQQQ")
    c2c_strategy = base_frames[CLOSE_TO_CLOSE_SHIFTED][0]
    n2n_strategy = base_frames[NEXT_OPEN_TO_NEXT_OPEN][0]
    decomposition = pd.DataFrame(
        [
            _decomposition_contribution(name="TQQQ buy-and-hold", decomposition=decomposition_source),
            _decomposition_contribution(
                name="VolTarget close_to_close_weighted",
                decomposition=decomposition_source,
                position=c2c_strategy["tqqq_weight"] if "tqqq_weight" in c2c_strategy.columns else c2c_strategy["position"],
            ),
            _decomposition_contribution(
                name="VolTarget next_open_to_next_open_weighted",
                decomposition=decomposition_source,
                position=n2n_strategy["tqqq_weight"] if "tqqq_weight" in n2n_strategy.columns else n2n_strategy["position"],
            ),
        ]
    )
    classification = classify_execution_semantics(summary)
    chart_paths = {
        "execution_model_equity_curves": _write_equity_chart(base_frames, output_dir),
        "overnight_vs_intraday_contribution": _write_decomposition_chart(decomposition, output_dir),
    }
    summary_path = output_dir / "execution_semantics_summary.csv"
    decomposition_path = output_dir / "return_decomposition.csv"
    report_path = output_dir / "execution_semantics_report.md"
    summary["classification"] = classification
    summary.to_csv(summary_path, index=False)
    decomposition.to_csv(decomposition_path, index=False)
    _write_report(
        report_path,
        summary=summary,
        decomposition=decomposition,
        classification=classification,
        chart_paths=chart_paths,
    )
    return ExecutionSemanticsResult(
        output_dir=output_dir,
        summary=summary,
        decomposition=decomposition,
        summary_path=summary_path,
        decomposition_path=decomposition_path,
        report_path=report_path,
        chart_paths=chart_paths,
        classification=classification,
    )
