from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .execution import CLOSE_TO_CLOSE_SHIFTED, normalize_execution_model
from .execution_financing import _load_price_data, _read_json, _resolve_output_dir, replay_locked_voltarget_candidate
from .fairness import constant_exposure_backtest
from .final_voltarget_audit import locate_plain_voltarget_candidate
from .metrics import annualized_return, calmar_ratio, max_drawdown, sharpe_ratio
from .voltarget_fair_leverage import realized_average_exposure, replay_simple_voltarget_no_trend


PERIODS = ("full_period", "ex_2022", "post_2022")


@dataclass(frozen=True)
class SimplificationBattleResult:
    output_dir: Path
    summary: pd.DataFrame
    summary_path: Path
    report_path: Path
    chart_paths: Dict[str, Path]
    full_period_winner: str
    ex_2022_winner: str
    post_2022_winner: str
    best_drawdown_model: str
    trend_momentum_answer: str


def _read_date_indexed_csv(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    data = pd.read_csv(path, parse_dates=["Date"] if "Date" in columns else None)
    if "Date" in data.columns:
        data = data.set_index("Date")
    else:
        first = data.columns[0]
        data[first] = pd.to_datetime(data[first])
        data = data.set_index(first)
    data.index.name = "Date"
    if "equity" not in data.columns and "ret" in data.columns:
        data["equity"] = (1.0 + data["ret"].astype(float).fillna(0.0)).cumprod()
    if "drawdown" not in data.columns and "equity" in data.columns:
        equity = data["equity"].astype(float)
        data["drawdown"] = equity / equity.cummax() - 1.0
    return data.sort_index()


def _period_mask(index: pd.DatetimeIndex, period: str) -> pd.Series:
    years = pd.Series(index.year, index=index)
    if period == "full_period":
        return pd.Series(True, index=index)
    if period == "ex_2022":
        return years != 2022
    if period == "post_2022":
        return years > 2022
    raise ValueError(f"Unsupported period: {period}")


def _period_return(frame: pd.DataFrame, period: str) -> pd.Series:
    mask = _period_mask(frame.index, period)
    return frame.loc[mask, "ret"].astype(float).fillna(0.0)


def _period_equity_multiple(frame: pd.DataFrame, period: str) -> float:
    ret = _period_return(frame, period)
    if ret.empty:
        return np.nan
    return float((1.0 + ret).cumprod().iloc[-1])


def _period_max_drawdown(frame: pd.DataFrame, period: str) -> float:
    ret = _period_return(frame, period)
    if ret.empty:
        return np.nan
    equity = (1.0 + ret).cumprod()
    return max_drawdown(equity)


def _summary_for_model(name: str, frame: pd.DataFrame) -> Dict[str, Any]:
    ret = frame["ret"].astype(float).fillna(0.0)
    equity = (1.0 + ret).cumprod()
    full_dd = max_drawdown(equity)
    row: Dict[str, Any] = {
        "model": name,
        "start_date": str(pd.Timestamp(frame.index.min()).date()),
        "end_date": str(pd.Timestamp(frame.index.max()).date()),
        "date_count": int(len(frame)),
        "full_period_final_multiple": _period_equity_multiple(frame, "full_period"),
        "ex_2022_final_multiple": _period_equity_multiple(frame, "ex_2022"),
        "post_2022_final_multiple": _period_equity_multiple(frame, "post_2022"),
        "full_period_max_drawdown": full_dd,
        "ex_2022_max_drawdown": _period_max_drawdown(frame, "ex_2022"),
        "post_2022_max_drawdown": _period_max_drawdown(frame, "post_2022"),
        "cagr": annualized_return(equity),
        "sharpe": sharpe_ratio(ret),
        "calmar": calmar_ratio(annualized_return(equity), full_dd),
        "avg_exposure": float(frame["position"].astype(float).mean()) if "position" in frame.columns else np.nan,
        "max_exposure": float(frame["position"].astype(float).max()) if "position" in frame.columns else np.nan,
        "total_transaction_cost": float(frame["cost"].astype(float).sum()) if "cost" in frame.columns else np.nan,
    }
    return row


def _winner(summary: pd.DataFrame, column: str) -> str:
    values = pd.to_numeric(summary[column], errors="coerce")
    if values.dropna().empty:
        return "not_available"
    return str(summary.loc[values.idxmax(), "model"])


def _drawdown_winner(summary: pd.DataFrame) -> str:
    values = pd.to_numeric(summary["full_period_max_drawdown"], errors="coerce")
    if values.dropna().empty:
        return "not_available"
    return str(summary.loc[values.idxmax(), "model"])


def _trend_momentum_answer(summary: pd.DataFrame) -> str:
    locked = summary[summary["model"].eq("locked_voltarget")]
    simple = summary[summary["model"].eq("simple_voltarget_no_trend")]
    if locked.empty or simple.empty:
        return "not_available"
    locked_row = locked.iloc[0]
    simple_row = simple.iloc[0]
    full_ratio = float(locked_row["full_period_final_multiple"]) / float(simple_row["full_period_final_multiple"])
    ex_ratio = float(locked_row["ex_2022_final_multiple"]) / float(simple_row["ex_2022_final_multiple"])
    dd_delta = float(locked_row["full_period_max_drawdown"]) - float(simple_row["full_period_max_drawdown"])
    if full_ratio > 1.0 and ex_ratio > 1.0 and dd_delta >= 0.0:
        return "yes; locked trend/momentum layer improves full-period and ex-2022 return without worse drawdown"
    if full_ratio > 1.0 and ex_ratio <= 1.0:
        return "mixed; trend/momentum helps full period but fails ex-2022 versus the simple no-trend version"
    if full_ratio <= 1.0:
        return "no; simple no-trend version matches or beats locked VolTarget over the full period"
    return "mixed; review return and drawdown tradeoff"


def _align_frames(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    common_index: Optional[pd.Index] = None
    for frame in frames.values():
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)
    if common_index is None or len(common_index) == 0:
        raise ValueError("No common comparison dates are available.")
    return {name: frame.loc[common_index].copy() for name, frame in frames.items()}


def _load_locked_stitched(candidate_dir: Path) -> pd.DataFrame:
    path = candidate_dir / "stitched_equity.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing stitched_equity.csv: {path}")
    return _read_date_indexed_csv(path)


def _make_model_frames(
    *,
    input_dir: Path,
    data_csv: Optional[Path],
    cache_dir: str,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    candidate = locate_plain_voltarget_candidate(input_dir)
    candidate_dir = _resolve_output_dir(input_dir, candidate.get("baseline_output_dir"))
    windows_path = candidate_dir / "walk_forward_windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing walk_forward_windows.csv: {windows_path}")
    windows = pd.read_csv(windows_path)
    run_config = _read_json(candidate_dir / "run_config.json")
    transaction_cost_bps = float(run_config.get("transaction_cost_bps", 10.0))
    execution_model = normalize_execution_model(run_config.get("execution_model", CLOSE_TO_CLOSE_SHIFTED))
    data = _load_price_data(
        run_config=run_config,
        windows=windows,
        data_csv=data_csv,
        cache_dir=cache_dir,
    )

    locked = _load_locked_stitched(candidate_dir)
    if "ret" not in locked.columns:
        locked = replay_locked_voltarget_candidate(
            data=data,
            windows=windows,
            transaction_cost_bps=transaction_cost_bps,
            execution_model=execution_model,
        )
    simple = replay_simple_voltarget_no_trend(
        price_data=data,
        wf_table=windows,
        transaction_cost_bps=transaction_cost_bps,
        execution_model=execution_model,
    )
    dates = locked.index
    avg_exposure = realized_average_exposure(locked)
    max_exposure = float(pd.to_numeric(locked.get("position", locked.get("tqqq_weight")), errors="coerce").max())
    trade_asset = str(run_config.get("benchmark_symbol", "TQQQ")).upper()
    same_avg = constant_exposure_backtest(
        data,
        dates=dates,
        exposure=avg_exposure,
        trade_asset=trade_asset,
        transaction_cost_bps=transaction_cost_bps,
        execution_model=execution_model,
    )
    same_max = constant_exposure_backtest(
        data,
        dates=dates,
        exposure=max_exposure,
        trade_asset=trade_asset,
        transaction_cost_bps=transaction_cost_bps,
        execution_model=execution_model,
    )
    buy_hold = constant_exposure_backtest(
        data,
        dates=dates,
        exposure=1.0,
        trade_asset=trade_asset,
        transaction_cost_bps=transaction_cost_bps,
        execution_model=execution_model,
    )
    frames = _align_frames(
        {
            "locked_voltarget": locked,
            "simple_voltarget_no_trend": simple,
            "constant_same_average_exposure_tqqq": same_avg,
            "constant_same_max_exposure_tqqq": same_max,
            "tqqq_buy_and_hold": buy_hold,
        }
    )
    metadata = {
        "candidate_dir": candidate_dir,
        "transaction_cost_bps": transaction_cost_bps,
        "execution_model": execution_model,
        "avg_exposure": avg_exposure,
        "max_exposure": max_exposure,
    }
    return frames, metadata


def _write_relative_equity_chart(frames: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
    benchmark = (1.0 + frames["tqqq_buy_and_hold"]["ret"].astype(float).fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, frame in frames.items():
        equity = (1.0 + frame["ret"].astype(float).fillna(0.0)).cumprod()
        relative = equity / benchmark
        relative.plot(ax=ax, label=name)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("Relative Equity Versus TQQQ Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Relative equity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "relative_equity_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _yearly_returns(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    benchmark = frames["tqqq_buy_and_hold"]
    benchmark_yearly = benchmark["ret"].astype(float).groupby(benchmark.index.year).agg(
        lambda values: float((1.0 + values).prod() - 1.0)
    )
    for name, frame in frames.items():
        yearly = frame["ret"].astype(float).groupby(frame.index.year).agg(
            lambda values: float((1.0 + values).prod() - 1.0)
        )
        for year, strategy_return in yearly.items():
            tqqq_return = float(benchmark_yearly.loc[year])
            rows.append(
                {
                    "year": int(year),
                    "model": name,
                    "strategy_return": float(strategy_return),
                    "tqqq_return": tqqq_return,
                    "relative_return": float(strategy_return - tqqq_return),
                }
            )
    return pd.DataFrame(rows)


def _write_yearly_relative_chart(yearly: pd.DataFrame, output_dir: Path) -> Path:
    pivot = yearly.pivot(index="year", columns="model", values="relative_return").sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Yearly Relative Returns Versus TQQQ Buy-and-Hold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return difference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "yearly_relative_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    metadata: Dict[str, Any],
    full_winner: str,
    ex_2022_winner: str,
    post_2022_winner: str,
    drawdown_winner: str,
    trend_answer: str,
) -> None:
    lines = [
        "# VolTarget Simplification Battle",
        "",
        "This audit compares the locked Stage 3 VolTarget candidate against simpler alternatives without parameter tuning or new strategy families.",
        "",
        "## Assumptions",
        f"- Transaction cost bps: {metadata['transaction_cost_bps']}",
        f"- Execution model: {metadata['execution_model']}",
        f"- Same-average exposure: {metadata['avg_exposure']}",
        f"- Same-max exposure: {metadata['max_exposure']}",
        "- All models use identical dates and the same configured cost assumption.",
        "",
        "## Direct Answers",
        f"- Which model wins full period? `{full_winner}`.",
        f"- Which wins ex-2022? `{ex_2022_winner}`.",
        f"- Which wins post-2022? `{post_2022_winner}`.",
        f"- Which has better drawdown? `{drawdown_winner}`.",
        f"- Is trend/momentum layer worth keeping? {trend_answer}.",
        "",
        "## Summary Table",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Boundary",
        "This is an empirical simplification comparison only. It is not an investment recommendation and does not change strategy signal logic.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_simplification_battle(
    *,
    input_dir: Path,
    output_dir: Path,
    data_csv: Optional[Path] = None,
    cache_dir: str = "./price_cache",
) -> SimplificationBattleResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, metadata = _make_model_frames(input_dir=input_dir, data_csv=data_csv, cache_dir=cache_dir)
    rows = [_summary_for_model(name, frame) for name, frame in frames.items()]
    summary = pd.DataFrame(rows)
    full_winner = _winner(summary, "full_period_final_multiple")
    ex_2022_winner = _winner(summary, "ex_2022_final_multiple")
    post_2022_winner = _winner(summary, "post_2022_final_multiple")
    drawdown_winner = _drawdown_winner(summary)
    trend_answer = _trend_momentum_answer(summary)

    summary["wins_full_period"] = summary["model"].eq(full_winner)
    summary["wins_ex_2022"] = summary["model"].eq(ex_2022_winner)
    summary["wins_post_2022"] = summary["model"].eq(post_2022_winner)
    summary["best_full_period_drawdown"] = summary["model"].eq(drawdown_winner)
    summary["trend_momentum_answer"] = trend_answer

    summary_path = output_dir / "simplification_battle_summary.csv"
    report_path = output_dir / "simplification_battle_report.md"
    summary.to_csv(summary_path, index=False)
    relative_chart = _write_relative_equity_chart(frames, output_dir)
    yearly_chart = _write_yearly_relative_chart(_yearly_returns(frames), output_dir)
    _write_report(
        report_path,
        summary=summary,
        metadata=metadata,
        full_winner=full_winner,
        ex_2022_winner=ex_2022_winner,
        post_2022_winner=post_2022_winner,
        drawdown_winner=drawdown_winner,
        trend_answer=trend_answer,
    )

    return SimplificationBattleResult(
        output_dir=output_dir,
        summary=summary,
        summary_path=summary_path,
        report_path=report_path,
        chart_paths={
            "relative_equity_comparison": relative_chart,
            "yearly_relative_returns": yearly_chart,
        },
        full_period_winner=full_winner,
        ex_2022_winner=ex_2022_winner,
        post_2022_winner=post_2022_winner,
        best_drawdown_model=drawdown_winner,
        trend_momentum_answer=trend_answer,
    )
