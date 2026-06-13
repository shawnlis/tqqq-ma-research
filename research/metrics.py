from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return np.nan

    total_return = equity.iloc[-1] / equity.iloc[0]
    if total_return <= 0:
        return np.nan

    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return np.nan

    return float(total_return ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=0) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    if len(returns) < 2:
        return np.nan

    # ddof=0 is intentional for internal consistency in this framework.
    excess_daily = returns - rf / periods_per_year
    vol = excess_daily.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return np.nan

    return float(excess_daily.mean() / vol * math.sqrt(periods_per_year))


def calmar_ratio(cagr: float, mdd: float) -> float:
    if np.isnan(cagr) or np.isnan(mdd) or mdd == 0:
        return np.nan
    return float(cagr / abs(mdd))


def _annual_turnover(turnover: Optional[pd.Series], periods_per_year: int) -> float:
    if turnover is None or len(turnover) == 0:
        return 0.0
    years = max((len(turnover) - 1) / periods_per_year, 1.0 / periods_per_year)
    return float(pd.Series(turnover).fillna(0.0).sum() / years)


def _yearly_relative_stats(
    strategy_ret: pd.Series,
    benchmark_ret: pd.Series,
) -> Dict[str, float]:
    if not isinstance(strategy_ret.index, pd.DatetimeIndex):
        return {
            "years_strategy_beats_benchmark": 0.0,
            "worst_relative_year_loss": 0.0,
            "average_relative_return_in_rebound_years": 0.0,
        }

    yearly = pd.DataFrame(
        {
            "strategy_return": strategy_ret,
            "benchmark_return": benchmark_ret,
        },
        index=strategy_ret.index,
    )
    yearly_returns = yearly.groupby(yearly.index.year).agg(
        lambda returns: float((1.0 + returns).prod() - 1.0)
    )
    relative = yearly_returns["strategy_return"] - yearly_returns["benchmark_return"]
    years_strategy_beats = int((relative > 0.0).sum())
    worst_relative_year_loss = float(max(-float(relative.min()), 0.0)) if not relative.empty else 0.0

    benchmark_equity = (1.0 + benchmark_ret).cumprod()
    benchmark_drawdown = benchmark_equity / benchmark_equity.cummax() - 1.0
    severe_dd_years = {
        int(year)
        for year, dd in benchmark_drawdown.groupby(benchmark_drawdown.index.year).min().items()
        if float(dd) <= -0.30
    }
    rebound_years = [year + 1 for year in severe_dd_years if year + 1 in yearly_returns.index]
    if rebound_years:
        avg_rebound_relative = float(relative.loc[rebound_years].mean())
    else:
        avg_rebound_relative = 0.0

    return {
        "years_strategy_beats_benchmark": float(years_strategy_beats),
        "worst_relative_year_loss": worst_relative_year_loss,
        "average_relative_return_in_rebound_years": avg_rebound_relative,
    }


def compute_objective_scores(
    strategy_ret: pd.Series,
    benchmark_ret: pd.Series,
    *,
    strategy_equity: Optional[pd.Series] = None,
    benchmark_equity: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    strategy_ret = pd.Series(strategy_ret, copy=True).astype(float)
    benchmark_ret = pd.Series(benchmark_ret, copy=True).astype(float).reindex(strategy_ret.index)
    if benchmark_ret.isna().any():
        raise ValueError("benchmark_ret is missing values on strategy dates.")

    strategy_equity = (
        pd.Series(strategy_equity, copy=True).astype(float).reindex(strategy_ret.index)
        if strategy_equity is not None
        else (1.0 + strategy_ret).cumprod()
    )
    benchmark_equity = (
        pd.Series(benchmark_equity, copy=True).astype(float).reindex(strategy_ret.index)
        if benchmark_equity is not None
        else (1.0 + benchmark_ret).cumprod()
    )
    if strategy_equity.isna().any() or benchmark_equity.isna().any():
        raise ValueError("objective equity inputs are missing values on strategy dates.")

    strategy_final_equity = float(strategy_equity.iloc[-1])
    benchmark_final_equity = float(benchmark_equity.iloc[-1])
    final_equity_ratio = (
        strategy_final_equity / benchmark_final_equity
        if benchmark_final_equity != 0.0
        else np.nan
    )

    strategy_cagr = annualized_return(strategy_equity, periods_per_year)
    benchmark_cagr = annualized_return(benchmark_equity, periods_per_year)
    excess_cagr = (
        strategy_cagr - benchmark_cagr
        if not (np.isnan(strategy_cagr) or np.isnan(benchmark_cagr))
        else np.nan
    )
    strategy_max_dd = max_drawdown(strategy_equity)
    benchmark_max_dd = max_drawdown(benchmark_equity)
    strategy_calmar = calmar_ratio(strategy_cagr, strategy_max_dd)
    dd_guard_penalty = (
        max(abs(strategy_max_dd) - abs(benchmark_max_dd), 0.0)
        if not (np.isnan(strategy_max_dd) or np.isnan(benchmark_max_dd))
        else np.nan
    )
    annual_turnover = _annual_turnover(turnover, periods_per_year)

    if np.isnan(final_equity_ratio):
        raw_survival = np.nan
    elif strategy_final_equity <= benchmark_final_equity:
        raw_survival = final_equity_ratio - 1.0
    else:
        calmar_bonus = 0.0 if np.isnan(strategy_calmar) else strategy_calmar
        raw_survival = final_equity_ratio + 0.25 * calmar_bonus

    yearly_stats = _yearly_relative_stats(strategy_ret, benchmark_ret)
    objective_yearly_consistency = (
        final_equity_ratio
        + 0.10 * yearly_stats["years_strategy_beats_benchmark"]
        - 0.10 * yearly_stats["worst_relative_year_loss"]
    )
    objective_rebound_capture = (
        final_equity_ratio
        + 0.25 * yearly_stats["average_relative_return_in_rebound_years"]
    )

    return {
        "objective_final_ratio": final_equity_ratio,
        "objective_excess_cagr_with_dd_guard": (
            excess_cagr - 0.25 * dd_guard_penalty - 0.02 * annual_turnover
            if not (np.isnan(excess_cagr) or np.isnan(dd_guard_penalty))
            else np.nan
        ),
        "objective_raw_outperformance_with_survival": raw_survival,
        "objective_yearly_consistency": objective_yearly_consistency,
        "objective_rebound_capture": objective_rebound_capture,
        "annual_turnover": annual_turnover,
        **yearly_stats,
    }


def summarize_performance(
    bt: pd.DataFrame,
    benchmark_ret: pd.Series,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    equity = (1.0 + bt["ret"]).cumprod()
    ret = bt["ret"]
    position = bt["position"]
    turnover = bt["turnover"]

    benchmark_ret = pd.Series(benchmark_ret, copy=True).astype(float).reindex(bt.index)
    if benchmark_ret.isna().any():
        raise ValueError("benchmark_ret is missing values on backtest dates.")
    benchmark_equity = (1.0 + benchmark_ret).cumprod()

    cagr = annualized_return(equity, periods_per_year)
    vol = annualized_volatility(ret, periods_per_year)
    sharpe = sharpe_ratio(ret, 0.0, periods_per_year)
    mdd = max_drawdown(equity)
    calmar = calmar_ratio(cagr, mdd)
    bench_cagr = annualized_return(benchmark_equity, periods_per_year)
    bench_vol = annualized_volatility(benchmark_ret, periods_per_year)
    bench_sharpe = sharpe_ratio(benchmark_ret, 0.0, periods_per_year)
    bench_mdd = max_drawdown(benchmark_equity)

    excess_cagr = cagr - bench_cagr if not (np.isnan(cagr) or np.isnan(bench_cagr)) else np.nan
    final_equity_ratio = (
        float(equity.iloc[-1] / benchmark_equity.iloc[-1])
        if benchmark_equity.iloc[-1] != 0.0
        else np.nan
    )
    excess_sharpe = (
        sharpe - bench_sharpe if not (np.isnan(sharpe) or np.isnan(bench_sharpe)) else np.nan
    )
    dd_penalty = (
        max(abs(mdd) - abs(bench_mdd), 0.0)
        if not (np.isnan(mdd) or np.isnan(bench_mdd))
        else np.nan
    )
    excess_score = (
        0.60 * excess_cagr + 0.25 * excess_sharpe - 0.15 * dd_penalty
        if not (np.isnan(excess_cagr) or np.isnan(excess_sharpe) or np.isnan(dd_penalty))
        else np.nan
    )

    objective_scores = compute_objective_scores(
        strategy_ret=ret,
        benchmark_ret=benchmark_ret,
        strategy_equity=equity,
        benchmark_equity=benchmark_equity,
        turnover=turnover,
        periods_per_year=periods_per_year,
    )

    return {
        "final_equity": float(equity.iloc[-1]),
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": mdd,
        "calmar": calmar,
        "final_equity_ratio": final_equity_ratio,
        "excess_cagr": excess_cagr,
        "excess_sharpe": excess_sharpe,
        "dd_penalty": dd_penalty,
        "excess_score": excess_score,
        "trades": int((turnover > 0).sum()),
        "avg_exposure": float(position.mean()),
        "bench_final_equity": float(benchmark_equity.iloc[-1]),
        "bench_cagr": bench_cagr,
        "bench_vol": bench_vol,
        "bench_sharpe": bench_sharpe,
        "bench_max_dd": bench_mdd,
        **objective_scores,
    }


def compute_state_score(
    full_perf: Dict[str, float],
    recent_perf: Dict[str, float],
    min_exposure: float = 0.55,
) -> float:
    full_excess_cagr = full_perf.get("excess_cagr", np.nan)
    full_excess_sharpe = full_perf.get("excess_sharpe", np.nan)
    full_dd_penalty = full_perf.get("dd_penalty", np.nan)
    full_avg_exposure = full_perf.get("avg_exposure", np.nan)
    full_trades = full_perf.get("trades", np.nan)

    recent_excess_cagr = recent_perf.get("excess_cagr", np.nan)
    recent_excess_sharpe = recent_perf.get("excess_sharpe", np.nan)

    vals = [
        full_excess_cagr,
        full_excess_sharpe,
        full_dd_penalty,
        full_avg_exposure,
        full_trades,
        recent_excess_cagr,
        recent_excess_sharpe,
    ]
    if any(np.isnan(v) for v in vals):
        return np.nan

    exposure_penalty = max(min_exposure - full_avg_exposure, 0.0)
    max_dd = full_perf.get("max_dd", np.nan)
    drawdown_over_60 = (
        max(abs(max_dd) - 0.60, 0.0)
        if not np.isnan(max_dd)
        else np.nan
    )
    if np.isnan(drawdown_over_60):
        return np.nan

    trade_penalty = max((full_trades - 140.0) / 220.0, 0.0)

    return float(
        0.45 * full_excess_cagr
        + 0.15 * full_excess_sharpe
        + 0.30 * recent_excess_cagr
        + 0.20 * recent_excess_sharpe
        - 0.10 * full_dd_penalty
        - 0.08 * exposure_penalty
        - 0.08 * drawdown_over_60
        - 0.05 * trade_penalty
    )
