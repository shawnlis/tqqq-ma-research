from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .assets import asset_config_from_row, asset_config_row_fields
from .backtest import (
    backtest_core_overlay_on_window,
    backtest_regime_on_window,
    backtest_rotation_on_window,
    backtest_strategy_on_window,
    backtest_vol_target_on_window,
    make_core_overlay_eval_piece,
    backtest_with_positions,
    make_eval_piece,
    make_regime_eval_piece,
    make_rotation_eval_piece,
    make_vol_target_eval_piece,
    summarize_backtest,
)
from .metrics import compute_state_score, summarize_performance
from .strategies.core_overlay_strategy import CoreOverlayParams
from .strategies.drawdown_governor import DrawdownGovernorParams
from .strategies.ma_strategy import Params
from .strategies.market_internals import DEFAULT_COMPONENTS, MarketInternalsParams
from .strategies.regime_strategy import RegimeParams
from .strategies.rebound import ReboundParams
from .strategies.rotation_strategy import RotationParams
from .strategies.vol_target_strategy import VolTargetParams

logger = logging.getLogger(__name__)


def _bool_from_row_value(value, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _row_to_market_internals_params(row: pd.Series) -> MarketInternalsParams:
    if "use_market_internals" not in row:
        return MarketInternalsParams(use_market_internals=False)

    components_value = row.get("market_internals_components", "")
    if isinstance(components_value, str) and components_value.strip():
        components = tuple(part for part in components_value.split(";") if part)
    else:
        components = DEFAULT_COMPONENTS

    signal_window = row.get("market_internals_signal_window", 100)
    risk_on_threshold = row.get("market_internals_risk_on_threshold", 0.60)
    risk_off_threshold = row.get("market_internals_risk_off_threshold", 0.40)
    score_method = row.get("market_internals_score_method", "binary")
    min_components = row.get("market_internals_min_components", 1)
    vix_risk_on = row.get("vix_risk_on_threshold", 25.0)
    vix_risk_off = row.get("vix_risk_off_threshold", 35.0)

    weights: Dict[str, float] = {}
    weights_value = row.get("market_internals_weights", "")
    if isinstance(weights_value, str) and weights_value.strip():
        for item in weights_value.split(";"):
            if not item or ":" not in item:
                continue
            key, value = item.split(":", 1)
            weights[key] = float(value)

    return MarketInternalsParams(
        use_market_internals=_bool_from_row_value(row.get("use_market_internals", False)),
        signal_window=int(100 if pd.isna(signal_window) else signal_window),
        risk_on_threshold=float(0.60 if pd.isna(risk_on_threshold) else risk_on_threshold),
        risk_off_threshold=float(0.40 if pd.isna(risk_off_threshold) else risk_off_threshold),
        score_method=str(score_method if not pd.isna(score_method) else "binary"),
        components=components,
        weights=weights,
        min_components=int(1 if pd.isna(min_components) else min_components),
        vix_risk_on_threshold=float(25.0 if pd.isna(vix_risk_on) else vix_risk_on),
        vix_risk_off_threshold=float(35.0 if pd.isna(vix_risk_off) else vix_risk_off),
    )


def _market_internals_row_fields(params) -> Dict[str, object]:
    market_internals = getattr(params, "market_internals", MarketInternalsParams())
    weights = ";".join(
        f"{key}:{float(value):g}"
        for key, value in sorted(market_internals.weights.items())
    )
    return {
        "use_market_internals": int(market_internals.use_market_internals),
        "market_internals_signal_window": int(market_internals.signal_window),
        "market_internals_risk_on_threshold": float(market_internals.risk_on_threshold),
        "market_internals_risk_off_threshold": float(market_internals.risk_off_threshold),
        "market_internals_score_method": str(market_internals.score_method),
        "market_internals_components": ";".join(market_internals.components),
        "market_internals_weights": weights,
        "market_internals_min_components": int(market_internals.min_components),
        "vix_risk_on_threshold": float(market_internals.vix_risk_on_threshold),
        "vix_risk_off_threshold": float(market_internals.vix_risk_off_threshold),
    }


def rank_results(df: pd.DataFrame, objective: str) -> pd.DataFrame:
    if objective == "final_equity_ratio":
        sort_cols = [
            col
            for col in ["final_equity_ratio", "excess_cagr", "max_dd"]
            if col in df.columns
        ]
        ascending = [False, False, False][: len(sort_cols)]
        return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    higher_is_better = {
        "cagr",
        "sharpe",
        "calmar",
        "final_equity",
        "final_equity_ratio",
        "excess_cagr",
        "excess_sharpe",
        "excess_score",
        "recent_excess_cagr",
        "recent_excess_sharpe",
        "state_score",
        "objective_final_ratio",
        "objective_excess_cagr_with_dd_guard",
        "objective_raw_outperformance_with_survival",
        "objective_yearly_consistency",
        "objective_rebound_capture",
    }
    lower_is_better = {"max_dd", "vol", "trades", "dd_penalty"}

    if objective in higher_is_better:
        return df.sort_values(objective, ascending=False).reset_index(drop=True)
    if objective in lower_is_better:
        return df.sort_values(objective, ascending=True).reset_index(drop=True)

    raise ValueError(f"Unsupported objective: {objective}")


def _backtest_window_wrapper(args) -> Optional[Dict[str, float]]:
    full_data, params, eval_start, eval_end, transaction_cost_bps, periods_per_year = args
    try:
        return backtest_strategy_on_window(
            full_data=full_data,
            params=params,
            eval_start=eval_start,
            eval_end=eval_end,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=periods_per_year,
            prev_position=None,
        )
    except Exception as e:
        logger.warning(
            "Params %s failed on window %s-%s: %s",
            params, eval_start, eval_end, e
        )
        return None


def log_multiple_testing_warning(grid_size: int, objective: str) -> None:
    logger.warning(
        "Full-sample ranking is based on %d parameter trials (objective=%s). "
        "Top in-sample results are likely inflated by multiple testing. "
        "Treat them as research hints only; rely more on walk-forward and parameter stability.",
        grid_size,
        objective,
    )


def log_workload_hint(grid_size: int, n_wf_windows: int, n_jobs: int) -> None:
    logger.warning(
        "Approx lower-bound workload: %d parameter evaluations before retries/errors. "
        "Actual runtime will be higher because later windows use longer histories. n_jobs=%d.",
        grid_size * (1 + n_wf_windows),
        n_jobs,
    )


def grid_search_on_window(
    full_data: pd.DataFrame,
    grid: List[Params],
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    objective: str = "sharpe",
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    n_jobs: int = 1,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    logger.debug(
        "Windowed grid search | %s to %s | %d parameter sets | objective=%s",
        eval_start.date(),
        eval_end.date(),
        len(grid),
        objective,
    )

    if n_jobs == 1:
        for params in grid:
            try:
                rows.append(
                    backtest_strategy_on_window(
                        full_data=full_data,
                        params=params,
                        eval_start=eval_start,
                        eval_end=eval_end,
                        transaction_cost_bps=transaction_cost_bps,
                        periods_per_year=periods_per_year,
                        prev_position=None,
                    )
                )
            except Exception as e:
                logger.warning(
                    "Params %s failed on window %s-%s: %s",
                    params, eval_start.date(), eval_end.date(), e
                )
                continue
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _backtest_window_wrapper,
                    (full_data, params, eval_start, eval_end, transaction_cost_bps, periods_per_year),
                )
                for params in grid
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    rows.append(result)

    results = pd.DataFrame(rows)
    results = results.replace([np.inf, -np.inf], np.nan).dropna(subset=[objective])

    if results.empty:
        raise ValueError("Windowed grid search returned no valid results.")

    return rank_results(results, objective)


def walk_forward_windows(
    index: pd.DatetimeIndex,
    train_years: int = 5,
    test_years: int = 1,
    mode: str = "rolling",
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    years = sorted(set(index.year))
    windows = []
    mode = str(mode).lower()
    if mode not in {"rolling", "anchored", "expanding", "anchored_expanding"}:
        raise ValueError(f"Unsupported walk-forward mode: {mode}")

    for i in range(train_years, len(years) - test_years + 1):
        train_start_year = years[0] if mode in {"anchored", "expanding", "anchored_expanding"} else years[i - train_years]
        train_end_year = years[i - 1]
        test_start_year = years[i]
        test_end_year = years[i + test_years - 1]

        train_start = pd.Timestamp(f"{train_start_year}-01-01")
        train_end = pd.Timestamp(f"{train_end_year}-12-31")
        test_start = pd.Timestamp(f"{test_start_year}-01-01")
        test_end = pd.Timestamp(f"{test_end_year}-12-31")
        windows.append((train_start, train_end, test_start, test_end))

    return windows


def _row_to_params(row: pd.Series) -> Params:
    return Params(
        short=int(row["short"]),
        long=int(row["long"]),
        ma_type=str(row["ma_type"]),
        signal_asset=str(row["signal_asset"]),
        threshold=float(row["threshold"]),
        cooldown_days=int(row["cooldown_days"]),
        qqq_filter=bool(row["qqq_filter"]),
        qqq_filter_window=int(row["qqq_filter_window"]),
        risk_off_symbol=str(row.get("risk_off_symbol", "CASH")),
        risk_off_weight=float(row.get("risk_off_weight", 0.0)),
        asset_config=asset_config_from_row(row),
    )


def make_eval_position_piece(
    full_data: pd.DataFrame,
    params: Params,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return daily return + position for one parameter set on a window with proper warm-up.
    Costs are not applied here; portfolio-level costs are applied after ensemble aggregation.
    """
    history = full_data.loc[:eval_end].copy()
    bt_full = backtest_with_positions(history, params, transaction_cost_bps=0.0)
    piece = bt_full.loc[eval_start:eval_end, ["daily_ret", "position"]].copy()
    if piece.empty:
        raise ValueError("Evaluation window is empty for ensemble position piece.")
    return piece


def compute_ensemble_weights(objective_values: np.ndarray, power: float = 2.0) -> np.ndarray:
    vals = np.asarray(objective_values, dtype=float)
    if vals.size == 0:
        raise ValueError("No objective values provided for ensemble weights.")

    if np.all(np.isnan(vals)):
        return np.ones_like(vals) / float(vals.size)

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.ones_like(vals) / float(vals.size)

    min_val = float(np.min(finite))
    shifted = np.where(np.isfinite(vals), vals - min_val + 1e-9, 0.0)
    weighted = shifted ** power
    total = float(weighted.sum())
    if total <= 0.0:
        return np.ones_like(vals) / float(vals.size)
    return weighted / total


def walk_forward_search(
    data: pd.DataFrame,
    grid: List[Params],
    train_objective: str = "sharpe",
    transaction_cost_bps: float = 10.0,
    train_years: int = 5,
    test_years: int = 1,
    n_jobs: int = 1,
    window_mode: str = "rolling",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Correct walk-forward logic:
    - Optimize ONLY on training window
    - Pick training rank #1
    - Apply directly to next test window
    - Use full history up to window end for indicator warm-up
    - Carry previous stitched ending position across test-window boundaries
    """
    wf_rows = []
    stitched = []
    prev_position = 0.0
    prev_risk_off_weight = 0.0

    windows = walk_forward_windows(data.index, train_years, test_years, mode=window_mode)
    for train_start, train_end, test_start, test_end in windows:
        train_len = len(data.loc[train_start:train_end])
        test_len = len(data.loc[test_start:test_end])

        if train_len < 252 or test_len < 30:
            logger.info(
                "Skipping window %s-%s => %s-%s due to insufficient data",
                train_start.date(),
                train_end.date(),
                test_start.date(),
                test_end.date(),
            )
            # Intentional: skipping a window does not imply forced flattening.
            continue

        logger.info(
            "WF train %s to %s | test %s to %s",
            train_start.date(),
            train_end.date(),
            test_start.date(),
            test_end.date(),
        )

        train_results = grid_search_on_window(
            full_data=data,
            grid=grid,
            eval_start=train_start,
            eval_end=train_end,
            objective=train_objective,
            transaction_cost_bps=transaction_cost_bps,
            n_jobs=n_jobs,
        )

        best_row = train_results.iloc[0]
        best_params = Params(
            short=int(best_row["short"]),
            long=int(best_row["long"]),
            ma_type=str(best_row["ma_type"]),
            signal_asset=str(best_row["signal_asset"]),
            threshold=float(best_row["threshold"]),
            cooldown_days=int(best_row["cooldown_days"]),
            qqq_filter=bool(best_row["qqq_filter"]),
            qqq_filter_window=int(best_row["qqq_filter_window"]),
            risk_off_symbol=str(best_row.get("risk_off_symbol", "CASH")),
            risk_off_weight=float(best_row.get("risk_off_weight", 0.0)),
            asset_config=asset_config_from_row(best_row),
        )

        test_piece = make_eval_piece(
            full_data=data,
            params=best_params,
            eval_start=test_start,
            eval_end=test_end,
            transaction_cost_bps=transaction_cost_bps,
            prev_position=prev_position,
            prev_risk_off_weight=prev_risk_off_weight,
        )

        benchmark_ret = data[best_params.asset_config.benchmark_symbol].pct_change().fillna(0.0).loc[test_start:test_end]
        test_metrics = summarize_backtest(
            bt=test_piece,
            benchmark_ret=benchmark_ret,
            params=best_params,
        )

        wf_rows.append(
            {
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "short": best_params.short,
                "long": best_params.long,
                "ma_type": best_params.ma_type,
                "signal_asset": best_params.signal_asset,
                "threshold": best_params.threshold,
                "cooldown_days": best_params.cooldown_days,
                "qqq_filter": int(best_params.qqq_filter),
                "qqq_filter_window": best_params.qqq_filter_window,
                "risk_off_symbol": best_params.risk_off_symbol,
                "risk_off_weight": best_params.risk_off_weight,
                **asset_config_row_fields(best_params.asset_config),
                "train_objective": train_objective,
                "train_best_objective": float(best_row[train_objective]),
                "train_best_sharpe": float(best_row["sharpe"]),
                "train_best_cagr": float(best_row["cagr"]),
                "train_best_max_dd": float(best_row["max_dd"]),
                "train_best_excess_score": float(best_row["excess_score"]),
                "test_cagr": float(test_metrics["cagr"]),
                "test_sharpe": float(test_metrics["sharpe"]),
                "test_max_dd": float(test_metrics["max_dd"]),
                "test_calmar": float(test_metrics["calmar"]),
                "test_excess_cagr": float(test_metrics["excess_cagr"]),
                "test_excess_sharpe": float(test_metrics["excess_sharpe"]),
                "test_excess_score": float(test_metrics["excess_score"]),
                "test_final_equity": float(test_metrics["final_equity"]),
            }
        )

        stitched.append(test_piece)
        prev_position = float(
            test_piece["tqqq_weight"].iloc[-1]
            if "tqqq_weight" in test_piece.columns
            else test_piece["position"].iloc[-1]
        )
        prev_risk_off_weight = float(
            test_piece["risk_off_weight"].iloc[-1]
            if "risk_off_weight" in test_piece.columns
            else 0.0
        )

    wf_table = pd.DataFrame(wf_rows)

    if not stitched:
        return wf_table, pd.DataFrame()

    stitched_df = pd.concat(stitched).sort_index()
    stitched_df = stitched_df[~stitched_df.index.duplicated(keep="first")].copy()
    stitched_df["equity"] = (1.0 + stitched_df["ret"]).cumprod()

    return wf_table, stitched_df


def walk_forward_search_ensemble(
    data: pd.DataFrame,
    grid: List[Params],
    train_objective: str = "excess_score",
    transaction_cost_bps: float = 10.0,
    train_years: int = 5,
    test_years: int = 1,
    n_jobs: int = 1,
    top_k: int = 5,
    weight_power: float = 2.0,
    window_mode: str = "rolling",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensemble walk-forward logic:
    - Optimize ONLY on training window
    - Pick training Top-K
    - Convert Top-K objective values into normalized weights
    - Build weighted portfolio position in next test window
    - Apply transaction costs once at portfolio level
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    wf_rows = []
    stitched = []
    prev_portfolio_position = 0.0

    windows = walk_forward_windows(data.index, train_years, test_years, mode=window_mode)
    benchmark_symbol = grid[0].asset_config.benchmark_symbol if grid else "TQQQ"
    full_benchmark_ret = data[benchmark_symbol].pct_change().fillna(0.0)

    for train_start, train_end, test_start, test_end in windows:
        train_len = len(data.loc[train_start:train_end])
        test_len = len(data.loc[test_start:test_end])

        if train_len < 252 or test_len < 30:
            logger.info(
                "Skipping ensemble window %s-%s => %s-%s due to insufficient data",
                train_start.date(),
                train_end.date(),
                test_start.date(),
                test_end.date(),
            )
            continue

        logger.info(
            "WF-ensemble train %s to %s | test %s to %s",
            train_start.date(),
            train_end.date(),
            test_start.date(),
            test_end.date(),
        )

        train_results = grid_search_on_window(
            full_data=data,
            grid=grid,
            eval_start=train_start,
            eval_end=train_end,
            objective=train_objective,
            transaction_cost_bps=transaction_cost_bps,
            n_jobs=n_jobs,
        )

        top = train_results.head(top_k).copy()
        weights = compute_ensemble_weights(
            top[train_objective].to_numpy(dtype=float),
            power=weight_power,
        )

        weighted_positions = []
        daily_ret_ref = None
        member_parts = []

        for i, (_, row) in enumerate(top.iterrows()):
            params_i = _row_to_params(row)
            pos_piece = make_eval_position_piece(
                full_data=data,
                params=params_i,
                eval_start=test_start,
                eval_end=test_end,
            )

            if daily_ret_ref is None:
                daily_ret_ref = pos_piece["daily_ret"].copy()

            weighted_positions.append(pos_piece["position"] * float(weights[i]))
            member_parts.append(
                (
                    f"{float(weights[i]):.4f}|"
                    f"{params_i.short}-{params_i.long}-{params_i.ma_type}-"
                    f"{params_i.signal_asset}-th{params_i.threshold:g}-"
                    f"cd{params_i.cooldown_days}-f{int(params_i.qqq_filter)}"
                )
            )

        if daily_ret_ref is None:
            continue

        position = pd.concat(weighted_positions, axis=1).sum(axis=1).clip(lower=0.0, upper=1.0)
        turnover = position.diff().abs().fillna(0.0)
        first_idx = position.index[0]
        turnover.loc[first_idx] = abs(float(position.loc[first_idx]) - prev_portfolio_position)
        cost = turnover * (transaction_cost_bps / 10000.0)
        ret = position * daily_ret_ref - cost

        test_piece = pd.DataFrame(
            {
                "daily_ret": daily_ret_ref,
                "ret": ret,
                "position": position,
                "turnover": turnover,
            },
            index=daily_ret_ref.index,
        )

        benchmark_ret = full_benchmark_ret.loc[test_start:test_end]
        test_metrics = summarize_performance(
            bt=test_piece,
            benchmark_ret=benchmark_ret,
        )

        top1 = top.iloc[0]
        top1_params = _row_to_params(top1)
        wf_rows.append(
            {
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "ensemble_top_k": int(len(top)),
                "ensemble_weight_power": float(weight_power),
                "ensemble_members": ";".join(member_parts),
                "train_objective": train_objective,
                "train_best_objective": float(top1[train_objective]),
                "train_topk_mean_objective": float(top[train_objective].mean()),
                "train_best_sharpe": float(top1["sharpe"]),
                "train_best_cagr": float(top1["cagr"]),
                "train_best_max_dd": float(top1["max_dd"]),
                "train_best_excess_score": float(top1["excess_score"]),
                "test_cagr": float(test_metrics["cagr"]),
                "test_sharpe": float(test_metrics["sharpe"]),
                "test_max_dd": float(test_metrics["max_dd"]),
                "test_calmar": float(test_metrics["calmar"]),
                "test_excess_cagr": float(test_metrics["excess_cagr"]),
                "test_excess_sharpe": float(test_metrics["excess_sharpe"]),
                "test_excess_score": float(test_metrics["excess_score"]),
                "test_final_equity": float(test_metrics["final_equity"]),
                "test_avg_exposure": float(test_metrics["avg_exposure"]),
                "test_trades": int(test_metrics["trades"]),
                **asset_config_row_fields(top1_params.asset_config),
            }
        )

        stitched.append(test_piece)
        prev_portfolio_position = float(position.iloc[-1])

    wf_table = pd.DataFrame(wf_rows)

    if not stitched:
        return wf_table, pd.DataFrame()

    stitched_df = pd.concat(stitched).sort_index()
    stitched_df = stitched_df[~stitched_df.index.duplicated(keep="first")].copy()
    stitched_df["equity"] = (1.0 + stitched_df["ret"]).cumprod()

    return wf_table, stitched_df


def regime_search_on_window(
    full_data: pd.DataFrame,
    grid: List[RegimeParams],
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    objective: str = "state_score",
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    n_jobs: int = 1,
    recent_days: int = 252,
) -> pd.DataFrame:
    if n_jobs != 1:
        logger.warning("regime_search_on_window currently runs single-threaded; n_jobs is ignored.")

    rows: List[Dict[str, float]] = []
    for params in grid:
        try:
            rows.append(
                backtest_regime_on_window(
                    full_data=full_data,
                    params=params,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    transaction_cost_bps=transaction_cost_bps,
                    periods_per_year=periods_per_year,
                    prev_tqqq_weight=None,
                    prev_qqq_weight=None,
                    recent_days=recent_days,
                )
            )
        except Exception as e:
            logger.warning(
                "Regime params %s failed on window %s-%s: %s",
                params, eval_start.date(), eval_end.date(), e
            )
            continue

    results = pd.DataFrame(rows)
    results = results.replace([np.inf, -np.inf], np.nan).dropna(subset=[objective])
    if results.empty:
        raise ValueError("Regime window search returned no valid results.")

    return rank_results(results, objective)


def _regime_row_to_params(row: pd.Series) -> RegimeParams:
    risk_off_weight = row.get("risk_off_weight", np.nan)
    return RegimeParams(
        trend_window=int(row["trend_window"]),
        momentum_window=int(row["momentum_window"]),
        vol_window=int(row["vol_window"]),
        trend_on=float(row["trend_on"]),
        trend_off=float(row["trend_off"]),
        mom_on=float(row["mom_on"]),
        mom_off=float(row["mom_off"]),
        vol_cap=float(row["vol_cap"]),
        risk_on_leverage=float(row["risk_on_leverage"]),
        risk_off_qqq_position=float(row["risk_off_qqq_position"]),
        transition_position=float(row["transition_position"]),
        min_hold_days=int(row["min_hold_days"]),
        cooldown_days=int(row["cooldown_days"]),
        rebound=_row_to_rebound_params(row),
        governor=_row_to_governor_params(row),
        risk_off_symbol=str(row.get("risk_off_symbol", "QQQ")),
        risk_off_weight=None if pd.isna(risk_off_weight) else float(risk_off_weight),
        market_internals=_row_to_market_internals_params(row),
        asset_config=asset_config_from_row(row),
    )


def _row_to_governor_params(row: pd.Series) -> DrawdownGovernorParams:
    if "use_drawdown_governor" not in row:
        return DrawdownGovernorParams(use_drawdown_governor=False)
    return DrawdownGovernorParams(
        use_drawdown_governor=bool(row["use_drawdown_governor"]),
        portfolio_dd_trigger=float(row["portfolio_dd_trigger"]),
        qqq_dd_trigger=float(row["qqq_dd_trigger"]),
        reduced_exposure=float(row["reduced_exposure"]),
        recovery_ma_window=int(row["recovery_ma_window"]),
        recovery_momentum_window=int(row["recovery_momentum_window"]),
        recovery_momentum_threshold=float(row["recovery_momentum_threshold"]),
        max_days_reduced=int(row["max_days_reduced"]),
    )


def _row_to_rebound_params(row: pd.Series) -> ReboundParams:
    if "use_rebound_module" not in row:
        return ReboundParams(use_rebound_module=False)
    return ReboundParams(
        use_rebound_module=bool(row["use_rebound_module"]),
        rolling_high_window=int(row["rolling_high_window"]),
        drawdown_trigger=float(row["drawdown_trigger"]),
        rebound_momentum_window=int(row["rebound_momentum_window"]),
        rebound_momentum_threshold=float(row["rebound_momentum_threshold"]),
        reclaim_ma_window=int(row["reclaim_ma_window"]),
        rebound_position=float(row["rebound_position"]),
        rebound_hold_days=int(row["rebound_hold_days"]),
        extreme_crash_vol_window=int(row.get("extreme_crash_vol_window", 20)),
        extreme_crash_vol_threshold=float(row.get("extreme_crash_vol_threshold", 0.05)),
    )


def walk_forward_regime_search(
    data: pd.DataFrame,
    grid: List[RegimeParams],
    train_objective: str = "state_score",
    transaction_cost_bps: float = 10.0,
    train_years: int = 5,
    test_years: int = 1,
    n_jobs: int = 1,
    recent_days: int = 252,
    window_mode: str = "rolling",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wf_rows = []
    stitched = []
    prev_tqqq_weight = 0.0
    prev_qqq_weight = 0.0

    windows = walk_forward_windows(data.index, train_years, test_years, mode=window_mode)
    for train_start, train_end, test_start, test_end in windows:
        train_len = len(data.loc[train_start:train_end])
        test_len = len(data.loc[test_start:test_end])
        if train_len < 252 or test_len < 30:
            logger.info(
                "Skipping regime WF window %s-%s => %s-%s due to insufficient data",
                train_start.date(),
                train_end.date(),
                test_start.date(),
                test_end.date(),
            )
            continue

        logger.info(
            "WF-regime train %s to %s | test %s to %s",
            train_start.date(),
            train_end.date(),
            test_start.date(),
            test_end.date(),
        )

        train_results = regime_search_on_window(
            full_data=data,
            grid=grid,
            eval_start=train_start,
            eval_end=train_end,
            objective=train_objective,
            transaction_cost_bps=transaction_cost_bps,
            n_jobs=n_jobs,
            recent_days=recent_days,
        )
        best_row = train_results.iloc[0]
        best_params = _regime_row_to_params(best_row)

        test_piece = make_regime_eval_piece(
            full_data=data,
            params=best_params,
            eval_start=test_start,
            eval_end=test_end,
            transaction_cost_bps=transaction_cost_bps,
            prev_tqqq_weight=prev_tqqq_weight,
            prev_qqq_weight=prev_qqq_weight,
        )
        benchmark_ret = data[best_params.asset_config.benchmark_symbol].pct_change().fillna(0.0).loc[test_start:test_end]
        test_perf = summarize_performance(test_piece, benchmark_ret)

        recent_n = min(recent_days, len(test_piece))
        test_recent_perf = summarize_performance(
            test_piece.tail(recent_n),
            benchmark_ret.tail(recent_n),
        )
        test_state_score = compute_state_score(test_perf, test_recent_perf)

        wf_rows.append(
            {
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "trend_window": best_params.trend_window,
                "momentum_window": best_params.momentum_window,
                "vol_window": best_params.vol_window,
                "trend_on": best_params.trend_on,
                "trend_off": best_params.trend_off,
                "mom_on": best_params.mom_on,
                "mom_off": best_params.mom_off,
                "vol_cap": best_params.vol_cap,
                "risk_on_leverage": best_params.risk_on_leverage,
                "risk_off_qqq_position": best_params.risk_off_qqq_position,
                "risk_off_symbol": best_params.risk_off_symbol,
                "risk_off_weight": best_params.risk_off_weight
                if best_params.risk_off_weight is not None
                else best_params.risk_off_qqq_position,
                **asset_config_row_fields(best_params.asset_config),
                **_market_internals_row_fields(best_params),
                "transition_position": best_params.transition_position,
                "min_hold_days": best_params.min_hold_days,
                "cooldown_days": best_params.cooldown_days,
                "use_rebound_module": int(best_params.rebound.use_rebound_module),
                "rolling_high_window": best_params.rebound.rolling_high_window,
                "drawdown_trigger": best_params.rebound.drawdown_trigger,
                "rebound_momentum_window": best_params.rebound.rebound_momentum_window,
                "rebound_momentum_threshold": best_params.rebound.rebound_momentum_threshold,
                "reclaim_ma_window": best_params.rebound.reclaim_ma_window,
                "rebound_position": best_params.rebound.rebound_position,
                "rebound_hold_days": best_params.rebound.rebound_hold_days,
                "extreme_crash_vol_window": best_params.rebound.extreme_crash_vol_window,
                "extreme_crash_vol_threshold": best_params.rebound.extreme_crash_vol_threshold,
                "use_drawdown_governor": int(best_params.governor.use_drawdown_governor),
                "portfolio_dd_trigger": best_params.governor.portfolio_dd_trigger,
                "qqq_dd_trigger": best_params.governor.qqq_dd_trigger,
                "reduced_exposure": best_params.governor.reduced_exposure,
                "recovery_ma_window": best_params.governor.recovery_ma_window,
                "recovery_momentum_window": best_params.governor.recovery_momentum_window,
                "recovery_momentum_threshold": best_params.governor.recovery_momentum_threshold,
                "max_days_reduced": best_params.governor.max_days_reduced,
                "train_objective": train_objective,
                "train_best_objective": float(best_row[train_objective]),
                "train_best_excess_score": float(best_row["excess_score"]),
                "train_best_recent_excess_cagr": float(best_row["recent_excess_cagr"]),
                "train_best_recent_excess_sharpe": float(best_row["recent_excess_sharpe"]),
                "test_cagr": float(test_perf["cagr"]),
                "test_sharpe": float(test_perf["sharpe"]),
                "test_max_dd": float(test_perf["max_dd"]),
                "test_calmar": float(test_perf["calmar"]),
                "test_excess_cagr": float(test_perf["excess_cagr"]),
                "test_excess_sharpe": float(test_perf["excess_sharpe"]),
                "test_excess_score": float(test_perf["excess_score"]),
                "test_recent_excess_cagr": float(test_recent_perf["excess_cagr"]),
                "test_recent_excess_sharpe": float(test_recent_perf["excess_sharpe"]),
                "test_state_score": float(test_state_score),
                "test_final_equity": float(test_perf["final_equity"]),
                "test_avg_exposure": float(test_perf["avg_exposure"]),
                "test_trades": int(test_perf["trades"]),
            }
        )

        stitched.append(test_piece)
        prev_tqqq_weight = float(test_piece["tqqq_weight"].iloc[-1])
        prev_qqq_weight = float(test_piece["risk_off_weight"].iloc[-1])

    wf_table = pd.DataFrame(wf_rows)
    if not stitched:
        return wf_table, pd.DataFrame()

    stitched_df = pd.concat(stitched).sort_index()
    stitched_df = stitched_df[~stitched_df.index.duplicated(keep="first")].copy()
    stitched_df["equity"] = (1.0 + stitched_df["ret"]).cumprod()
    return wf_table, stitched_df


def rotation_search_on_window(
    full_data: pd.DataFrame,
    grid: List[RotationParams],
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    objective: str = "final_equity_ratio",
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    n_jobs: int = 1,
) -> pd.DataFrame:
    if n_jobs != 1:
        logger.warning("rotation_search_on_window currently runs single-threaded; n_jobs is ignored.")

    rows: List[Dict[str, float]] = []
    for params in grid:
        defensive_asset = str(params.defensive_asset).upper()
        if defensive_asset != "CASH":
            if defensive_asset not in full_data.columns:
                continue
            defensive_prices = full_data.loc[eval_start:eval_end, defensive_asset]
            if defensive_prices.empty or bool(defensive_prices.isna().any()):
                continue
        try:
            rows.append(
                backtest_rotation_on_window(
                    full_data=full_data,
                    params=params,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    transaction_cost_bps=transaction_cost_bps,
                    periods_per_year=periods_per_year,
                    prev_weights=None,
                )
            )
        except Exception as e:
            logger.warning(
                "Rotation params %s failed on window %s-%s: %s",
                params,
                eval_start.date(),
                eval_end.date(),
                e,
            )
            continue

    results = pd.DataFrame(rows)
    results = results.replace([np.inf, -np.inf], np.nan).dropna(subset=[objective])
    if results.empty:
        raise ValueError("Rotation window search returned no valid results.")

    return rank_results(results, objective)


def _rotation_row_to_params(row: pd.Series) -> RotationParams:
    risk_assets_value = str(row.get("risk_assets", "TQQQ;SOXL;UPRO;TECL"))
    risk_assets = tuple(part for part in risk_assets_value.split(";") if part)
    return RotationParams(
        risk_assets=risk_assets,
        rebalance_frequency=str(row["rebalance_frequency"]),
        top_n=int(row["top_n"]),
        max_asset_weight=float(row["max_asset_weight"]),
        max_total_leveraged_exposure=float(row["max_total_leveraged_exposure"]),
        trend_filter_symbol=str(row["trend_filter_symbol"]),
        trend_filter_window=int(row["trend_filter_window"]),
        defensive_asset=str(row["defensive_asset"]),
        require_positive_momentum=_bool_from_row_value(row["require_positive_momentum"]),
        asset_config=asset_config_from_row(row),
    )


def _rotation_param_row(params: RotationParams) -> Dict[str, object]:
    return {
        "risk_assets": ";".join(params.risk_assets),
        "rebalance_frequency": params.rebalance_frequency,
        "top_n": params.top_n,
        "max_asset_weight": params.max_asset_weight,
        "max_total_leveraged_exposure": params.max_total_leveraged_exposure,
        "trend_filter_symbol": params.trend_filter_symbol,
        "trend_filter_window": params.trend_filter_window,
        "defensive_asset": params.defensive_asset,
        "require_positive_momentum": int(params.require_positive_momentum),
        **asset_config_row_fields(params.asset_config),
    }


def walk_forward_rotation_search(
    data: pd.DataFrame,
    grid: List[RotationParams],
    train_objective: str = "final_equity_ratio",
    transaction_cost_bps: float = 10.0,
    train_years: int = 5,
    test_years: int = 1,
    n_jobs: int = 1,
    window_mode: str = "rolling",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wf_rows = []
    stitched = []
    prev_weights: Optional[pd.Series] = None

    windows = walk_forward_windows(data.index, train_years, test_years, mode=window_mode)
    for train_start, train_end, test_start, test_end in windows:
        train_len = len(data.loc[train_start:train_end])
        test_len = len(data.loc[test_start:test_end])
        if train_len < 252 or test_len < 30:
            logger.info(
                "Skipping rotation window %s-%s => %s-%s due to insufficient data",
                train_start.date(),
                train_end.date(),
                test_start.date(),
                test_end.date(),
            )
            continue

        train_results = rotation_search_on_window(
            full_data=data,
            grid=grid,
            eval_start=train_start,
            eval_end=train_end,
            objective=train_objective,
            transaction_cost_bps=transaction_cost_bps,
            n_jobs=n_jobs,
        )
        best_row = train_results.iloc[0]
        best_params = _rotation_row_to_params(best_row)

        test_piece = make_rotation_eval_piece(
            full_data=data,
            params=best_params,
            eval_start=test_start,
            eval_end=test_end,
            transaction_cost_bps=transaction_cost_bps,
            prev_weights=prev_weights,
        )
        benchmark_ret = data[best_params.asset_config.benchmark_symbol].pct_change().fillna(0.0).loc[test_start:test_end]
        test_perf = summarize_performance(test_piece, benchmark_ret)

        wf_rows.append(
            {
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                **_rotation_param_row(best_params),
                "train_objective": train_objective,
                "train_best_objective": float(best_row[train_objective]),
                "train_best_final_equity_ratio": float(best_row["final_equity_ratio"]),
                "train_best_excess_cagr": float(best_row["excess_cagr"]),
                "train_best_max_dd": float(best_row["max_dd"]),
                "test_cagr": float(test_perf["cagr"]),
                "test_sharpe": float(test_perf["sharpe"]),
                "test_max_dd": float(test_perf["max_dd"]),
                "test_calmar": float(test_perf["calmar"]),
                "test_excess_cagr": float(test_perf["excess_cagr"]),
                "test_excess_sharpe": float(test_perf["excess_sharpe"]),
                "test_excess_score": float(test_perf["excess_score"]),
                "test_final_equity": float(test_perf["final_equity"]),
                "test_final_equity_ratio": float(test_perf["final_equity_ratio"]),
                "test_avg_exposure": float(test_perf["avg_exposure"]),
                "test_avg_leveraged_exposure": float(test_piece["leveraged_exposure"].mean()),
                "test_total_turnover": float(test_piece["turnover"].sum()),
                "test_total_cost": float(test_piece["cost"].sum()),
                "test_trades": int(test_perf["trades"]),
            }
        )

        stitched.append(test_piece)
        weight_cols = [column for column in test_piece.columns if column.startswith("weight_")]
        prev_weights = test_piece[weight_cols].iloc[-1].copy()

    wf_table = pd.DataFrame(wf_rows)
    if not stitched:
        return wf_table, pd.DataFrame()

    stitched_df = pd.concat(stitched).sort_index()
    stitched_df = stitched_df[~stitched_df.index.duplicated(keep="first")].copy()
    stitched_df["equity"] = (1.0 + stitched_df["ret"]).cumprod()
    stitched_df["drawdown"] = stitched_df["equity"] / stitched_df["equity"].cummax() - 1.0
    return wf_table, stitched_df


def core_overlay_search_on_window(
    full_data: pd.DataFrame,
    grid: List[CoreOverlayParams],
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    objective: str = "final_equity_ratio",
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    n_jobs: int = 1,
) -> pd.DataFrame:
    if n_jobs != 1:
        logger.warning("core_overlay_search_on_window currently runs single-threaded; n_jobs is ignored.")

    rows: List[Dict[str, float]] = []
    for params in grid:
        try:
            rows.append(
                backtest_core_overlay_on_window(
                    full_data=full_data,
                    params=params,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    transaction_cost_bps=transaction_cost_bps,
                    periods_per_year=periods_per_year,
                    prev_position=None,
                )
            )
        except Exception as e:
            logger.warning(
                "CoreOverlay params %s failed on window %s-%s: %s",
                params,
                eval_start.date(),
                eval_end.date(),
                e,
            )
            continue

    results = pd.DataFrame(rows)
    results = results.replace([np.inf, -np.inf], np.nan).dropna(subset=[objective])
    if results.empty:
        raise ValueError("CoreOverlay window search returned no valid results.")

    return rank_results(results, objective)


def _core_overlay_row_to_params(row: pd.Series) -> CoreOverlayParams:
    return CoreOverlayParams(
        core_exposure=float(row["core_exposure"]),
        overlay_max=float(row["overlay_max"]),
        max_total_exposure=float(row["max_total_exposure"]),
        trend_window=int(row["trend_window"]),
        fast_trend_window=int(row["fast_trend_window"]),
        momentum_window=int(row["momentum_window"]),
        vol_window=int(row["vol_window"]),
        vol_cap=float(row["vol_cap"]),
        crash_cut_exposure=float(row["crash_cut_exposure"]),
        rebound_boost=bool(row["rebound_boost"]),
        rebound=_row_to_rebound_params(row),
        governor=_row_to_governor_params(row),
        risk_off_symbol=str(row.get("risk_off_symbol", "CASH")),
        risk_off_weight=float(row.get("risk_off_weight", 0.0)),
        market_internals=_row_to_market_internals_params(row),
        asset_config=asset_config_from_row(row),
    )


def walk_forward_core_overlay_search(
    data: pd.DataFrame,
    grid: List[CoreOverlayParams],
    train_objective: str = "final_equity_ratio",
    transaction_cost_bps: float = 10.0,
    train_years: int = 5,
    test_years: int = 1,
    n_jobs: int = 1,
    window_mode: str = "rolling",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wf_rows = []
    stitched = []
    prev_position = 0.0
    prev_risk_off_weight = 0.0

    windows = walk_forward_windows(data.index, train_years, test_years, mode=window_mode)
    for train_start, train_end, test_start, test_end in windows:
        train_len = len(data.loc[train_start:train_end])
        test_len = len(data.loc[test_start:test_end])
        if train_len < 252 or test_len < 30:
            logger.info(
                "Skipping CoreOverlay window %s-%s => %s-%s due to insufficient data",
                train_start.date(),
                train_end.date(),
                test_start.date(),
                test_end.date(),
            )
            continue

        logger.info(
            "WF-CoreOverlay train %s to %s | test %s to %s",
            train_start.date(),
            train_end.date(),
            test_start.date(),
            test_end.date(),
        )

        train_results = core_overlay_search_on_window(
            full_data=data,
            grid=grid,
            eval_start=train_start,
            eval_end=train_end,
            objective=train_objective,
            transaction_cost_bps=transaction_cost_bps,
            n_jobs=n_jobs,
        )
        best_row = train_results.iloc[0]
        best_params = _core_overlay_row_to_params(best_row)

        test_piece = make_core_overlay_eval_piece(
            full_data=data,
            params=best_params,
            eval_start=test_start,
            eval_end=test_end,
            transaction_cost_bps=transaction_cost_bps,
            prev_position=prev_position,
            prev_risk_off_weight=prev_risk_off_weight,
        )
        benchmark_ret = data[best_params.asset_config.benchmark_symbol].pct_change().fillna(0.0).loc[test_start:test_end]
        test_perf = summarize_performance(test_piece, benchmark_ret)

        wf_rows.append(
            {
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "core_exposure": best_params.core_exposure,
                "overlay_max": best_params.overlay_max,
                "max_total_exposure": best_params.max_total_exposure,
                "trend_window": best_params.trend_window,
                "fast_trend_window": best_params.fast_trend_window,
                "momentum_window": best_params.momentum_window,
                "vol_window": best_params.vol_window,
                "vol_cap": best_params.vol_cap,
                "crash_cut_exposure": best_params.crash_cut_exposure,
                "rebound_boost": int(best_params.rebound_boost),
                "use_rebound_module": int(best_params.rebound.use_rebound_module),
                "rolling_high_window": best_params.rebound.rolling_high_window,
                "drawdown_trigger": best_params.rebound.drawdown_trigger,
                "rebound_momentum_window": best_params.rebound.rebound_momentum_window,
                "rebound_momentum_threshold": best_params.rebound.rebound_momentum_threshold,
                "reclaim_ma_window": best_params.rebound.reclaim_ma_window,
                "rebound_position": best_params.rebound.rebound_position,
                "rebound_hold_days": best_params.rebound.rebound_hold_days,
                "extreme_crash_vol_window": best_params.rebound.extreme_crash_vol_window,
                "extreme_crash_vol_threshold": best_params.rebound.extreme_crash_vol_threshold,
                "use_drawdown_governor": int(best_params.governor.use_drawdown_governor),
                "portfolio_dd_trigger": best_params.governor.portfolio_dd_trigger,
                "qqq_dd_trigger": best_params.governor.qqq_dd_trigger,
                "reduced_exposure": best_params.governor.reduced_exposure,
                "recovery_ma_window": best_params.governor.recovery_ma_window,
                "recovery_momentum_window": best_params.governor.recovery_momentum_window,
                "recovery_momentum_threshold": best_params.governor.recovery_momentum_threshold,
                "max_days_reduced": best_params.governor.max_days_reduced,
                "risk_off_symbol": best_params.risk_off_symbol,
                "risk_off_weight": best_params.risk_off_weight,
                **asset_config_row_fields(best_params.asset_config),
                **_market_internals_row_fields(best_params),
                "train_objective": train_objective,
                "train_best_objective": float(best_row[train_objective]),
                "train_best_final_equity_ratio": float(best_row["final_equity_ratio"]),
                "train_best_excess_cagr": float(best_row["excess_cagr"]),
                "train_best_max_dd": float(best_row["max_dd"]),
                "test_cagr": float(test_perf["cagr"]),
                "test_sharpe": float(test_perf["sharpe"]),
                "test_max_dd": float(test_perf["max_dd"]),
                "test_calmar": float(test_perf["calmar"]),
                "test_excess_cagr": float(test_perf["excess_cagr"]),
                "test_excess_sharpe": float(test_perf["excess_sharpe"]),
                "test_excess_score": float(test_perf["excess_score"]),
                "test_final_equity": float(test_perf["final_equity"]),
                "test_final_equity_ratio": float(test_perf["final_equity_ratio"]),
                "test_avg_exposure": float(test_perf["avg_exposure"]),
                "test_trades": int(test_perf["trades"]),
            }
        )

        stitched.append(test_piece)
        prev_position = float(
            test_piece["tqqq_weight"].iloc[-1]
            if "tqqq_weight" in test_piece.columns
            else test_piece["position"].iloc[-1]
        )
        prev_risk_off_weight = float(
            test_piece["risk_off_weight"].iloc[-1]
            if "risk_off_weight" in test_piece.columns
            else 0.0
        )

    wf_table = pd.DataFrame(wf_rows)
    if not stitched:
        return wf_table, pd.DataFrame()

    stitched_df = pd.concat(stitched).sort_index()
    stitched_df = stitched_df[~stitched_df.index.duplicated(keep="first")].copy()
    stitched_df["equity"] = (1.0 + stitched_df["ret"]).cumprod()
    return wf_table, stitched_df


def vol_target_search_on_window(
    full_data: pd.DataFrame,
    grid: List[VolTargetParams],
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    objective: str = "final_equity_ratio",
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    n_jobs: int = 1,
) -> pd.DataFrame:
    if n_jobs != 1:
        logger.warning("vol_target_search_on_window currently runs single-threaded; n_jobs is ignored.")

    rows: List[Dict[str, float]] = []
    for params in grid:
        try:
            rows.append(
                backtest_vol_target_on_window(
                    full_data=full_data,
                    params=params,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    transaction_cost_bps=transaction_cost_bps,
                    periods_per_year=periods_per_year,
                    prev_position=None,
                )
            )
        except Exception as e:
            logger.warning(
                "VolTarget params %s failed on window %s-%s: %s",
                params,
                eval_start.date(),
                eval_end.date(),
                e,
            )
            continue

    results = pd.DataFrame(rows)
    results = results.replace([np.inf, -np.inf], np.nan).dropna(subset=[objective])
    if results.empty:
        raise ValueError("VolTarget window search returned no valid results.")

    return rank_results(results, objective)


def _vol_target_row_to_params(row: pd.Series) -> VolTargetParams:
    return VolTargetParams(
        target_ann_vol=float(row["target_ann_vol"]),
        realized_vol_window=int(row["realized_vol_window"]),
        min_exposure=float(row["min_exposure"]),
        max_exposure=float(row["max_exposure"]),
        trend_window=int(row["trend_window"]),
        momentum_window=int(row["momentum_window"]),
        trend_multiplier_below_ma=float(row["trend_multiplier_below_ma"]),
        momentum_boost=float(row["momentum_boost"]),
        crash_vol_cutoff=float(row["crash_vol_cutoff"]),
        crash_exposure=float(row["crash_exposure"]),
        rebound=_row_to_rebound_params(row),
        governor=_row_to_governor_params(row),
        risk_off_symbol=str(row.get("risk_off_symbol", "CASH")),
        risk_off_weight=float(row.get("risk_off_weight", 0.0)),
        market_internals=_row_to_market_internals_params(row),
        asset_config=asset_config_from_row(row),
    )


def walk_forward_vol_target_search(
    data: pd.DataFrame,
    grid: List[VolTargetParams],
    train_objective: str = "final_equity_ratio",
    transaction_cost_bps: float = 10.0,
    train_years: int = 5,
    test_years: int = 1,
    n_jobs: int = 1,
    window_mode: str = "rolling",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wf_rows = []
    stitched = []
    prev_position = 0.0
    prev_risk_off_weight = 0.0

    windows = walk_forward_windows(data.index, train_years, test_years, mode=window_mode)
    for train_start, train_end, test_start, test_end in windows:
        train_len = len(data.loc[train_start:train_end])
        test_len = len(data.loc[test_start:test_end])
        if train_len < 252 or test_len < 30:
            logger.info(
                "Skipping VolTarget window %s-%s => %s-%s due to insufficient data",
                train_start.date(),
                train_end.date(),
                test_start.date(),
                test_end.date(),
            )
            continue

        logger.info(
            "WF-VolTarget train %s to %s | test %s to %s",
            train_start.date(),
            train_end.date(),
            test_start.date(),
            test_end.date(),
        )

        train_results = vol_target_search_on_window(
            full_data=data,
            grid=grid,
            eval_start=train_start,
            eval_end=train_end,
            objective=train_objective,
            transaction_cost_bps=transaction_cost_bps,
            n_jobs=n_jobs,
        )
        best_row = train_results.iloc[0]
        best_params = _vol_target_row_to_params(best_row)

        test_piece = make_vol_target_eval_piece(
            full_data=data,
            params=best_params,
            eval_start=test_start,
            eval_end=test_end,
            transaction_cost_bps=transaction_cost_bps,
            prev_position=prev_position,
            prev_risk_off_weight=prev_risk_off_weight,
        )
        benchmark_ret = data[best_params.asset_config.benchmark_symbol].pct_change().fillna(0.0).loc[test_start:test_end]
        test_perf = summarize_performance(test_piece, benchmark_ret)

        wf_rows.append(
            {
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "target_ann_vol": best_params.target_ann_vol,
                "realized_vol_window": best_params.realized_vol_window,
                "min_exposure": best_params.min_exposure,
                "max_exposure": best_params.max_exposure,
                "trend_window": best_params.trend_window,
                "momentum_window": best_params.momentum_window,
                "trend_multiplier_below_ma": best_params.trend_multiplier_below_ma,
                "momentum_boost": best_params.momentum_boost,
                "crash_vol_cutoff": best_params.crash_vol_cutoff,
                "crash_exposure": best_params.crash_exposure,
                "use_rebound_module": int(best_params.rebound.use_rebound_module),
                "rolling_high_window": best_params.rebound.rolling_high_window,
                "drawdown_trigger": best_params.rebound.drawdown_trigger,
                "rebound_momentum_window": best_params.rebound.rebound_momentum_window,
                "rebound_momentum_threshold": best_params.rebound.rebound_momentum_threshold,
                "reclaim_ma_window": best_params.rebound.reclaim_ma_window,
                "rebound_position": best_params.rebound.rebound_position,
                "rebound_hold_days": best_params.rebound.rebound_hold_days,
                "extreme_crash_vol_window": best_params.rebound.extreme_crash_vol_window,
                "extreme_crash_vol_threshold": best_params.rebound.extreme_crash_vol_threshold,
                "use_drawdown_governor": int(best_params.governor.use_drawdown_governor),
                "portfolio_dd_trigger": best_params.governor.portfolio_dd_trigger,
                "qqq_dd_trigger": best_params.governor.qqq_dd_trigger,
                "reduced_exposure": best_params.governor.reduced_exposure,
                "recovery_ma_window": best_params.governor.recovery_ma_window,
                "recovery_momentum_window": best_params.governor.recovery_momentum_window,
                "recovery_momentum_threshold": best_params.governor.recovery_momentum_threshold,
                "max_days_reduced": best_params.governor.max_days_reduced,
                "risk_off_symbol": best_params.risk_off_symbol,
                "risk_off_weight": best_params.risk_off_weight,
                **asset_config_row_fields(best_params.asset_config),
                **_market_internals_row_fields(best_params),
                "train_objective": train_objective,
                "train_best_objective": float(best_row[train_objective]),
                "train_best_final_equity_ratio": float(best_row["final_equity_ratio"]),
                "train_best_excess_cagr": float(best_row["excess_cagr"]),
                "train_best_max_dd": float(best_row["max_dd"]),
                "train_best_avg_target_exposure": float(best_row["avg_target_exposure"]),
                "train_best_avg_realized_ann_vol": float(best_row["avg_realized_ann_vol"]),
                "test_cagr": float(test_perf["cagr"]),
                "test_sharpe": float(test_perf["sharpe"]),
                "test_max_dd": float(test_perf["max_dd"]),
                "test_calmar": float(test_perf["calmar"]),
                "test_excess_cagr": float(test_perf["excess_cagr"]),
                "test_excess_sharpe": float(test_perf["excess_sharpe"]),
                "test_excess_score": float(test_perf["excess_score"]),
                "test_final_equity": float(test_perf["final_equity"]),
                "test_final_equity_ratio": float(test_perf["final_equity_ratio"]),
                "test_avg_exposure": float(test_perf["avg_exposure"]),
                "test_avg_target_exposure": float(test_piece["target_exposure"].mean()),
                "test_avg_realized_ann_vol": float(test_piece["realized_ann_vol"].mean()),
                "test_total_cost": float(test_piece["cost"].sum()),
                "test_trades": int(test_perf["trades"]),
            }
        )

        stitched.append(test_piece)
        prev_position = float(
            test_piece["tqqq_weight"].iloc[-1]
            if "tqqq_weight" in test_piece.columns
            else test_piece["position"].iloc[-1]
        )
        prev_risk_off_weight = float(
            test_piece["risk_off_weight"].iloc[-1]
            if "risk_off_weight" in test_piece.columns
            else 0.0
        )

    wf_table = pd.DataFrame(wf_rows)
    if not stitched:
        return wf_table, pd.DataFrame()

    stitched_df = pd.concat(stitched).sort_index()
    stitched_df = stitched_df[~stitched_df.index.duplicated(keep="first")].copy()
    stitched_df["equity"] = (1.0 + stitched_df["ret"]).cumprod()
    stitched_df["drawdown"] = stitched_df["equity"] / stitched_df["equity"].cummax() - 1.0
    return wf_table, stitched_df
