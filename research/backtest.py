from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .assets import AssetConfig, asset_config_row_fields
from .execution import (
    asset_returns,
    execution_model_from_data,
    resolve_execution_model,
)
from .metrics import compute_state_score, summarize_performance
from .strategies.core_overlay_strategy import CoreOverlayParams, build_core_overlay_signal
from .strategies.ma_strategy import Params, build_signal
from .strategies.market_internals import build_market_internals_signal
from .strategies.regime_strategy import RegimeParams, build_regime_weights
from .strategies.rotation_strategy import RotationParams, build_rotation_signal
from .strategies.vol_target_strategy import VolTargetParams, build_vol_target_signal


def _risk_off_config(params, default_symbol: str = "CASH") -> tuple[str, float]:
    symbol = str(getattr(params, "risk_off_symbol", default_symbol) or default_symbol).upper()
    weight = getattr(params, "risk_off_weight", 0.0)
    if weight is None:
        weight = 0.0
    return symbol, float(np.clip(float(weight), 0.0, 1.0))


def _asset_config(params) -> AssetConfig:
    asset_config = getattr(params, "asset_config", AssetConfig())
    return asset_config.normalized()


def _benchmark_ret(full_data: pd.DataFrame, params, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.Series:
    benchmark_symbol = _asset_config(params).benchmark_symbol
    if benchmark_symbol not in full_data.columns:
        raise ValueError(f"Benchmark symbol is missing from data: {benchmark_symbol}")
    return full_data[benchmark_symbol].pct_change().fillna(0.0).loc[eval_start:eval_end]


def _asset_fields(params) -> Dict[str, str]:
    return asset_config_row_fields(_asset_config(params))


def _risk_off_returns(data: pd.DataFrame, symbol: str) -> pd.Series:
    if symbol not in data.columns:
        raise ValueError(f"Risk-off symbol is missing from price data: {symbol}")
    return asset_returns(data, symbol, execution_model_from_data(data))


def _market_internals_enabled(params) -> bool:
    market_internals = getattr(params, "market_internals", None)
    return bool(market_internals is not None and market_internals.use_market_internals)


def _attach_market_internals_columns(
    out: pd.DataFrame,
    data: pd.DataFrame,
    params,
) -> pd.DataFrame:
    if not _market_internals_enabled(params):
        return out
    diagnostics = build_market_internals_signal(
        data,
        params.market_internals,
    ).reindex(out.index)
    for column in diagnostics.columns:
        out[column] = diagnostics[column]
    return out


def _market_internals_param_fields(params) -> Dict[str, object]:
    market_internals = getattr(params, "market_internals", None)
    if market_internals is None:
        return {
            "use_market_internals": 0,
            "market_internals_signal_window": np.nan,
            "market_internals_risk_on_threshold": np.nan,
            "market_internals_risk_off_threshold": np.nan,
            "market_internals_score_method": "",
            "market_internals_components": "",
            "market_internals_weights": "",
            "market_internals_min_components": np.nan,
            "vix_risk_on_threshold": np.nan,
            "vix_risk_off_threshold": np.nan,
        }
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


def _call_core_overlay_signal(
    tqqq: pd.Series,
    qqq: pd.Series,
    params: CoreOverlayParams,
    data: pd.DataFrame,
    filter_close: Optional[pd.Series] = None,
) -> pd.Series:
    try:
        return build_core_overlay_signal(tqqq, qqq, params, price_data=data, filter_close=filter_close)
    except TypeError as exc:
        if "price_data" not in str(exc) and "filter_close" not in str(exc):
            raise
        return build_core_overlay_signal(tqqq, qqq, params)


def _call_ma_signal(
    trade_close: pd.Series,
    signal_close: pd.Series,
    params: Params,
    filter_close: Optional[pd.Series] = None,
) -> pd.Series:
    try:
        return build_signal(trade_close, signal_close, params, filter_close=filter_close)
    except TypeError as exc:
        if "filter_close" not in str(exc):
            raise
        return build_signal(trade_close, signal_close, params)


def _portfolio_from_trade_weight(
    data: pd.DataFrame,
    trade_weight: pd.Series,
    transaction_cost_bps: float,
    risk_off_symbol: str,
    risk_off_weight: float,
    explicit_risk_off_weight: Optional[pd.Series] = None,
    trade_asset: str = "TQQQ",
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    trade_asset = str(trade_asset).upper()
    if trade_asset not in data.columns:
        raise ValueError(f"Trade asset is missing from data: {trade_asset}")
    trade_prices = data[trade_asset].copy()
    trade_weight = pd.Series(trade_weight, copy=True).astype(float).reindex(data.index).fillna(0.0)
    risk_off_symbol = str(risk_off_symbol).upper()
    risk_off_weight = float(np.clip(risk_off_weight, 0.0, 1.0))

    if explicit_risk_off_weight is None:
        unused_capital = (1.0 - trade_weight).clip(lower=0.0)
        risk_weight = unused_capital * risk_off_weight
    else:
        risk_weight = (
            pd.Series(explicit_risk_off_weight, copy=True)
            .astype(float)
            .reindex(data.index)
            .fillna(0.0)
        )

    requested_execution_model = execution_model_from_data(data, execution_model)
    required_execution_symbols = [trade_asset]
    if risk_off_symbol != "CASH" and (
        explicit_risk_off_weight is not None or float(risk_weight.abs().sum()) > 0.0
    ):
        required_execution_symbols.append(risk_off_symbol)
    effective_execution_model, execution_warning = resolve_execution_model(
        data,
        requested_execution_model,
        required_execution_symbols,
    )

    daily_ret_trade = asset_returns(data, trade_asset, effective_execution_model)
    if float(risk_weight.abs().sum()) == 0.0 and risk_off_symbol not in data.columns:
        daily_ret_risk_off = pd.Series(0.0, index=data.index, name="daily_ret_risk_off")
    else:
        daily_ret_risk_off = asset_returns(data, risk_off_symbol, effective_execution_model)
    gross_ret = trade_weight * daily_ret_trade + risk_weight * daily_ret_risk_off

    turnover_trade = trade_weight.diff().abs().fillna(trade_weight.abs())
    turnover_risk_off = risk_weight.diff().abs().fillna(risk_weight.abs())
    turnover = turnover_trade + turnover_risk_off
    cost = turnover * (transaction_cost_bps / 10000.0)
    ret = gross_ret - cost
    trade_col = trade_asset.lower()

    return pd.DataFrame(
        {
            "daily_ret": daily_ret_trade,
            "daily_ret_trade": daily_ret_trade,
            "daily_ret_tqqq": daily_ret_trade,
            f"daily_ret_{trade_col}": daily_ret_trade,
            "daily_ret_risk_off": daily_ret_risk_off,
            "gross_ret": gross_ret,
            "ret": ret,
            "position": (trade_weight + risk_weight).rename("position"),
            "trade_asset": trade_asset,
            "trade_weight": trade_weight,
            "tqqq_weight": trade_weight,
            f"{trade_col}_weight": trade_weight,
            "risk_off_weight": risk_weight,
            "risk_off_symbol": risk_off_symbol,
            "requested_execution_model": requested_execution_model,
            "execution_model": effective_execution_model,
            "execution_model_warning": execution_warning,
            "turnover_trade": turnover_trade,
            "turnover_tqqq": turnover_trade,
            f"turnover_{trade_col}": turnover_trade,
            "turnover_risk_off": turnover_risk_off,
            "turnover": turnover,
            "cost": cost,
        },
        index=data.index,
    )


def _portfolio_from_tqqq_weight(
    data: pd.DataFrame,
    tqqq_weight: pd.Series,
    transaction_cost_bps: float,
    risk_off_symbol: str,
    risk_off_weight: float,
    explicit_risk_off_weight: Optional[pd.Series] = None,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    return _portfolio_from_trade_weight(
        data=data,
        trade_weight=tqqq_weight,
        transaction_cost_bps=transaction_cost_bps,
        risk_off_symbol=risk_off_symbol,
        risk_off_weight=risk_off_weight,
        explicit_risk_off_weight=explicit_risk_off_weight,
        trade_asset="TQQQ",
        execution_model=execution_model,
    )


def _correct_first_bar_for_previous_weights(
    piece: pd.DataFrame,
    bt_full: pd.DataFrame,
    transaction_cost_bps: float,
    prev_tqqq_weight: Optional[float],
    prev_risk_off_weight: Optional[float],
) -> pd.DataFrame:
    if prev_tqqq_weight is None:
        return piece
    if prev_risk_off_weight is None:
        prev_risk_off_weight = 0.0

    first_idx = piece.index[0]
    desired_tqqq = float(piece.at[first_idx, "tqqq_weight"])
    desired_risk_off = float(piece.at[first_idx, "risk_off_weight"])
    turnover_tqqq0 = abs(desired_tqqq - float(prev_tqqq_weight))
    turnover_risk_off0 = abs(desired_risk_off - float(prev_risk_off_weight))
    turnover0 = turnover_tqqq0 + turnover_risk_off0
    cost0 = turnover0 * (transaction_cost_bps / 10000.0)
    gross_ret0 = (
        desired_tqqq * float(bt_full.at[first_idx, "daily_ret_tqqq"])
        + desired_risk_off * float(bt_full.at[first_idx, "daily_ret_risk_off"])
    )

    piece.at[first_idx, "turnover_tqqq"] = turnover_tqqq0
    piece.at[first_idx, "turnover_risk_off"] = turnover_risk_off0
    piece.at[first_idx, "turnover"] = turnover0
    piece.at[first_idx, "cost"] = cost0
    piece.at[first_idx, "gross_ret"] = gross_ret0
    piece.at[first_idx, "ret"] = gross_ret0 - cost0
    return piece


def backtest_with_positions(
    data: pd.DataFrame,
    params: Params,
    transaction_cost_bps: float = 10.0,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assumptions:
    - signal formed at close of day t
    - position takes effect on day t+1 (shift(1)) => avoids look-ahead
    - execution asset is configured by params.asset_config.trade_asset
    - transaction cost charged whenever position changes
    """
    asset_config = _asset_config(params)
    tqqq = data[asset_config.trade_asset].copy()
    qqq = data[asset_config.primary_signal_asset].copy()
    filter_close = data[asset_config.secondary_filter_asset].copy()

    signal = _call_ma_signal(tqqq, qqq, params, filter_close=filter_close)
    position = signal.shift(1).fillna(0.0)
    risk_off_symbol, risk_off_weight = _risk_off_config(params)

    out = _portfolio_from_trade_weight(
        data=data,
        trade_weight=position,
        transaction_cost_bps=transaction_cost_bps,
        risk_off_symbol=risk_off_symbol,
        risk_off_weight=risk_off_weight,
        trade_asset=asset_config.trade_asset,
        execution_model=execution_model,
    )
    return out


def summarize_backtest(
    bt: pd.DataFrame,
    benchmark_ret: pd.Series,
    params: Params,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    perf = summarize_performance(
        bt=bt,
        benchmark_ret=benchmark_ret,
        periods_per_year=periods_per_year,
    )

    return {
        "short": params.short,
        "long": params.long,
        "ma_type": params.ma_type,
        "signal_asset": params.signal_asset,
        "threshold": params.threshold,
        "cooldown_days": params.cooldown_days,
        "qqq_filter": int(params.qqq_filter),
        "qqq_filter_window": params.qqq_filter_window,
        "risk_off_symbol": params.risk_off_symbol,
        "risk_off_weight": params.risk_off_weight,
        **_asset_fields(params),
        **perf,
    }


def make_eval_piece(
    full_data: pd.DataFrame,
    params: Params,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    prev_position: Optional[float] = None,
    prev_risk_off_weight: Optional[float] = None,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a window piece using full history up to eval_end for proper indicator warm-up.

    If prev_position is provided, correct the first bar turnover/cost so the
    window is stitched to an external prior position.
    """
    history = full_data.loc[:eval_end].copy()
    bt_full = backtest_with_positions(
        history,
        params,
        transaction_cost_bps,
        execution_model=execution_model,
    )

    piece = bt_full.loc[eval_start:eval_end].copy()
    if piece.empty:
        raise ValueError("Evaluation window is empty.")

    if prev_position is not None:
        piece = _correct_first_bar_for_previous_weights(
            piece=piece,
            bt_full=bt_full,
            transaction_cost_bps=transaction_cost_bps,
            prev_tqqq_weight=prev_position,
            prev_risk_off_weight=prev_risk_off_weight,
        )

    return piece


def backtest_strategy_on_window(
    full_data: pd.DataFrame,
    params: Params,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    prev_position: Optional[float] = None,
    prev_risk_off_weight: Optional[float] = None,
    execution_model: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a parameter set on a window, using full history up to eval_end for warm-up.
    """
    piece = make_eval_piece(
        full_data=full_data,
        params=params,
        eval_start=eval_start,
        eval_end=eval_end,
        transaction_cost_bps=transaction_cost_bps,
        prev_position=prev_position,
        prev_risk_off_weight=prev_risk_off_weight,
        execution_model=execution_model,
    )

    benchmark_ret = _benchmark_ret(full_data, params, eval_start, eval_end)

    return summarize_backtest(
        bt=piece,
        benchmark_ret=benchmark_ret,
        params=params,
        periods_per_year=periods_per_year,
    )


def backtest_regime_with_positions(
    data: pd.DataFrame,
    params: RegimeParams,
    transaction_cost_bps: float = 10.0,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    asset_config = _asset_config(params)
    tqqq = data[asset_config.trade_asset].copy()
    qqq = data[asset_config.primary_signal_asset].copy()
    filter_close = data[asset_config.secondary_filter_asset].copy()

    weights = build_regime_weights(tqqq, qqq, params, price_data=data, filter_close=filter_close)
    tqqq_weight = weights["tqqq_weight"].shift(1).fillna(0.0)
    risk_off_weight = weights["risk_off_weight"].shift(1).fillna(0.0)
    risk_off_symbol = str(params.risk_off_symbol).upper()

    out = _portfolio_from_trade_weight(
        data=data,
        trade_weight=tqqq_weight,
        transaction_cost_bps=transaction_cost_bps,
        risk_off_symbol=risk_off_symbol,
        risk_off_weight=0.0,
        explicit_risk_off_weight=risk_off_weight,
        trade_asset=asset_config.trade_asset,
        execution_model=execution_model,
    )
    out["qqq_weight"] = weights["qqq_weight"].shift(1).fillna(0.0)
    effective_execution_model = str(out["execution_model"].iloc[0]) if not out.empty else execution_model_from_data(data)
    out["daily_ret_qqq"] = asset_returns(data, asset_config.primary_signal_asset, effective_execution_model)
    out[f"daily_ret_{asset_config.primary_signal_asset.lower()}"] = asset_returns(
        data,
        asset_config.primary_signal_asset,
        effective_execution_model,
    )
    out = _attach_market_internals_columns(out, data, params)
    return out


def make_regime_eval_piece(
    full_data: pd.DataFrame,
    params: RegimeParams,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    prev_tqqq_weight: Optional[float] = None,
    prev_qqq_weight: Optional[float] = None,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    history = full_data.loc[:eval_end].copy()
    bt_full = backtest_regime_with_positions(
        history,
        params,
        transaction_cost_bps,
        execution_model=execution_model,
    )

    piece = bt_full.loc[eval_start:eval_end].copy()
    if piece.empty:
        raise ValueError("Evaluation window is empty for regime strategy.")

    if prev_tqqq_weight is not None and prev_qqq_weight is not None:
        piece = _correct_first_bar_for_previous_weights(
            piece=piece,
            bt_full=bt_full,
            transaction_cost_bps=transaction_cost_bps,
            prev_tqqq_weight=prev_tqqq_weight,
            prev_risk_off_weight=prev_qqq_weight,
        )

    return piece


def backtest_regime_on_window(
    full_data: pd.DataFrame,
    params: RegimeParams,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    prev_tqqq_weight: Optional[float] = None,
    prev_qqq_weight: Optional[float] = None,
    recent_days: int = 252,
    execution_model: Optional[str] = None,
) -> Dict[str, float]:
    piece = make_regime_eval_piece(
        full_data=full_data,
        params=params,
        eval_start=eval_start,
        eval_end=eval_end,
        transaction_cost_bps=transaction_cost_bps,
        prev_tqqq_weight=prev_tqqq_weight,
        prev_qqq_weight=prev_qqq_weight,
        execution_model=execution_model,
    )

    benchmark_ret = _benchmark_ret(full_data, params, eval_start, eval_end)
    full_perf = summarize_performance(piece, benchmark_ret, periods_per_year)

    recent_n = min(recent_days, len(piece))
    recent_piece = piece.tail(recent_n).copy()
    recent_benchmark = benchmark_ret.tail(recent_n).copy()
    recent_perf = summarize_performance(recent_piece, recent_benchmark, periods_per_year)
    state_score = compute_state_score(full_perf, recent_perf)

    return {
        "trend_window": params.trend_window,
        "momentum_window": params.momentum_window,
        "vol_window": params.vol_window,
        "trend_on": params.trend_on,
        "trend_off": params.trend_off,
        "mom_on": params.mom_on,
        "mom_off": params.mom_off,
        "vol_cap": params.vol_cap,
        "risk_on_leverage": params.risk_on_leverage,
        "risk_off_qqq_position": params.risk_off_qqq_position,
        "risk_off_symbol": params.risk_off_symbol,
        "risk_off_weight": params.risk_off_weight
        if params.risk_off_weight is not None
        else params.risk_off_qqq_position,
        **_asset_fields(params),
        "transition_position": params.transition_position,
        "min_hold_days": params.min_hold_days,
        "cooldown_days": params.cooldown_days,
        "use_rebound_module": int(params.rebound.use_rebound_module),
        "rolling_high_window": params.rebound.rolling_high_window,
        "drawdown_trigger": params.rebound.drawdown_trigger,
        "rebound_momentum_window": params.rebound.rebound_momentum_window,
        "rebound_momentum_threshold": params.rebound.rebound_momentum_threshold,
        "reclaim_ma_window": params.rebound.reclaim_ma_window,
        "rebound_position": params.rebound.rebound_position,
        "rebound_hold_days": params.rebound.rebound_hold_days,
        "extreme_crash_vol_window": params.rebound.extreme_crash_vol_window,
        "extreme_crash_vol_threshold": params.rebound.extreme_crash_vol_threshold,
        "use_drawdown_governor": int(params.governor.use_drawdown_governor),
        "portfolio_dd_trigger": params.governor.portfolio_dd_trigger,
        "qqq_dd_trigger": params.governor.qqq_dd_trigger,
        "reduced_exposure": params.governor.reduced_exposure,
        "recovery_ma_window": params.governor.recovery_ma_window,
        "recovery_momentum_window": params.governor.recovery_momentum_window,
        "recovery_momentum_threshold": params.governor.recovery_momentum_threshold,
        "max_days_reduced": params.governor.max_days_reduced,
        **full_perf,
        "recent_excess_cagr": recent_perf.get("excess_cagr", np.nan),
        "recent_excess_sharpe": recent_perf.get("excess_sharpe", np.nan),
        "recent_cagr": recent_perf.get("cagr", np.nan),
        "recent_sharpe": recent_perf.get("sharpe", np.nan),
        "state_score": state_score,
        **_market_internals_param_fields(params),
    }


def backtest_core_overlay_with_positions(
    data: pd.DataFrame,
    params: CoreOverlayParams,
    transaction_cost_bps: float = 10.0,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    asset_config = _asset_config(params)
    tqqq = data[asset_config.trade_asset].copy()
    qqq = data[asset_config.primary_signal_asset].copy()
    filter_close = data[asset_config.secondary_filter_asset].copy()

    signal = _call_core_overlay_signal(tqqq, qqq, params, data, filter_close=filter_close)
    position = signal.shift(1).fillna(0.0)
    risk_off_symbol, risk_off_weight = _risk_off_config(params)

    out = _portfolio_from_trade_weight(
        data=data,
        trade_weight=position,
        transaction_cost_bps=transaction_cost_bps,
        risk_off_symbol=risk_off_symbol,
        risk_off_weight=risk_off_weight,
        trade_asset=asset_config.trade_asset,
        execution_model=execution_model,
    )
    out = _attach_market_internals_columns(out, data, params)
    return out


def summarize_core_overlay_backtest(
    bt: pd.DataFrame,
    benchmark_ret: pd.Series,
    params: CoreOverlayParams,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    perf = summarize_performance(
        bt=bt,
        benchmark_ret=benchmark_ret,
        periods_per_year=periods_per_year,
    )

    return {
        "core_exposure": params.core_exposure,
        "overlay_max": params.overlay_max,
        "max_total_exposure": params.max_total_exposure,
        "trend_window": params.trend_window,
        "fast_trend_window": params.fast_trend_window,
        "momentum_window": params.momentum_window,
        "vol_window": params.vol_window,
        "vol_cap": params.vol_cap,
        "crash_cut_exposure": params.crash_cut_exposure,
        "rebound_boost": int(params.rebound_boost),
        "use_rebound_module": int(params.rebound.use_rebound_module),
        "rolling_high_window": params.rebound.rolling_high_window,
        "drawdown_trigger": params.rebound.drawdown_trigger,
        "rebound_momentum_window": params.rebound.rebound_momentum_window,
        "rebound_momentum_threshold": params.rebound.rebound_momentum_threshold,
        "reclaim_ma_window": params.rebound.reclaim_ma_window,
        "rebound_position": params.rebound.rebound_position,
        "rebound_hold_days": params.rebound.rebound_hold_days,
        "extreme_crash_vol_window": params.rebound.extreme_crash_vol_window,
        "extreme_crash_vol_threshold": params.rebound.extreme_crash_vol_threshold,
        "use_drawdown_governor": int(params.governor.use_drawdown_governor),
        "portfolio_dd_trigger": params.governor.portfolio_dd_trigger,
        "qqq_dd_trigger": params.governor.qqq_dd_trigger,
        "reduced_exposure": params.governor.reduced_exposure,
        "recovery_ma_window": params.governor.recovery_ma_window,
        "recovery_momentum_window": params.governor.recovery_momentum_window,
        "recovery_momentum_threshold": params.governor.recovery_momentum_threshold,
        "max_days_reduced": params.governor.max_days_reduced,
        "risk_off_symbol": params.risk_off_symbol,
        "risk_off_weight": params.risk_off_weight,
        **_asset_fields(params),
        **_market_internals_param_fields(params),
        **perf,
    }


def make_core_overlay_eval_piece(
    full_data: pd.DataFrame,
    params: CoreOverlayParams,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    prev_position: Optional[float] = None,
    prev_risk_off_weight: Optional[float] = None,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    history = full_data.loc[:eval_end].copy()
    bt_full = backtest_core_overlay_with_positions(
        history,
        params,
        transaction_cost_bps,
        execution_model=execution_model,
    )

    piece = bt_full.loc[eval_start:eval_end].copy()
    if piece.empty:
        raise ValueError("Evaluation window is empty for core-overlay strategy.")

    if prev_position is not None:
        piece = _correct_first_bar_for_previous_weights(
            piece=piece,
            bt_full=bt_full,
            transaction_cost_bps=transaction_cost_bps,
            prev_tqqq_weight=prev_position,
            prev_risk_off_weight=prev_risk_off_weight,
        )

    return piece


def backtest_core_overlay_on_window(
    full_data: pd.DataFrame,
    params: CoreOverlayParams,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    prev_position: Optional[float] = None,
    prev_risk_off_weight: Optional[float] = None,
    execution_model: Optional[str] = None,
) -> Dict[str, float]:
    piece = make_core_overlay_eval_piece(
        full_data=full_data,
        params=params,
        eval_start=eval_start,
        eval_end=eval_end,
        transaction_cost_bps=transaction_cost_bps,
        prev_position=prev_position,
        prev_risk_off_weight=prev_risk_off_weight,
        execution_model=execution_model,
    )
    benchmark_ret = _benchmark_ret(full_data, params, eval_start, eval_end)

    return summarize_core_overlay_backtest(
        bt=piece,
        benchmark_ret=benchmark_ret,
        params=params,
        periods_per_year=periods_per_year,
    )


def backtest_vol_target_with_positions(
    data: pd.DataFrame,
    params: VolTargetParams,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    asset_config = _asset_config(params)
    tqqq = data[asset_config.trade_asset].copy()
    qqq = data[asset_config.primary_signal_asset].copy()
    filter_close = data[asset_config.secondary_filter_asset].copy()

    signal = build_vol_target_signal(
        tqqq_close=tqqq,
        qqq_close=qqq,
        params=params,
        periods_per_year=periods_per_year,
        price_data=data,
        filter_close=filter_close,
    )
    position = signal["target_exposure"].shift(1).fillna(0.0).rename("position")
    risk_off_symbol, risk_off_weight = _risk_off_config(params)

    out = _portfolio_from_trade_weight(
        data=data,
        trade_weight=position,
        transaction_cost_bps=transaction_cost_bps,
        risk_off_symbol=risk_off_symbol,
        risk_off_weight=risk_off_weight,
        trade_asset=asset_config.trade_asset,
        execution_model=execution_model,
    )
    for column in [
        "target_exposure",
        "realized_daily_vol",
        "realized_ann_vol",
        "qqq_trend_ma",
        "qqq_momentum",
        "below_trend",
        "strong_momentum",
        "crash_regime",
        "governor_active",
        "governor_portfolio_drawdown",
        "governor_qqq_drawdown",
        "governor_recovery_signal",
        "governor_restore_fraction",
        "governor_days_reduced",
    ]:
        out[column] = signal[column]
    out = _attach_market_internals_columns(out, data, params)
    return out


def summarize_vol_target_backtest(
    bt: pd.DataFrame,
    benchmark_ret: pd.Series,
    params: VolTargetParams,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    perf = summarize_performance(
        bt=bt,
        benchmark_ret=benchmark_ret,
        periods_per_year=periods_per_year,
    )

    return {
        "target_ann_vol": params.target_ann_vol,
        "realized_vol_window": params.realized_vol_window,
        "min_exposure": params.min_exposure,
        "max_exposure": params.max_exposure,
        "trend_window": params.trend_window,
        "momentum_window": params.momentum_window,
        "trend_multiplier_below_ma": params.trend_multiplier_below_ma,
        "momentum_boost": params.momentum_boost,
        "crash_vol_cutoff": params.crash_vol_cutoff,
        "crash_exposure": params.crash_exposure,
        "use_rebound_module": int(params.rebound.use_rebound_module),
        "rolling_high_window": params.rebound.rolling_high_window,
        "drawdown_trigger": params.rebound.drawdown_trigger,
        "rebound_momentum_window": params.rebound.rebound_momentum_window,
        "rebound_momentum_threshold": params.rebound.rebound_momentum_threshold,
        "reclaim_ma_window": params.rebound.reclaim_ma_window,
        "rebound_position": params.rebound.rebound_position,
        "rebound_hold_days": params.rebound.rebound_hold_days,
        "extreme_crash_vol_window": params.rebound.extreme_crash_vol_window,
        "extreme_crash_vol_threshold": params.rebound.extreme_crash_vol_threshold,
        "use_drawdown_governor": int(params.governor.use_drawdown_governor),
        "portfolio_dd_trigger": params.governor.portfolio_dd_trigger,
        "qqq_dd_trigger": params.governor.qqq_dd_trigger,
        "reduced_exposure": params.governor.reduced_exposure,
        "recovery_ma_window": params.governor.recovery_ma_window,
        "recovery_momentum_window": params.governor.recovery_momentum_window,
        "recovery_momentum_threshold": params.governor.recovery_momentum_threshold,
        "max_days_reduced": params.governor.max_days_reduced,
        "risk_off_symbol": params.risk_off_symbol,
        "risk_off_weight": params.risk_off_weight,
        **_asset_fields(params),
        **_market_internals_param_fields(params),
        "avg_target_exposure": float(bt["target_exposure"].mean()),
        "avg_realized_ann_vol": float(bt["realized_ann_vol"].mean()),
        "avg_realized_daily_vol": float(bt["realized_daily_vol"].mean()),
        "total_cost": float(bt["cost"].sum()) if "cost" in bt.columns else np.nan,
        **perf,
    }


def make_vol_target_eval_piece(
    full_data: pd.DataFrame,
    params: VolTargetParams,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    prev_position: Optional[float] = None,
    prev_risk_off_weight: Optional[float] = None,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    history = full_data.loc[:eval_end].copy()
    bt_full = backtest_vol_target_with_positions(
        history,
        params,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        execution_model=execution_model,
    )

    piece = bt_full.loc[eval_start:eval_end].copy()
    if piece.empty:
        raise ValueError("Evaluation window is empty for vol-target strategy.")

    if prev_position is not None:
        piece = _correct_first_bar_for_previous_weights(
            piece=piece,
            bt_full=bt_full,
            transaction_cost_bps=transaction_cost_bps,
            prev_tqqq_weight=prev_position,
            prev_risk_off_weight=prev_risk_off_weight,
        )

    return piece


def backtest_vol_target_on_window(
    full_data: pd.DataFrame,
    params: VolTargetParams,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    prev_position: Optional[float] = None,
    prev_risk_off_weight: Optional[float] = None,
    execution_model: Optional[str] = None,
) -> Dict[str, float]:
    piece = make_vol_target_eval_piece(
        full_data=full_data,
        params=params,
        eval_start=eval_start,
        eval_end=eval_end,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        prev_position=prev_position,
        prev_risk_off_weight=prev_risk_off_weight,
        execution_model=execution_model,
    )
    benchmark_ret = _benchmark_ret(full_data, params, eval_start, eval_end)

    return summarize_vol_target_backtest(
        bt=piece,
        benchmark_ret=benchmark_ret,
        params=params,
        periods_per_year=periods_per_year,
    )


def _rotation_weight_columns(bt: pd.DataFrame) -> list[str]:
    return [column for column in bt.columns if column.startswith("weight_")]


def backtest_rotation_with_positions(
    data: pd.DataFrame,
    params: RotationParams,
    transaction_cost_bps: float = 10.0,
    validate_start: Optional[pd.Timestamp] = None,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    signal = build_rotation_signal(data, params)
    target_weight_cols = [column for column in signal.columns if column.startswith("target_weight_")]
    weight_cols = [column.replace("target_weight_", "weight_", 1) for column in target_weight_cols]
    weights = signal[target_weight_cols].shift(1).fillna(0.0).copy()
    weights.columns = weight_cols

    traded_assets = [column.replace("weight_", "", 1) for column in weight_cols]
    validation_weights = weights
    if validate_start is not None:
        validation_weights = validation_weights.loc[pd.Timestamp(validate_start):]
    for symbol in traded_assets:
        missing_held = (
            (validation_weights[f"weight_{symbol}"].abs() > 1e-12)
            & data[symbol].reindex(validation_weights.index).isna()
        )
        if bool(missing_held.any()):
            first_missing = missing_held[missing_held].index[0]
            raise ValueError(f"Rotation holds {symbol} without price data on {first_missing.date()}.")

    requested_execution_model = execution_model_from_data(data, execution_model)
    effective_execution_model, execution_warning = resolve_execution_model(
        data,
        requested_execution_model,
        traded_assets,
    )
    returns = pd.DataFrame(
        {
            f"daily_ret_{symbol}": asset_returns(data, symbol, effective_execution_model)
            for symbol in traded_assets
        },
        index=data.index,
    )

    gross_ret = pd.Series(0.0, index=data.index, dtype=float)
    for symbol in traded_assets:
        gross_ret = gross_ret + weights[f"weight_{symbol}"] * returns[f"daily_ret_{symbol}"]

    turnover_by_asset = weights.diff().abs().fillna(weights.abs())
    turnover_by_asset.columns = [
        column.replace("weight_", "turnover_", 1)
        for column in turnover_by_asset.columns
    ]
    turnover = turnover_by_asset.sum(axis=1)
    cost = turnover * (transaction_cost_bps / 10000.0)
    ret = gross_ret - cost

    risk_assets = [symbol for symbol in params.risk_assets if symbol in traded_assets]
    defensive_asset = str(params.defensive_asset).upper()
    leveraged_exposure = (
        weights[[f"weight_{symbol}" for symbol in risk_assets]].sum(axis=1)
        if risk_assets
        else pd.Series(0.0, index=data.index)
    )
    defensive_weight = (
        weights[f"weight_{defensive_asset}"]
        if f"weight_{defensive_asset}" in weights.columns
        else pd.Series(0.0, index=data.index)
    )

    selected_assets = signal["selected_assets"].shift(1).fillna("NONE")
    selected_score = signal["selected_score"].shift(1)
    selected_assets.loc[leveraged_exposure <= 0.0] = defensive_asset

    out = pd.DataFrame(
        {
            "ret": ret,
            "gross_ret": gross_ret,
            "position": weights.sum(axis=1),
            "leveraged_exposure": leveraged_exposure,
            "defensive_weight": defensive_weight,
            "selected_assets": selected_assets,
            "selected_score": selected_score,
            "rebalance_signal": signal["is_rebalance_date"],
            "trend_filter_allows_risk": signal["trend_filter_allows_risk"],
            "available_risk_assets": signal["available_risk_assets"],
            "requested_execution_model": requested_execution_model,
            "execution_model": effective_execution_model,
            "execution_model_warning": execution_warning,
            "turnover": turnover,
            "cost": cost,
        },
        index=data.index,
    )
    out = pd.concat([out, weights, returns, turnover_by_asset], axis=1)
    return out


def _correct_rotation_first_bar_for_previous_weights(
    piece: pd.DataFrame,
    bt_full: pd.DataFrame,
    transaction_cost_bps: float,
    prev_weights: Optional[pd.Series],
) -> pd.DataFrame:
    if prev_weights is None:
        return piece
    weight_cols = _rotation_weight_columns(piece)
    if not weight_cols:
        return piece

    first_idx = piece.index[0]
    prev = prev_weights.reindex(weight_cols).fillna(0.0).astype(float)
    desired = piece.loc[first_idx, weight_cols].astype(float)
    turnover_by_asset = (desired - prev).abs()
    turnover0 = float(turnover_by_asset.sum())
    cost0 = turnover0 * (transaction_cost_bps / 10000.0)
    gross_ret0 = 0.0
    for weight_col in weight_cols:
        symbol = weight_col.replace("weight_", "", 1)
        ret_col = f"daily_ret_{symbol}"
        gross_ret0 += float(desired[weight_col]) * float(bt_full.at[first_idx, ret_col])
        turnover_col = f"turnover_{symbol}"
        if turnover_col in piece.columns:
            piece.at[first_idx, turnover_col] = float(turnover_by_asset[weight_col])

    piece.at[first_idx, "turnover"] = turnover0
    piece.at[first_idx, "cost"] = cost0
    piece.at[first_idx, "gross_ret"] = gross_ret0
    piece.at[first_idx, "ret"] = gross_ret0 - cost0
    return piece


def make_rotation_eval_piece(
    full_data: pd.DataFrame,
    params: RotationParams,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    prev_weights: Optional[pd.Series] = None,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    history = full_data.loc[:eval_end].copy()
    bt_full = backtest_rotation_with_positions(
        history,
        params,
        transaction_cost_bps,
        validate_start=eval_start,
        execution_model=execution_model,
    )
    piece = bt_full.loc[eval_start:eval_end].copy()
    if piece.empty:
        raise ValueError("Evaluation window is empty for rotation strategy.")
    return _correct_rotation_first_bar_for_previous_weights(
        piece=piece,
        bt_full=bt_full,
        transaction_cost_bps=transaction_cost_bps,
        prev_weights=prev_weights,
    )


def summarize_rotation_backtest(
    bt: pd.DataFrame,
    benchmark_ret: pd.Series,
    params: RotationParams,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    perf = summarize_performance(bt, benchmark_ret, periods_per_year)
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
        **_asset_fields(params),
        "avg_leveraged_exposure": float(bt["leveraged_exposure"].mean()),
        "avg_defensive_weight": float(bt["defensive_weight"].mean()),
        "total_turnover": float(bt["turnover"].sum()),
        "total_cost": float(bt["cost"].sum()),
        **perf,
    }


def backtest_rotation_on_window(
    full_data: pd.DataFrame,
    params: RotationParams,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
    prev_weights: Optional[pd.Series] = None,
    execution_model: Optional[str] = None,
) -> Dict[str, float]:
    piece = make_rotation_eval_piece(
        full_data=full_data,
        params=params,
        eval_start=eval_start,
        eval_end=eval_end,
        transaction_cost_bps=transaction_cost_bps,
        prev_weights=prev_weights,
        execution_model=execution_model,
    )
    benchmark_ret = _benchmark_ret(full_data, params, eval_start, eval_end)
    return summarize_rotation_backtest(
        bt=piece,
        benchmark_ret=benchmark_ret,
        params=params,
        periods_per_year=periods_per_year,
    )
