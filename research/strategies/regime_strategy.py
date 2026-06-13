from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from ..assets import AssetConfig, DEFAULT_ASSET_CONFIG
from .drawdown_governor import (
    DrawdownGovernorParams,
    apply_drawdown_governor,
    estimate_strategy_equity,
    make_drawdown_governor_param_grid,
)
from .ma_strategy import apply_cooldown, compute_ma
from .market_internals import MarketInternalsParams, build_market_internals_signal
from .rebound import ReboundParams, apply_rebound_reentry


@dataclass(frozen=True)
class RegimeParams:
    trend_window: int
    momentum_window: int
    vol_window: int
    trend_on: float
    trend_off: float
    mom_on: float
    mom_off: float
    vol_cap: float
    risk_on_leverage: float = 1.0
    risk_off_qqq_position: float = 0.0
    transition_position: float = 0.35
    min_hold_days: int = 0
    cooldown_days: int = 2
    rebound: ReboundParams = field(default_factory=ReboundParams)
    governor: DrawdownGovernorParams = field(default_factory=DrawdownGovernorParams)
    risk_off_symbol: str = "QQQ"
    risk_off_weight: Optional[float] = None
    market_internals: MarketInternalsParams = field(default_factory=MarketInternalsParams)
    asset_config: AssetConfig = DEFAULT_ASSET_CONFIG


def apply_min_holding(position: pd.Series, min_hold_days: int) -> pd.Series:
    """
    Enforce a minimum number of bars between position changes.
    """
    if min_hold_days <= 0:
        return position.astype(float)

    raw = position.fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    n = raw.shape[0]
    out = np.empty_like(raw)
    tol = 1e-12

    out[0] = raw[0]
    bars_since_change = 0

    for i in range(1, n):
        desired = raw[i]
        prev = out[i - 1]

        if abs(desired - prev) <= tol:
            out[i] = prev
            bars_since_change += 1
            continue

        if bars_since_change < min_hold_days:
            out[i] = prev
            bars_since_change += 1
        else:
            out[i] = desired
            bars_since_change = 0

    return pd.Series(out, index=position.index, name="min_hold_position")


def build_regime_signal(
    tqqq_close: pd.Series,
    qqq_close: pd.Series,
    params: RegimeParams,
    price_data: Optional[pd.DataFrame] = None,
    filter_close: Optional[pd.Series] = None,
) -> pd.Series:
    filter_close = qqq_close if filter_close is None else filter_close.reindex(qqq_close.index)
    risk_on_leverage = float(np.clip(params.risk_on_leverage, 1.0, 1.5))
    transition_position = float(np.clip(params.transition_position, 0.0, 1.0))
    raw_position = pd.Series(0.0, index=tqqq_close.index, name="raw_position")

    if params.market_internals.use_market_internals:
        internals_source = price_data if price_data is not None else pd.DataFrame({"QQQ": qqq_close})
        internals = build_market_internals_signal(
            internals_source,
            params.market_internals,
        ).reindex(tqqq_close.index)
        risk_score = internals["market_internals_risk_score"].fillna(0.5)
        risk_on = risk_score >= float(params.market_internals.risk_on_threshold)
        risk_off = risk_score <= float(params.market_internals.risk_off_threshold)
        transition = (~risk_on) & (~risk_off)
        super_risk_on = risk_score >= min(
            0.95,
            float(params.market_internals.risk_on_threshold) + 0.15,
        )
    else:
        qqq_ma = compute_ma(filter_close, params.trend_window, "sma")
        trend = filter_close / qqq_ma - 1.0
        momentum = qqq_close.pct_change(params.momentum_window)
        qqq_daily_ret = filter_close.pct_change()
        vol = qqq_daily_ret.rolling(params.vol_window, min_periods=params.vol_window).std()

        risk_on = (
            (trend > params.trend_on)
            & (momentum > params.mom_on)
            & (vol < params.vol_cap)
        ).fillna(False)

        # Super trend layer allows moderate leverage only in strongest conditions.
        super_trend = (trend > params.trend_on + 0.015).fillna(False)
        super_momentum = (momentum > params.mom_on + 0.04).fillna(False)
        super_low_vol = (vol < params.vol_cap * 0.90).fillna(False)
        super_risk_on = risk_on & super_trend & super_momentum & super_low_vol

        transition = (
            (~risk_on)
            & (trend > params.trend_off)
            & (momentum > params.mom_off)
        ).fillna(False)

    raw_position.loc[transition] = transition_position
    raw_position.loc[risk_on] = 1.0
    raw_position.loc[super_risk_on] = risk_on_leverage

    cooled = apply_cooldown(raw_position, params.cooldown_days)
    held = apply_min_holding(cooled, params.min_hold_days)
    if params.rebound.use_rebound_module:
        held = apply_rebound_reentry(
            qqq_close=filter_close,
            tqqq_close=tqqq_close,
            current_position=held,
            rolling_high_window=params.rebound.rolling_high_window,
            drawdown_trigger=params.rebound.drawdown_trigger,
            rebound_momentum_window=params.rebound.rebound_momentum_window,
            rebound_momentum_threshold=params.rebound.rebound_momentum_threshold,
            reclaim_ma_window=params.rebound.reclaim_ma_window,
            rebound_position=params.rebound.rebound_position,
            rebound_hold_days=params.rebound.rebound_hold_days,
            extreme_crash_vol_window=params.rebound.extreme_crash_vol_window,
            extreme_crash_vol_threshold=params.rebound.extreme_crash_vol_threshold,
        )
    if params.governor.use_drawdown_governor:
        strategy_equity = estimate_strategy_equity(held, tqqq_close)
        governor = apply_drawdown_governor(
            strategy_equity=strategy_equity,
            desired_position=held,
            qqq_close=filter_close,
            tqqq_close=tqqq_close,
            params=params.governor,
        )
        held = governor["governed_position"].clip(lower=0.0, upper=float(params.risk_on_leverage))
    return held.fillna(0.0)


def build_regime_weights(
    tqqq_close: pd.Series,
    qqq_close: pd.Series,
    params: RegimeParams,
    price_data: Optional[pd.DataFrame] = None,
    filter_close: Optional[pd.Series] = None,
) -> pd.DataFrame:
    tqqq_weight = build_regime_signal(
        tqqq_close,
        qqq_close,
        params,
        price_data=price_data,
        filter_close=filter_close,
    ).fillna(0.0)
    if params.risk_off_weight is None:
        risk_off_qqq_position = float(np.clip(params.risk_off_qqq_position, 0.0, 1.0))
        risk_off_weight = pd.Series(
            np.where(tqqq_weight > 0.0, 0.0, risk_off_qqq_position),
            index=tqqq_weight.index,
            name="risk_off_weight",
        )
    else:
        risk_off_allocation_weight = float(np.clip(params.risk_off_weight, 0.0, 1.0))
        risk_off_weight = ((1.0 - tqqq_weight).clip(lower=0.0) * risk_off_allocation_weight).rename(
            "risk_off_weight"
        )

    qqq_weight = pd.Series(
        np.where(str(params.risk_off_symbol).upper() == "QQQ", risk_off_weight, 0.0),
        index=tqqq_weight.index,
        name="qqq_weight",
    )
    total_position = (tqqq_weight + risk_off_weight).rename("position")

    return pd.DataFrame(
        {
            "tqqq_weight": tqqq_weight,
            "qqq_weight": qqq_weight,
            "risk_off_weight": risk_off_weight,
            "position": total_position,
        },
        index=tqqq_close.index,
    )


def make_regime_param_grid(
    trend_windows: Iterable[int],
    momentum_windows: Iterable[int],
    vol_windows: Iterable[int],
    trend_on_levels: Iterable[float],
    trend_off_levels: Iterable[float],
    mom_on_levels: Iterable[float],
    mom_off_levels: Iterable[float],
    vol_caps: Iterable[float],
    risk_on_leverages: Iterable[float],
    risk_off_qqq_positions: Iterable[float],
    transition_positions: Iterable[float],
    min_hold_days_options: Iterable[int],
    cooldown_days_options: Iterable[int],
    use_rebound_options: Iterable[bool] = (False,),
    rolling_high_windows: Iterable[int] = (126,),
    drawdown_triggers: Iterable[float] = (-0.20,),
    rebound_momentum_windows: Iterable[int] = (10,),
    rebound_momentum_thresholds: Iterable[float] = (0.05,),
    reclaim_ma_windows: Iterable[int] = (20,),
    rebound_positions: Iterable[float] = (1.0,),
    rebound_hold_days_options: Iterable[int] = (10,),
    extreme_crash_vol_windows: Iterable[int] = (20,),
    extreme_crash_vol_thresholds: Iterable[float] = (0.05,),
    use_drawdown_governor_options: Iterable[bool] = (False,),
    portfolio_dd_triggers: Iterable[float] = (-0.30,),
    qqq_dd_triggers: Iterable[float] = (-0.15,),
    reduced_exposures: Iterable[float] = (0.50,),
    recovery_ma_windows: Iterable[int] = (20,),
    recovery_momentum_windows: Iterable[int] = (10,),
    recovery_momentum_thresholds: Iterable[float] = (0.05,),
    max_days_reduced_options: Iterable[int] = (40,),
    risk_off_symbols: Iterable[str] = ("QQQ",),
    risk_off_weights: Iterable[Optional[float]] = (None,),
    market_internals_options: Optional[Iterable[MarketInternalsParams]] = None,
    asset_config: AssetConfig = DEFAULT_ASSET_CONFIG,
) -> List[RegimeParams]:
    grid: List[RegimeParams] = []
    if market_internals_options is None:
        market_internals_options = (MarketInternalsParams(),)
    rebound_grid: List[ReboundParams] = []
    for use_rebound in use_rebound_options:
        if not use_rebound:
            rebound_grid.append(ReboundParams(use_rebound_module=False))
            continue
        for rolling_high_window in rolling_high_windows:
            for drawdown_trigger in drawdown_triggers:
                for rebound_momentum_window in rebound_momentum_windows:
                    for rebound_momentum_threshold in rebound_momentum_thresholds:
                        for reclaim_ma_window in reclaim_ma_windows:
                            for rebound_position in rebound_positions:
                                for rebound_hold_days in rebound_hold_days_options:
                                    for extreme_crash_vol_window in extreme_crash_vol_windows:
                                        for extreme_crash_vol_threshold in extreme_crash_vol_thresholds:
                                            rebound_grid.append(
                                                ReboundParams(
                                                    use_rebound_module=True,
                                                    rolling_high_window=int(rolling_high_window),
                                                    drawdown_trigger=float(drawdown_trigger),
                                                    rebound_momentum_window=int(rebound_momentum_window),
                                                    rebound_momentum_threshold=float(rebound_momentum_threshold),
                                                    reclaim_ma_window=int(reclaim_ma_window),
                                                    rebound_position=float(rebound_position),
                                                    rebound_hold_days=int(rebound_hold_days),
                                                    extreme_crash_vol_window=int(extreme_crash_vol_window),
                                                    extreme_crash_vol_threshold=float(extreme_crash_vol_threshold),
                                                )
                                            )
    governor_grid = make_drawdown_governor_param_grid(
        use_drawdown_governor_options=use_drawdown_governor_options,
        portfolio_dd_triggers=portfolio_dd_triggers,
        qqq_dd_triggers=qqq_dd_triggers,
        reduced_exposures=reduced_exposures,
        recovery_ma_windows=recovery_ma_windows,
        recovery_momentum_windows=recovery_momentum_windows,
        recovery_momentum_thresholds=recovery_momentum_thresholds,
        max_days_reduced_options=max_days_reduced_options,
    )
    for trend_window in trend_windows:
        for momentum_window in momentum_windows:
            for vol_window in vol_windows:
                for trend_on in trend_on_levels:
                    for trend_off in trend_off_levels:
                        if trend_on < trend_off:
                            continue
                        for mom_on in mom_on_levels:
                            for mom_off in mom_off_levels:
                                if mom_on < mom_off:
                                    continue
                                for vol_cap in vol_caps:
                                    for risk_on_leverage in risk_on_leverages:
                                        if risk_on_leverage < 1.0 or risk_on_leverage > 1.5:
                                            continue
                                        for risk_off_qqq_position in risk_off_qqq_positions:
                                            if not (0.0 <= risk_off_qqq_position <= 1.0):
                                                continue
                                            for transition_position in transition_positions:
                                                if not (0.0 <= transition_position <= 1.0):
                                                    continue
                                                for min_hold_days in min_hold_days_options:
                                                    if min_hold_days < 0:
                                                        continue
                                                    for cooldown_days in cooldown_days_options:
                                                        for rebound in rebound_grid:
                                                            for governor in governor_grid:
                                                                for risk_off_symbol in risk_off_symbols:
                                                                    for risk_off_weight in risk_off_weights:
                                                                        for market_internals in market_internals_options:
                                                                            grid.append(
                                                                                RegimeParams(
                                                                                    trend_window=int(trend_window),
                                                                                    momentum_window=int(momentum_window),
                                                                                    vol_window=int(vol_window),
                                                                                    trend_on=float(trend_on),
                                                                                    trend_off=float(trend_off),
                                                                                    mom_on=float(mom_on),
                                                                                    mom_off=float(mom_off),
                                                                                    vol_cap=float(vol_cap),
                                                                                    risk_on_leverage=float(risk_on_leverage),
                                                                                    risk_off_qqq_position=float(risk_off_qqq_position),
                                                                                    transition_position=float(transition_position),
                                                                                    min_hold_days=int(min_hold_days),
                                                                                    cooldown_days=int(cooldown_days),
                                                                                    rebound=rebound,
                                                                                    governor=governor,
                                                                                    risk_off_symbol=str(risk_off_symbol).upper(),
                                                                                    risk_off_weight=(
                                                                                        None
                                                                                        if risk_off_weight is None
                                                                                        else float(risk_off_weight)
                                                                                    ),
                                                                                    market_internals=market_internals,
                                                                                    asset_config=asset_config.normalized(),
                                                                                )
                                                                            )
    return grid
