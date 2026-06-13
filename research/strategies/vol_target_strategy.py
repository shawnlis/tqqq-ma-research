from __future__ import annotations

import math
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
from .ma_strategy import compute_ma
from .market_internals import MarketInternalsParams, build_market_internals_signal
from .rebound import ReboundParams, apply_rebound_reentry


STRONG_MOMENTUM_THRESHOLD = 0.05


@dataclass(frozen=True)
class VolTargetParams:
    target_ann_vol: float
    realized_vol_window: int
    min_exposure: float
    max_exposure: float
    trend_window: int
    momentum_window: int
    trend_multiplier_below_ma: float
    momentum_boost: float
    crash_vol_cutoff: float
    crash_exposure: float
    rebound: ReboundParams = field(default_factory=ReboundParams)
    governor: DrawdownGovernorParams = field(default_factory=DrawdownGovernorParams)
    risk_off_symbol: str = "CASH"
    risk_off_weight: float = 0.0
    market_internals: MarketInternalsParams = field(default_factory=MarketInternalsParams)
    asset_config: AssetConfig = DEFAULT_ASSET_CONFIG


def build_vol_target_signal(
    tqqq_close: pd.Series,
    qqq_close: pd.Series,
    params: VolTargetParams,
    periods_per_year: int = 252,
    price_data: Optional[pd.DataFrame] = None,
    filter_close: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build unshifted target exposure and realized volatility diagnostics.

    The returned target is intentionally unshifted. Backtests apply shift(1) so
    the exposure chosen from close t is only held starting on t+1.
    """
    filter_close = qqq_close if filter_close is None else filter_close.reindex(qqq_close.index)
    tqqq_returns = tqqq_close.pct_change().fillna(0.0)
    realized_daily_vol = tqqq_returns.rolling(
        int(params.realized_vol_window),
        min_periods=int(params.realized_vol_window),
    ).std(ddof=0)
    realized_ann_vol = realized_daily_vol * math.sqrt(periods_per_year)

    min_exposure = max(float(params.min_exposure), 0.0)
    max_exposure = max(float(params.max_exposure), 0.0)
    if max_exposure < min_exposure:
        max_exposure = min_exposure

    base_exposure = float(params.target_ann_vol) / realized_ann_vol.replace(0.0, np.nan)
    target_exposure = base_exposure.clip(lower=min_exposure, upper=max_exposure)
    target_exposure = target_exposure.fillna(min_exposure)

    trend_ma = compute_ma(filter_close, int(params.trend_window), "sma")
    qqq_momentum = qqq_close.pct_change(int(params.momentum_window))

    below_trend = (filter_close < trend_ma).fillna(False)
    strong_momentum = (qqq_momentum > STRONG_MOMENTUM_THRESHOLD).fillna(False)
    if params.market_internals.use_market_internals:
        internals_source = price_data if price_data is not None else pd.DataFrame({"QQQ": qqq_close})
        internals = build_market_internals_signal(
            internals_source,
            params.market_internals,
        ).reindex(tqqq_close.index)
        risk_score = internals["market_internals_risk_score"].fillna(0.5)
        below_trend = (risk_score <= float(params.market_internals.risk_off_threshold)).fillna(False)
        strong_momentum = (risk_score >= float(params.market_internals.risk_on_threshold)).fillna(False)
    crash_regime = (
        (realized_daily_vol > float(params.crash_vol_cutoff))
        & below_trend
    ).fillna(False)

    target_exposure.loc[below_trend] = (
        target_exposure.loc[below_trend] * float(params.trend_multiplier_below_ma)
    )
    target_exposure.loc[strong_momentum] = (
        target_exposure.loc[strong_momentum] * float(params.momentum_boost)
    )
    target_exposure = target_exposure.clip(lower=0.0, upper=max_exposure)
    target_exposure.loc[crash_regime] = float(params.crash_exposure)
    target_exposure = target_exposure.clip(lower=0.0, upper=max_exposure)

    if params.rebound.use_rebound_module:
        target_exposure = apply_rebound_reentry(
            qqq_close=filter_close,
            tqqq_close=tqqq_close,
            current_position=target_exposure,
            rolling_high_window=params.rebound.rolling_high_window,
            drawdown_trigger=params.rebound.drawdown_trigger,
            rebound_momentum_window=params.rebound.rebound_momentum_window,
            rebound_momentum_threshold=params.rebound.rebound_momentum_threshold,
            reclaim_ma_window=params.rebound.reclaim_ma_window,
            rebound_position=params.rebound.rebound_position,
            rebound_hold_days=params.rebound.rebound_hold_days,
            extreme_crash_vol_window=params.rebound.extreme_crash_vol_window,
            extreme_crash_vol_threshold=params.rebound.extreme_crash_vol_threshold,
        ).clip(lower=0.0, upper=max_exposure)

    if params.governor.use_drawdown_governor:
        strategy_equity = estimate_strategy_equity(target_exposure, tqqq_close)
        governor = apply_drawdown_governor(
            strategy_equity=strategy_equity,
            desired_position=target_exposure,
            qqq_close=filter_close,
            tqqq_close=tqqq_close,
            params=params.governor,
        )
        target_exposure = governor["governed_position"].clip(lower=0.0, upper=max_exposure)
    else:
        governor = pd.DataFrame(
            {
                "governor_active": 0,
                "portfolio_drawdown": np.nan,
                "qqq_drawdown": np.nan,
                "recovery_signal": 0,
                "restore_fraction": 1.0,
                "days_reduced": 0,
            },
            index=tqqq_close.index,
        )

    return pd.DataFrame(
        {
            "target_exposure": target_exposure,
            "realized_daily_vol": realized_daily_vol,
            "realized_ann_vol": realized_ann_vol,
            "qqq_trend_ma": trend_ma,
            "qqq_momentum": qqq_momentum,
            "below_trend": below_trend.astype(int),
            "strong_momentum": strong_momentum.astype(int),
            "crash_regime": crash_regime.astype(int),
            "governor_active": governor["governor_active"].reindex(tqqq_close.index).fillna(0).astype(int),
            "governor_portfolio_drawdown": governor["portfolio_drawdown"].reindex(tqqq_close.index),
            "governor_qqq_drawdown": governor["qqq_drawdown"].reindex(tqqq_close.index),
            "governor_recovery_signal": governor["recovery_signal"].reindex(tqqq_close.index).fillna(0).astype(int),
            "governor_restore_fraction": governor["restore_fraction"].reindex(tqqq_close.index).fillna(1.0),
            "governor_days_reduced": governor["days_reduced"].reindex(tqqq_close.index).fillna(0).astype(int),
        },
        index=tqqq_close.index,
    )


def make_vol_target_param_grid(
    target_ann_vols: Iterable[float],
    realized_vol_windows: Iterable[int],
    min_exposures: Iterable[float],
    max_exposures: Iterable[float],
    trend_windows: Iterable[int],
    momentum_windows: Iterable[int],
    trend_multiplier_below_ma_options: Iterable[float],
    momentum_boosts: Iterable[float],
    crash_vol_cutoffs: Iterable[float],
    crash_exposures: Iterable[float],
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
    risk_off_symbols: Iterable[str] = ("CASH",),
    risk_off_weights: Iterable[float] = (0.0,),
    market_internals_options: Optional[Iterable[MarketInternalsParams]] = None,
    asset_config: AssetConfig = DEFAULT_ASSET_CONFIG,
) -> List[VolTargetParams]:
    grid: List[VolTargetParams] = []
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
    for target_ann_vol in target_ann_vols:
        if float(target_ann_vol) <= 0.0:
            continue
        for realized_vol_window in realized_vol_windows:
            if int(realized_vol_window) <= 1:
                continue
            for min_exposure in min_exposures:
                for max_exposure in max_exposures:
                    if float(min_exposure) < 0.0 or float(max_exposure) <= 0.0:
                        continue
                    if float(min_exposure) > float(max_exposure):
                        continue
                    for trend_window in trend_windows:
                        if int(trend_window) <= 1:
                            continue
                        for momentum_window in momentum_windows:
                            if int(momentum_window) <= 0:
                                continue
                            for trend_multiplier in trend_multiplier_below_ma_options:
                                if float(trend_multiplier) < 0.0:
                                    continue
                                for momentum_boost in momentum_boosts:
                                    if float(momentum_boost) < 0.0:
                                        continue
                                    for crash_vol_cutoff in crash_vol_cutoffs:
                                        if float(crash_vol_cutoff) <= 0.0:
                                            continue
                                        for crash_exposure in crash_exposures:
                                            if float(crash_exposure) < 0.0:
                                                continue
                                            for rebound in rebound_grid:
                                                for governor in governor_grid:
                                                    for risk_off_symbol in risk_off_symbols:
                                                        for risk_off_weight in risk_off_weights:
                                                            for market_internals in market_internals_options:
                                                                grid.append(
                                                                    VolTargetParams(
                                                                        target_ann_vol=float(target_ann_vol),
                                                                        realized_vol_window=int(realized_vol_window),
                                                                        min_exposure=float(min_exposure),
                                                                        max_exposure=float(max_exposure),
                                                                        trend_window=int(trend_window),
                                                                        momentum_window=int(momentum_window),
                                                                        trend_multiplier_below_ma=float(trend_multiplier),
                                                                        momentum_boost=float(momentum_boost),
                                                                        crash_vol_cutoff=float(crash_vol_cutoff),
                                                                        crash_exposure=float(crash_exposure),
                                                                        rebound=rebound,
                                                                        governor=governor,
                                                                        risk_off_symbol=str(risk_off_symbol).upper(),
                                                                        risk_off_weight=float(risk_off_weight),
                                                                        market_internals=market_internals,
                                                                        asset_config=asset_config.normalized(),
                                                                    )
                                                                )
    return grid
