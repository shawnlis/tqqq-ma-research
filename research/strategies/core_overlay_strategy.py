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
from .ma_strategy import compute_ma
from .market_internals import MarketInternalsParams, build_market_internals_signal
from .rebound import ReboundParams, apply_rebound_reentry


@dataclass(frozen=True)
class CoreOverlayParams:
    core_exposure: float
    overlay_max: float
    max_total_exposure: float
    trend_window: int
    fast_trend_window: int
    momentum_window: int
    vol_window: int
    vol_cap: float
    crash_cut_exposure: float
    rebound_boost: bool
    rebound: ReboundParams = field(default_factory=ReboundParams)
    governor: DrawdownGovernorParams = field(default_factory=DrawdownGovernorParams)
    risk_off_symbol: str = "CASH"
    risk_off_weight: float = 0.0
    market_internals: MarketInternalsParams = field(default_factory=MarketInternalsParams)
    asset_config: AssetConfig = DEFAULT_ASSET_CONFIG


def build_core_overlay_signal(
    tqqq_close: pd.Series,
    qqq_close: pd.Series,
    params: CoreOverlayParams,
    price_data: Optional[pd.DataFrame] = None,
    filter_close: Optional[pd.Series] = None,
) -> pd.Series:
    filter_close = qqq_close if filter_close is None else filter_close.reindex(qqq_close.index)
    core_exposure = float(np.clip(params.core_exposure, 0.0, params.max_total_exposure))
    overlay_max = max(float(params.overlay_max), 0.0)
    max_total_exposure = max(float(params.max_total_exposure), 0.0)
    crash_cut_exposure = float(np.clip(params.crash_cut_exposure, 0.0, max_total_exposure))

    trend_ma = compute_ma(filter_close, params.trend_window, "sma")
    fast_trend_ma = compute_ma(filter_close, params.fast_trend_window, "sma")
    momentum = qqq_close.pct_change(params.momentum_window)
    fast_momentum_20d = qqq_close.pct_change(20)
    qqq_vol = filter_close.pct_change().rolling(
        params.vol_window,
        min_periods=params.vol_window,
    ).std()

    trend_ok = ((filter_close > trend_ma) & (momentum > 0.0)).fillna(False)
    rebound_ok = (
        params.rebound_boost
        & (filter_close > fast_trend_ma)
        & (fast_momentum_20d > 0.05)
    ).fillna(False)
    crash_regime = ((qqq_vol > params.vol_cap) & (filter_close < trend_ma)).fillna(False)
    if params.market_internals.use_market_internals:
        internals_source = price_data if price_data is not None else pd.DataFrame({"QQQ": qqq_close})
        internals = build_market_internals_signal(
            internals_source,
            params.market_internals,
        ).reindex(tqqq_close.index)
        risk_score = internals["market_internals_risk_score"].fillna(0.5)
        market_risk_on = risk_score >= float(params.market_internals.risk_on_threshold)
        market_risk_off = risk_score <= float(params.market_internals.risk_off_threshold)
        trend_ok = market_risk_on
        rebound_ok = rebound_ok & (~market_risk_off)
        crash_regime = crash_regime | market_risk_off

    raw_position = pd.Series(core_exposure, index=tqqq_close.index, name="raw_position")
    raw_position.loc[trend_ok | rebound_ok] = core_exposure + overlay_max
    raw_position = raw_position.clip(lower=crash_cut_exposure, upper=max_total_exposure)
    raw_position.loc[crash_regime] = crash_cut_exposure
    if params.rebound.use_rebound_module:
        raw_position = apply_rebound_reentry(
            qqq_close=filter_close,
            tqqq_close=tqqq_close,
            current_position=raw_position,
            rolling_high_window=params.rebound.rolling_high_window,
            drawdown_trigger=params.rebound.drawdown_trigger,
            rebound_momentum_window=params.rebound.rebound_momentum_window,
            rebound_momentum_threshold=params.rebound.rebound_momentum_threshold,
            reclaim_ma_window=params.rebound.reclaim_ma_window,
            rebound_position=params.rebound.rebound_position,
            rebound_hold_days=params.rebound.rebound_hold_days,
            extreme_crash_vol_window=params.rebound.extreme_crash_vol_window,
            extreme_crash_vol_threshold=params.rebound.extreme_crash_vol_threshold,
        ).clip(upper=max_total_exposure)

    if params.governor.use_drawdown_governor:
        strategy_equity = estimate_strategy_equity(raw_position, tqqq_close)
        governor = apply_drawdown_governor(
            strategy_equity=strategy_equity,
            desired_position=raw_position,
            qqq_close=filter_close,
            tqqq_close=tqqq_close,
            params=params.governor,
        )
        raw_position = governor["governed_position"].clip(lower=0.0, upper=max_total_exposure)

    return raw_position.fillna(core_exposure)


def make_core_overlay_param_grid(
    core_exposures: Iterable[float],
    overlay_maxes: Iterable[float],
    max_total_exposures: Iterable[float],
    trend_windows: Iterable[int],
    fast_trend_windows: Iterable[int],
    momentum_windows: Iterable[int],
    vol_windows: Iterable[int],
    vol_caps: Iterable[float],
    crash_cut_exposures: Iterable[float],
    rebound_boost_options: Iterable[bool],
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
) -> List[CoreOverlayParams]:
    grid: List[CoreOverlayParams] = []
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
    for core_exposure in core_exposures:
        for overlay_max in overlay_maxes:
            for max_total_exposure in max_total_exposures:
                if core_exposure < 0.0 or overlay_max < 0.0 or max_total_exposure <= 0.0:
                    continue
                if core_exposure > max_total_exposure:
                    continue
                for crash_cut_exposure in crash_cut_exposures:
                    if crash_cut_exposure < 0.0 or crash_cut_exposure > core_exposure:
                        continue
                    for trend_window in trend_windows:
                        for fast_trend_window in fast_trend_windows:
                            if fast_trend_window >= trend_window:
                                continue
                            for momentum_window in momentum_windows:
                                for vol_window in vol_windows:
                                    for vol_cap in vol_caps:
                                        if vol_cap <= 0.0:
                                            continue
                                        for rebound_boost in rebound_boost_options:
                                            for rebound in rebound_grid:
                                                for governor in governor_grid:
                                                    for risk_off_symbol in risk_off_symbols:
                                                        for risk_off_weight in risk_off_weights:
                                                            for market_internals in market_internals_options:
                                                                grid.append(
                                                                    CoreOverlayParams(
                                                                        core_exposure=float(core_exposure),
                                                                        overlay_max=float(overlay_max),
                                                                        max_total_exposure=float(max_total_exposure),
                                                                        trend_window=int(trend_window),
                                                                        fast_trend_window=int(fast_trend_window),
                                                                        momentum_window=int(momentum_window),
                                                                        vol_window=int(vol_window),
                                                                        vol_cap=float(vol_cap),
                                                                        crash_cut_exposure=float(crash_cut_exposure),
                                                                        rebound_boost=bool(rebound_boost),
                                                                        rebound=rebound,
                                                                        governor=governor,
                                                                        risk_off_symbol=str(risk_off_symbol).upper(),
                                                                        risk_off_weight=float(risk_off_weight),
                                                                        market_internals=market_internals,
                                                                        asset_config=asset_config.normalized(),
                                                                    )
                                                                )
    return grid
