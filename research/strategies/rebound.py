from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ma_strategy import compute_ma


@dataclass(frozen=True)
class ReboundParams:
    use_rebound_module: bool = False
    rolling_high_window: int = 126
    drawdown_trigger: float = -0.20
    rebound_momentum_window: int = 10
    rebound_momentum_threshold: float = 0.05
    reclaim_ma_window: int = 20
    rebound_position: float = 1.0
    rebound_hold_days: int = 10
    extreme_crash_vol_window: int = 20
    extreme_crash_vol_threshold: float = 0.05


def apply_rebound_reentry(
    qqq_close: pd.Series,
    tqqq_close: pd.Series,
    current_position: pd.Series,
    rolling_high_window: int,
    drawdown_trigger: float,
    rebound_momentum_window: int,
    rebound_momentum_threshold: float,
    reclaim_ma_window: int,
    rebound_position: float,
    rebound_hold_days: int,
    extreme_crash_vol_window: int = 20,
    extreme_crash_vol_threshold: float = 0.05,
) -> pd.Series:
    """
    Force faster re-entry after a QQQ drawdown when rebound evidence appears.
    """
    _ = tqqq_close
    position = current_position.astype(float).copy()
    if position.empty or rebound_hold_days <= 0:
        return position.rename("rebound_position")

    rolling_high = qqq_close.rolling(
        rolling_high_window,
        min_periods=rolling_high_window,
    ).max()
    drawdown = qqq_close / rolling_high - 1.0
    reclaim_ma = compute_ma(qqq_close, reclaim_ma_window, "sma")
    rebound_momentum = qqq_close.pct_change(rebound_momentum_window)
    qqq_vol = qqq_close.pct_change().rolling(
        extreme_crash_vol_window,
        min_periods=extreme_crash_vol_window,
    ).std()

    out = position.copy()
    candidate_active = False
    hold_remaining = 0

    for idx in out.index:
        dd = drawdown.loc[idx]
        extreme_vol = bool(
            pd.notna(qqq_vol.loc[idx])
            and qqq_vol.loc[idx] > extreme_crash_vol_threshold
        )

        if pd.notna(dd) and dd <= drawdown_trigger:
            candidate_active = True
        elif pd.notna(dd) and dd >= 0.0:
            candidate_active = False

        rebound_trigger = (
            candidate_active
            and pd.notna(reclaim_ma.loc[idx])
            and pd.notna(rebound_momentum.loc[idx])
            and qqq_close.loc[idx] > reclaim_ma.loc[idx]
            and rebound_momentum.loc[idx] > rebound_momentum_threshold
        )

        if extreme_vol:
            hold_remaining = 0
            continue

        if rebound_trigger:
            hold_remaining = max(hold_remaining, int(rebound_hold_days))

        if hold_remaining > 0:
            out.loc[idx] = max(float(out.loc[idx]), float(rebound_position))
            hold_remaining -= 1

    return out.rename("rebound_position")


def without_rebound(params: ReboundParams) -> ReboundParams:
    return ReboundParams(
        use_rebound_module=False,
        rolling_high_window=params.rolling_high_window,
        drawdown_trigger=params.drawdown_trigger,
        rebound_momentum_window=params.rebound_momentum_window,
        rebound_momentum_threshold=params.rebound_momentum_threshold,
        reclaim_ma_window=params.reclaim_ma_window,
        rebound_position=params.rebound_position,
        rebound_hold_days=params.rebound_hold_days,
        extreme_crash_vol_window=params.extreme_crash_vol_window,
        extreme_crash_vol_threshold=params.extreme_crash_vol_threshold,
    )
