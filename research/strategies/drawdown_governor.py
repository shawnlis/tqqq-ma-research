from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from .ma_strategy import compute_ma


RESTORE_RAMP_DAYS = 5


@dataclass(frozen=True)
class DrawdownGovernorParams:
    use_drawdown_governor: bool = False
    portfolio_dd_trigger: float = -0.30
    qqq_dd_trigger: float = -0.15
    reduced_exposure: float = 0.50
    recovery_ma_window: int = 20
    recovery_momentum_window: int = 10
    recovery_momentum_threshold: float = 0.05
    max_days_reduced: int = 40


def estimate_strategy_equity(
    desired_position: pd.Series,
    tqqq_close: pd.Series,
) -> pd.Series:
    """
    Estimate pre-governor equity using the same close-t signal, t+1 position timing.
    """
    position = pd.Series(desired_position, copy=True).astype(float).shift(1).fillna(0.0)
    returns = pd.Series(tqqq_close, copy=True).astype(float).pct_change().fillna(0.0)
    equity = (1.0 + position.reindex(returns.index).fillna(0.0) * returns).cumprod()
    equity.name = "strategy_equity"
    return equity


def apply_drawdown_governor(
    strategy_equity: pd.Series,
    desired_position: pd.Series,
    qqq_close: pd.Series,
    tqqq_close: pd.Series,
    params: DrawdownGovernorParams,
) -> pd.DataFrame:
    """
    Cap desired exposure during severe drawdowns and restore when recovery appears.

    `desired_position` is an unshifted target formed at close t. The governed
    position returned here is also an unshifted target; backtests still apply
    shift(1) before returns and costs.
    """
    desired = pd.Series(desired_position, copy=True).astype(float)
    if desired.empty:
        return pd.DataFrame(index=desired.index)

    aligned = pd.concat(
        [
            pd.Series(strategy_equity, copy=True).rename("strategy_equity"),
            desired.rename("desired_position"),
            pd.Series(qqq_close, copy=True).rename("qqq_close"),
            pd.Series(tqqq_close, copy=True).rename("tqqq_close"),
        ],
        axis=1,
    ).sort_index()
    aligned["desired_position"] = aligned["desired_position"].ffill().fillna(0.0)
    aligned["strategy_equity"] = aligned["strategy_equity"].ffill()
    aligned["qqq_close"] = aligned["qqq_close"].ffill()

    if not params.use_drawdown_governor:
        out = pd.DataFrame(index=aligned.index)
        out["governed_position"] = aligned["desired_position"]
        out["governor_active"] = 0
        out["portfolio_drawdown"] = np.nan
        out["qqq_drawdown"] = np.nan
        out["recovery_signal"] = 0
        out["restore_fraction"] = 1.0
        out["days_reduced"] = 0
        return out.reindex(desired.index)

    portfolio_equity = aligned["strategy_equity"]
    portfolio_drawdown = portfolio_equity / portfolio_equity.cummax() - 1.0
    qqq = aligned["qqq_close"]
    qqq_drawdown = qqq / qqq.cummax() - 1.0
    recovery_ma = compute_ma(qqq, int(params.recovery_ma_window), "sma")
    recovery_momentum = qqq.pct_change(int(params.recovery_momentum_window))
    recovery_signal = (
        (qqq > recovery_ma)
        & (recovery_momentum > float(params.recovery_momentum_threshold))
    ).fillna(False)

    governed = aligned["desired_position"].copy()
    active_flags = pd.Series(0, index=aligned.index, dtype=int)
    restore_fraction = pd.Series(1.0, index=aligned.index, dtype=float)
    days_reduced_series = pd.Series(0, index=aligned.index, dtype=int)

    active = False
    days_reduced = 0
    qqq_low_since_reduction = np.nan
    rearm_after_qqq_low = np.nan
    reduced_exposure = max(float(params.reduced_exposure), 0.0)
    max_days_reduced = max(int(params.max_days_reduced), 1)

    for idx in aligned.index:
        desired_value = float(aligned.at[idx, "desired_position"])
        qqq_value = float(qqq.loc[idx]) if pd.notna(qqq.loc[idx]) else np.nan

        if pd.notna(rearm_after_qqq_low) and pd.notna(qqq_value) and qqq_value < rearm_after_qqq_low:
            rearm_after_qqq_low = np.nan

        trigger = (
            pd.notna(portfolio_drawdown.loc[idx])
            and portfolio_drawdown.loc[idx] <= float(params.portfolio_dd_trigger)
        ) or (
            pd.notna(qqq_drawdown.loc[idx])
            and qqq_drawdown.loc[idx] <= float(params.qqq_dd_trigger)
        )

        if trigger and not active and pd.isna(rearm_after_qqq_low):
            active = True
            days_reduced = 0
            qqq_low_since_reduction = qqq_value

        new_low = False
        if active and pd.notna(qqq_value):
            if pd.isna(qqq_low_since_reduction) or qqq_value < qqq_low_since_reduction:
                qqq_low_since_reduction = qqq_value
                new_low = True
                if days_reduced >= max_days_reduced:
                    days_reduced = 0

        if active and bool(recovery_signal.loc[idx]):
            active = False
            days_reduced = 0
            rearm_after_qqq_low = qqq_low_since_reduction
            restore_fraction.loc[idx] = 1.0
            governed.loc[idx] = desired_value
        elif active:
            if days_reduced >= max_days_reduced and not new_low:
                fraction = min(
                    (days_reduced - max_days_reduced + 1) / float(RESTORE_RAMP_DAYS),
                    1.0,
                )
                cap = reduced_exposure + max(desired_value - reduced_exposure, 0.0) * fraction
                governed.loc[idx] = min(desired_value, cap)
                restore_fraction.loc[idx] = fraction
            else:
                governed.loc[idx] = min(desired_value, reduced_exposure)
                restore_fraction.loc[idx] = 0.0

            active_flags.loc[idx] = 1
            days_reduced_series.loc[idx] = days_reduced
            days_reduced += 1
        else:
            governed.loc[idx] = desired_value
            restore_fraction.loc[idx] = 1.0

    out = pd.DataFrame(
        {
            "governed_position": governed,
            "governor_active": active_flags,
            "portfolio_drawdown": portfolio_drawdown,
            "qqq_drawdown": qqq_drawdown,
            "recovery_signal": recovery_signal.astype(int),
            "restore_fraction": restore_fraction,
            "days_reduced": days_reduced_series,
        },
        index=aligned.index,
    )
    return out.reindex(desired.index)


def make_drawdown_governor_param_grid(
    use_drawdown_governor_options: Iterable[bool] = (False,),
    portfolio_dd_triggers: Iterable[float] = (-0.30,),
    qqq_dd_triggers: Iterable[float] = (-0.15,),
    reduced_exposures: Iterable[float] = (0.50,),
    recovery_ma_windows: Iterable[int] = (20,),
    recovery_momentum_windows: Iterable[int] = (10,),
    recovery_momentum_thresholds: Iterable[float] = (0.05,),
    max_days_reduced_options: Iterable[int] = (40,),
) -> List[DrawdownGovernorParams]:
    grid: List[DrawdownGovernorParams] = []
    for use_governor in use_drawdown_governor_options:
        if not use_governor:
            grid.append(DrawdownGovernorParams(use_drawdown_governor=False))
            continue
        for portfolio_dd_trigger in portfolio_dd_triggers:
            for qqq_dd_trigger in qqq_dd_triggers:
                for reduced_exposure in reduced_exposures:
                    if float(reduced_exposure) < 0.0:
                        continue
                    for recovery_ma_window in recovery_ma_windows:
                        if int(recovery_ma_window) <= 1:
                            continue
                        for recovery_momentum_window in recovery_momentum_windows:
                            if int(recovery_momentum_window) <= 0:
                                continue
                            for recovery_momentum_threshold in recovery_momentum_thresholds:
                                for max_days_reduced in max_days_reduced_options:
                                    if int(max_days_reduced) <= 0:
                                        continue
                                    grid.append(
                                        DrawdownGovernorParams(
                                            use_drawdown_governor=True,
                                            portfolio_dd_trigger=float(portfolio_dd_trigger),
                                            qqq_dd_trigger=float(qqq_dd_trigger),
                                            reduced_exposure=float(reduced_exposure),
                                            recovery_ma_window=int(recovery_ma_window),
                                            recovery_momentum_window=int(recovery_momentum_window),
                                            recovery_momentum_threshold=float(recovery_momentum_threshold),
                                            max_days_reduced=int(max_days_reduced),
                                        )
                                    )
    return grid
