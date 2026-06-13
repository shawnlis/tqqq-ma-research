from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from ..assets import AssetConfig, DEFAULT_ASSET_CONFIG


@dataclass(frozen=True)
class Params:
    short: int
    long: int
    ma_type: str                      # "sma" or "ema"
    signal_asset: str = "TQQQ"        # "TQQQ" or "QQQ"
    threshold: float = 0.0            # e.g. 0.005 => short_ma / long_ma - 1 > 0.5%
    cooldown_days: int = 0            # bars to wait after exit
    qqq_filter: bool = False
    qqq_filter_window: int = 200
    risk_off_symbol: str = "CASH"
    risk_off_weight: float = 0.0
    asset_config: AssetConfig = DEFAULT_ASSET_CONFIG


def compute_ma(series: pd.Series, window: int, ma_type: str) -> pd.Series:
    ma_type = ma_type.lower()
    if ma_type == "sma":
        return series.rolling(window=window, min_periods=window).mean()
    if ma_type == "ema":
        return series.ewm(span=window, adjust=False, min_periods=window).mean()
    raise ValueError(f"Unsupported ma_type: {ma_type}")


def apply_cooldown(raw_signal: pd.Series, cooldown_days: int) -> pd.Series:
    """
    Enforce a waiting period after each exit.

    If an effective exit happens on bar i, re-entry is blocked for the next
    cooldown_days bars: i+1 ... i+cooldown_days.
    Works for both binary and fractional desired positions.
    """
    if cooldown_days <= 0:
        return raw_signal.astype(float)

    raw = raw_signal.fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    n = raw.shape[0]
    out = np.empty_like(raw)

    prev_effective = 0.0
    blocked_until = -1

    for i in range(n):
        desired = raw[i]

        # Exit occurs today -> start blocking future bars
        if prev_effective > 0.0 and desired <= 0.0:
            blocked_until = i + cooldown_days

        if desired > 0.0 and i <= blocked_until:
            effective = 0.0
        else:
            effective = desired

        out[i] = effective
        prev_effective = effective

    return pd.Series(out, index=raw_signal.index, name="cooldown_signal")


def build_signal(
    tqqq_close: pd.Series,
    qqq_close: pd.Series,
    params: Params,
    filter_close: pd.Series | None = None,
) -> pd.Series:
    """
    Multi-tier desired position:
      - 1.00 in strong trend
      - 0.35 in weak trend
      - 0.00 in risk regime

    Optional:
      - signal can be generated from TQQQ or QQQ
      - QQQ regime filter
      - cooldown after exits
    """
    filter_close = qqq_close if filter_close is None else filter_close.reindex(qqq_close.index)
    asset_config = params.asset_config.normalized()
    signal_asset = params.signal_asset.upper()
    if signal_asset in {"TRADE", asset_config.trade_asset}:
        signal_source = tqqq_close
    elif signal_asset in {"PRIMARY", asset_config.primary_signal_asset}:
        signal_source = qqq_close
    else:
        raise ValueError(f"Unsupported signal_asset for configured assets: {params.signal_asset}")

    short_ma = compute_ma(signal_source, params.short, params.ma_type)
    long_ma = compute_ma(signal_source, params.long, params.ma_type)
    spread = short_ma / long_ma - 1.0

    # Use a medium trend check on TQQQ to avoid all-in during chop.
    medium_window = max(params.short + 2, min(params.long - 1, 30))
    tqqq_medium_ma = compute_ma(tqqq_close, medium_window, params.ma_type)
    tqqq_medium_ok = (tqqq_close > tqqq_medium_ma).fillna(False)

    if params.qqq_filter:
        qqq_filter_ma = compute_ma(filter_close, params.qqq_filter_window, "sma")
        qqq_regime_ok = (filter_close > qqq_filter_ma).fillna(False)
    else:
        qqq_regime_ok = pd.Series(True, index=qqq_close.index)

    strong_trend = (spread > params.threshold).fillna(False)
    weak_trend = (spread > 0.0).fillna(False)

    strong_condition = strong_trend & qqq_regime_ok & tqqq_medium_ok
    weak_condition = (~strong_condition) & weak_trend & (qqq_regime_ok | tqqq_medium_ok)

    raw_position = pd.Series(0.0, index=spread.index, name="raw_position")
    raw_position.loc[weak_condition] = 0.35
    raw_position.loc[strong_condition] = 1.0

    cooled_signal = apply_cooldown(raw_position.fillna(0.0), params.cooldown_days)
    return cooled_signal.fillna(0.0)


def make_param_grid(
    short_range: Iterable[int],
    long_range: Iterable[int],
    ma_types: Iterable[str] = ("sma", "ema"),
    signal_assets: Iterable[str] = ("TQQQ", "QQQ"),
    thresholds: Iterable[float] = (0.0,),
    cooldown_days_options: Iterable[int] = (0,),
    use_qqq_filter_options: Iterable[bool] = (False, True),
    qqq_filter_windows: Iterable[int] = (200,),
    risk_off_symbols: Iterable[str] = ("CASH",),
    risk_off_weights: Iterable[float] = (0.0,),
    asset_config: AssetConfig = DEFAULT_ASSET_CONFIG,
) -> List[Params]:
    grid: List[Params] = []
    qqq_filter_windows = tuple(qqq_filter_windows)

    for ma_type in ma_types:
        for signal_asset in signal_assets:
            for threshold in thresholds:
                for cooldown_days in cooldown_days_options:
                    for use_filter in use_qqq_filter_options:
                        for filter_window in qqq_filter_windows:
                            # If filter is off, qqq_filter_window has no effect.
                            if not use_filter and filter_window != qqq_filter_windows[0]:
                                continue

                            for short_window in short_range:
                                for long_window in long_range:
                                    if short_window >= long_window:
                                        continue

                                    for risk_off_symbol in risk_off_symbols:
                                        for risk_off_weight in risk_off_weights:
                                            grid.append(
                                                Params(
                                                    short=short_window,
                                                    long=long_window,
                                                    ma_type=ma_type,
                                                    signal_asset=signal_asset,
                                                    threshold=float(threshold),
                                                    cooldown_days=int(cooldown_days),
                                                    qqq_filter=use_filter,
                                                    qqq_filter_window=filter_window,
                                                    risk_off_symbol=str(risk_off_symbol).upper(),
                                                    risk_off_weight=float(risk_off_weight),
                                                    asset_config=asset_config.normalized(),
                                                )
                                        )
    return grid
