from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..assets import AssetConfig, DEFAULT_ASSET_CONFIG


DEFAULT_RISK_ASSETS: Tuple[str, ...] = ("TQQQ", "SOXL", "UPRO", "TECL")
DEFAULT_DEFENSIVE_ASSETS: Tuple[str, ...] = ("CASH", "SGOV", "BIL", "QQQ", "SPY")


@dataclass(frozen=True)
class RotationParams:
    risk_assets: Tuple[str, ...] = DEFAULT_RISK_ASSETS
    rebalance_frequency: str = "monthly"
    top_n: int = 1
    max_asset_weight: float = 1.0
    max_total_leveraged_exposure: float = 1.0
    trend_filter_symbol: str = "QQQ"
    trend_filter_window: int = 200
    defensive_asset: str = "CASH"
    require_positive_momentum: bool = True
    asset_config: AssetConfig = DEFAULT_ASSET_CONFIG


def _normalize_symbols(symbols: Iterable[str]) -> Tuple[str, ...]:
    return tuple(str(symbol).upper() for symbol in symbols)


def available_rotation_risk_assets(data: pd.DataFrame, risk_assets: Sequence[str]) -> Tuple[str, ...]:
    return tuple(symbol for symbol in _normalize_symbols(risk_assets) if symbol in data.columns)


def _rebalance_mask(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    frequency = str(frequency).lower()
    if frequency not in {"weekly", "monthly"}:
        raise ValueError(f"Unsupported rebalance_frequency: {frequency}")

    frame = pd.DataFrame(index=index)
    if frequency == "weekly":
        keys = pd.MultiIndex.from_arrays([index.isocalendar().year, index.isocalendar().week])
    else:
        keys = pd.MultiIndex.from_arrays([index.year, index.month])
    last_dates = frame.groupby(keys).tail(1).index
    return pd.Series(index.isin(last_dates), index=index, name="is_rebalance_date")


def build_rotation_signal(data: pd.DataFrame, params: RotationParams) -> pd.DataFrame:
    """
    Build unshifted target weights from close-of-day ranking data.

    Backtests shift these weights by one bar before applying returns, so the
    rebalance decision formed at close t is held starting on t+1.
    """
    risk_assets = available_rotation_risk_assets(data, params.risk_assets)
    if not risk_assets:
        raise ValueError("No configured rotation risk assets are available in price data.")

    defensive_asset = str(params.defensive_asset).upper()
    trend_filter_symbol = str(params.trend_filter_symbol).upper()
    if defensive_asset not in data.columns:
        raise ValueError(f"Defensive asset is missing from price data: {defensive_asset}")
    if trend_filter_symbol not in data.columns:
        raise ValueError(f"Trend filter symbol is missing from price data: {trend_filter_symbol}")

    prices = data[list(risk_assets)].astype(float)
    returns = prices.pct_change()
    mom20 = prices.pct_change(20)
    mom63 = prices.pct_change(63)
    mom126 = prices.pct_change(126)
    vol20 = returns.rolling(20, min_periods=20).std(ddof=0)
    momentum_avg = (mom20 + mom63 + mom126) / 3.0
    momentum_vol_score = momentum_avg / vol20.replace(0.0, np.nan)
    drawdown_63 = prices / prices.rolling(63, min_periods=63).max() - 1.0
    rank_score = (
        mom20.rank(axis=1, pct=True, ascending=True)
        + mom63.rank(axis=1, pct=True, ascending=True)
        + mom126.rank(axis=1, pct=True, ascending=True)
        + vol20.rank(axis=1, pct=True, ascending=False)
        + momentum_vol_score.rank(axis=1, pct=True, ascending=True)
        + drawdown_63.rank(axis=1, pct=True, ascending=True)
    ) / 6.0

    trend_prices = data[trend_filter_symbol].astype(float)
    trend_ma = trend_prices.rolling(
        int(params.trend_filter_window),
        min_periods=int(params.trend_filter_window),
    ).mean()
    trend_allows_risk = (trend_prices > trend_ma).fillna(False)
    is_rebalance = _rebalance_mask(data.index, params.rebalance_frequency)

    traded_assets = list(dict.fromkeys([*risk_assets, defensive_asset]))
    weights = pd.DataFrame(0.0, index=data.index, columns=traded_assets)
    selected_assets: List[str] = []
    selected_scores: List[float] = []
    current_weights = pd.Series(0.0, index=traded_assets, dtype=float)
    current_selected = "NONE"
    current_score = np.nan

    top_n = max(int(params.top_n), 1)
    max_asset_weight = float(np.clip(float(params.max_asset_weight), 0.0, 1.0))
    max_total_leveraged_exposure = max(float(params.max_total_leveraged_exposure), 0.0)

    for date in data.index:
        if bool(is_rebalance.loc[date]):
            current_weights = pd.Series(0.0, index=traded_assets, dtype=float)
            current_selected = defensive_asset
            current_score = np.nan

            if bool(trend_allows_risk.loc[date]):
                scores = rank_score.loc[date].dropna().sort_values(ascending=False)
                if not scores.empty:
                    if params.require_positive_momentum:
                        positive = momentum_avg.loc[date].reindex(scores.index) > 0.0
                        scores = scores.loc[positive.fillna(False)]

                    winners = list(scores.head(top_n).index)
                    if winners:
                        per_asset_weight = min(
                            max_asset_weight,
                            max_total_leveraged_exposure / float(len(winners)),
                        )
                        for symbol in winners:
                            current_weights.loc[symbol] = per_asset_weight
                        risk_total = float(current_weights.reindex(risk_assets).sum())
                        if risk_total < 1.0:
                            current_weights.loc[defensive_asset] += 1.0 - risk_total
                        current_selected = ";".join(winners)
                        current_score = float(scores.loc[winners].mean())
                    else:
                        current_weights.loc[defensive_asset] = 1.0
                else:
                    current_weights.loc[defensive_asset] = 1.0
            else:
                current_weights.loc[defensive_asset] = 1.0

        weights.loc[date, traded_assets] = current_weights
        selected_assets.append(current_selected)
        selected_scores.append(current_score)

    out = weights.add_prefix("target_weight_")
    out["selected_assets"] = selected_assets
    out["selected_score"] = selected_scores
    out["is_rebalance_date"] = is_rebalance.astype(int)
    out["trend_filter_symbol"] = trend_filter_symbol
    out["trend_filter_window"] = int(params.trend_filter_window)
    out["trend_filter_allows_risk"] = trend_allows_risk.astype(int)
    out["available_risk_assets"] = ";".join(risk_assets)
    return out


def make_rotation_param_grid(
    risk_asset_sets: Iterable[Sequence[str]] = (DEFAULT_RISK_ASSETS,),
    rebalance_frequencies: Iterable[str] = ("monthly",),
    top_ns: Iterable[int] = (1,),
    max_asset_weights: Iterable[float] = (1.0,),
    max_total_leveraged_exposures: Iterable[float] = (1.0,),
    trend_filter_symbols: Iterable[str] = ("QQQ",),
    trend_filter_windows: Iterable[int] = (200,),
    defensive_assets: Iterable[str] = ("CASH",),
    require_positive_momentum_options: Iterable[bool] = (True,),
    asset_config: AssetConfig = DEFAULT_ASSET_CONFIG,
) -> List[RotationParams]:
    grid: List[RotationParams] = []
    for risk_assets in risk_asset_sets:
        parsed_risk_assets = _normalize_symbols(risk_assets)
        if not parsed_risk_assets:
            continue
        for rebalance_frequency in rebalance_frequencies:
            frequency = str(rebalance_frequency).lower()
            if frequency not in {"weekly", "monthly"}:
                continue
            for top_n in top_ns:
                if int(top_n) < 1:
                    continue
                for max_asset_weight in max_asset_weights:
                    if float(max_asset_weight) <= 0.0:
                        continue
                    for max_total in max_total_leveraged_exposures:
                        if float(max_total) <= 0.0:
                            continue
                        for trend_symbol in trend_filter_symbols:
                            for trend_window in trend_filter_windows:
                                if int(trend_window) <= 1:
                                    continue
                                for defensive_asset in defensive_assets:
                                    for require_positive in require_positive_momentum_options:
                                        grid.append(
                                            RotationParams(
                                                risk_assets=parsed_risk_assets,
                                                rebalance_frequency=frequency,
                                                top_n=int(top_n),
                                                max_asset_weight=float(max_asset_weight),
                                                max_total_leveraged_exposure=float(max_total),
                                                trend_filter_symbol=str(trend_symbol).upper(),
                                                trend_filter_window=int(trend_window),
                                                defensive_asset=str(defensive_asset).upper(),
                                                require_positive_momentum=bool(require_positive),
                                                asset_config=asset_config.normalized(),
                                            )
                                        )
    return grid
