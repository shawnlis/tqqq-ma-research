from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MARKET_INTERNALS_SYMBOLS: Tuple[str, ...] = (
    "SPY",
    "QQQ",
    "RSP",
    "IWM",
    "HYG",
    "LQD",
    "IEF",
    "TLT",
    "^VIX",
    "XLK",
    "SMH",
    "SOXX",
)

DEFAULT_COMPONENTS: Tuple[str, ...] = (
    "qqq_trend",
    "spy_trend",
    "equal_weight_strength",
    "small_cap_strength",
    "credit_risk",
    "duration_stress",
    "tech_strength",
    "semiconductor_strength",
    "vix_regime",
)


@dataclass(frozen=True)
class MarketInternalsParams:
    use_market_internals: bool = False
    signal_window: int = 100
    risk_on_threshold: float = 0.60
    risk_off_threshold: float = 0.40
    score_method: str = "binary"
    components: Tuple[str, ...] = DEFAULT_COMPONENTS
    weights: Mapping[str, float] = field(default_factory=dict)
    min_components: int = 1
    vix_risk_on_threshold: float = 25.0
    vix_risk_off_threshold: float = 35.0


def normalize_market_symbol(symbol: Any) -> str:
    return str(symbol).upper()


def requested_market_internals_symbols(extra_symbols: Optional[Iterable[str]] = None) -> List[str]:
    symbols = [normalize_market_symbol(symbol) for symbol in MARKET_INTERNALS_SYMBOLS]
    if extra_symbols is not None:
        symbols.extend(normalize_market_symbol(symbol) for symbol in extra_symbols)
    return list(dict.fromkeys(symbols))


def _column(data: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
    symbol = normalize_market_symbol(symbol)
    if symbol in data.columns:
        return data[symbol].astype(float)
    for column in data.columns:
        if normalize_market_symbol(column) == symbol:
            return data[column].astype(float)
    return None


def _ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(int(window), min_periods=int(window)).mean()


def _binary_above_ma(series: pd.Series, window: int) -> pd.Series:
    ma = _ma(series, int(window))
    return (series > ma).where(ma.notna()).astype(float)


def _score_from_zscore(raw: pd.Series, window: int) -> pd.Series:
    mean = raw.rolling(int(window), min_periods=int(window)).mean()
    std = raw.rolling(int(window), min_periods=int(window)).std(ddof=0)
    z = (raw - mean) / std.replace(0.0, np.nan)
    return ((z.clip(lower=-3.0, upper=3.0) + 3.0) / 6.0).clip(0.0, 1.0)


def _ratio_component(
    data: pd.DataFrame,
    numerator: str,
    denominator: str,
    window: int,
    score_method: str,
) -> Optional[pd.Series]:
    left = _column(data, numerator)
    right = _column(data, denominator)
    if left is None or right is None:
        return None
    ratio = (left / right).replace([np.inf, -np.inf], np.nan)
    if ratio.dropna().shape[0] < int(window):
        return None
    if score_method == "zscore":
        return _score_from_zscore(ratio, int(window))
    return _binary_above_ma(ratio, int(window))


def _qqq_trend(data: pd.DataFrame, score_method: str) -> Optional[pd.Series]:
    qqq = _column(data, "QQQ")
    if qqq is None:
        return None
    if qqq.dropna().shape[0] < 100:
        return None
    if score_method == "zscore":
        raw = qqq / _ma(qqq, 200) - 1.0
        return _score_from_zscore(raw, 100)

    parts = []
    for window in (100, 150, 200):
        if qqq.dropna().shape[0] >= window:
            parts.append(_binary_above_ma(qqq, window))
    if not parts:
        return None
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _spy_trend(data: pd.DataFrame, score_method: str) -> Optional[pd.Series]:
    spy = _column(data, "SPY")
    if spy is None or spy.dropna().shape[0] < 200:
        return None
    if score_method == "zscore":
        raw = spy / _ma(spy, 200) - 1.0
        return _score_from_zscore(raw, 100)
    return _binary_above_ma(spy, 200)


def _vix_regime(data: pd.DataFrame, params: MarketInternalsParams) -> Optional[pd.Series]:
    vix = _column(data, "^VIX")
    if vix is None or vix.dropna().empty:
        return None
    if params.score_method == "zscore":
        # Lower VIX is risk-on, so invert the z-score.
        score = 1.0 - _score_from_zscore(vix, int(params.signal_window))
        return score.clip(0.0, 1.0)

    out = pd.Series(0.5, index=vix.index, dtype=float)
    out.loc[vix <= float(params.vix_risk_on_threshold)] = 1.0
    out.loc[vix >= float(params.vix_risk_off_threshold)] = 0.0
    out.loc[vix.isna()] = np.nan
    return out


def _component_series(data: pd.DataFrame, params: MarketInternalsParams) -> Dict[str, pd.Series]:
    window = int(params.signal_window)
    method = str(params.score_method).lower()
    if method not in {"binary", "zscore"}:
        raise ValueError(f"Unsupported market internals score_method: {params.score_method}")

    components: Dict[str, Optional[pd.Series]] = {
        "qqq_trend": _qqq_trend(data, method),
        "spy_trend": _spy_trend(data, method),
        "equal_weight_strength": _ratio_component(data, "RSP", "SPY", window, method),
        "small_cap_strength": _ratio_component(data, "IWM", "SPY", window, method),
        "credit_risk": _ratio_component(data, "HYG", "LQD", window, method),
        # For composite risk-on scoring, this is inverted: 1 means duration stress is absent.
        "duration_stress": None,
        "tech_strength": _ratio_component(data, "XLK", "SPY", window, method),
        "semiconductor_strength": _ratio_component(data, "SMH", "QQQ", window, method),
        "vix_regime": _vix_regime(data, params),
    }

    components["duration_stress"] = _ratio_component(data, "TLT", "IEF", window, method)

    if components["semiconductor_strength"] is None:
        components["semiconductor_strength"] = _ratio_component(data, "SOXX", "QQQ", window, method)

    selected = set(params.components or DEFAULT_COMPONENTS)
    return {
        f"mi_{name}": series.rename(f"mi_{name}")
        for name, series in components.items()
        if name in selected and series is not None
    }


def build_market_internals_signal(
    price_data: pd.DataFrame,
    params: MarketInternalsParams,
) -> pd.DataFrame:
    index = price_data.index
    component_map = _component_series(price_data, params) if params.use_market_internals else {}
    component_df = pd.DataFrame(component_map, index=index)

    if component_df.empty:
        risk_score = pd.Series(0.5, index=index, name="market_internals_risk_score")
        component_count = pd.Series(0, index=index, name="market_internals_component_count")
    else:
        weights = pd.Series(
            {
                f"mi_{name}": float(params.weights.get(name, 1.0))
                for name in params.components
                if f"mi_{name}" in component_df.columns
            },
            dtype=float,
        )
        weights = weights.reindex(component_df.columns).fillna(1.0)
        available = component_df.notna()
        score_components = component_df.copy()
        if "mi_duration_stress" in score_components.columns:
            score_components["mi_duration_stress"] = (
                1.0 - score_components["mi_duration_stress"]
            ).clip(0.0, 1.0)
        weighted_values = score_components.fillna(0.0).mul(weights, axis=1)
        active_weight = available.mul(weights, axis=1).sum(axis=1)
        risk_score = (weighted_values.sum(axis=1) / active_weight.replace(0.0, np.nan)).rename(
            "market_internals_risk_score"
        )
        component_count = available.sum(axis=1).rename("market_internals_component_count")
        risk_score = risk_score.where(component_count >= int(params.min_components), 0.5)
        risk_score = risk_score.fillna(0.5).clip(0.0, 1.0)

    out = pd.DataFrame(
        {
            "market_internals_risk_score": risk_score,
            "market_internals_component_count": component_count,
            "market_internals_risk_on": (
                risk_score >= float(params.risk_on_threshold)
            ).astype(int),
            "market_internals_risk_off": (
                risk_score <= float(params.risk_off_threshold)
            ).astype(int),
            "market_internals_signal_window": int(params.signal_window),
        },
        index=index,
    )
    for column in component_df.columns:
        out[column] = component_df[column]
    return out


def market_internals_availability(
    price_data: pd.DataFrame,
    max_required_window: int = 200,
    requested_symbols: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    requested = requested_market_internals_symbols(requested_symbols)
    skipped: List[Dict[str, Any]] = []
    available: List[str] = []

    for symbol in requested:
        series = _column(price_data, symbol)
        if series is None:
            skipped.append({"symbol": symbol, "reason": "missing"})
            continue
        non_null = series.dropna()
        if len(non_null) < int(max_required_window):
            skipped.append(
                {
                    "symbol": symbol,
                    "reason": "insufficient_history",
                    "non_null_rows": int(len(non_null)),
                    "first_date": non_null.index[0].date().isoformat() if not non_null.empty else None,
                    "last_date": non_null.index[-1].date().isoformat() if not non_null.empty else None,
                }
            )
            continue
        available.append(symbol)

    return {
        "requested_symbols": requested,
        "available_symbols": available,
        "skipped_symbols": skipped,
        "max_required_window": int(max_required_window),
    }


def make_market_internals_param_grid(
    use_market_internals_options: Iterable[bool] = (False,),
    signal_windows: Iterable[int] = (100,),
    risk_on_thresholds: Iterable[float] = (0.60,),
    risk_off_thresholds: Iterable[float] = (0.40,),
    component_sets: Optional[Iterable[Sequence[str]]] = None,
    weights_options: Optional[Iterable[Mapping[str, float]]] = None,
    score_methods: Iterable[str] = ("binary",),
    min_components_options: Iterable[int] = (1,),
    vix_risk_on_thresholds: Iterable[float] = (25.0,),
    vix_risk_off_thresholds: Iterable[float] = (35.0,),
) -> List[MarketInternalsParams]:
    if component_sets is None:
        component_sets = (DEFAULT_COMPONENTS,)
    if weights_options is None:
        weights_options = ({},)

    grid: List[MarketInternalsParams] = []
    for use_market_internals in use_market_internals_options:
        if not use_market_internals:
            grid.append(MarketInternalsParams(use_market_internals=False))
            continue
        for signal_window in signal_windows:
            if int(signal_window) <= 1:
                continue
            for risk_on_threshold in risk_on_thresholds:
                for risk_off_threshold in risk_off_thresholds:
                    if float(risk_off_threshold) > float(risk_on_threshold):
                        continue
                    for score_method in score_methods:
                        for min_components in min_components_options:
                            for vix_on in vix_risk_on_thresholds:
                                for vix_off in vix_risk_off_thresholds:
                                    for components in component_sets:
                                        parsed_components = tuple(
                                            str(component) for component in components
                                        )
                                        if not parsed_components:
                                            continue
                                        for weights in weights_options:
                                            grid.append(
                                                MarketInternalsParams(
                                                    use_market_internals=True,
                                                    signal_window=int(signal_window),
                                                    risk_on_threshold=float(risk_on_threshold),
                                                    risk_off_threshold=float(risk_off_threshold),
                                                    score_method=str(score_method),
                                                    components=parsed_components,
                                                    weights=dict(weights),
                                                    min_components=int(min_components),
                                                    vix_risk_on_threshold=float(vix_on),
                                                    vix_risk_off_threshold=float(vix_off),
                                                )
                                            )
    return grid
