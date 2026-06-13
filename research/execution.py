from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


CLOSE_TO_CLOSE_SHIFTED = "close_to_close_shifted"
NEXT_OPEN_TO_CLOSE = "next_open_to_close"
NEXT_OPEN_TO_NEXT_OPEN = "next_open_to_next_open"
EXECUTION_MODELS = (
    CLOSE_TO_CLOSE_SHIFTED,
    NEXT_OPEN_TO_CLOSE,
    NEXT_OPEN_TO_NEXT_OPEN,
)
OHLC_FIELDS = ("OPEN", "HIGH", "LOW", "CLOSE")


def normalize_execution_model(execution_model: str | None) -> str:
    model = str(execution_model or CLOSE_TO_CLOSE_SHIFTED).strip().lower()
    aliases = {
        "close": CLOSE_TO_CLOSE_SHIFTED,
        "close_to_close": CLOSE_TO_CLOSE_SHIFTED,
        "close-to-close-shifted": CLOSE_TO_CLOSE_SHIFTED,
        "next-open-to-close": NEXT_OPEN_TO_CLOSE,
        "open_to_close": NEXT_OPEN_TO_CLOSE,
        "next-open-to-next-open": NEXT_OPEN_TO_NEXT_OPEN,
        "open_to_open": NEXT_OPEN_TO_NEXT_OPEN,
        "next_open_to_open": NEXT_OPEN_TO_NEXT_OPEN,
    }
    model = aliases.get(model, model)
    if model not in EXECUTION_MODELS:
        raise ValueError(f"Unsupported execution_model: {execution_model}")
    return model


def ohlc_column(symbol: str, field: str) -> str:
    return f"{str(symbol).upper()}_{str(field).upper()}"


def ohlc_columns(symbol: str) -> list[str]:
    return [ohlc_column(symbol, field) for field in OHLC_FIELDS]


def has_ohlc(data: pd.DataFrame, symbol: str) -> bool:
    symbol = str(symbol).upper()
    if symbol == "CASH":
        return True
    return ohlc_column(symbol, "OPEN") in data.columns and ohlc_column(symbol, "CLOSE") in data.columns


def execution_model_from_data(data: pd.DataFrame, requested: str | None = None) -> str:
    if requested is not None:
        return normalize_execution_model(requested)
    return normalize_execution_model(data.attrs.get("execution_model", CLOSE_TO_CLOSE_SHIFTED))


def resolve_execution_model(
    data: pd.DataFrame,
    requested: str | None,
    required_symbols: Iterable[str],
) -> Tuple[str, str]:
    requested_model = execution_model_from_data(data, requested)
    if requested_model == CLOSE_TO_CLOSE_SHIFTED:
        return requested_model, ""

    missing = sorted(
        {
            str(symbol).upper()
            for symbol in required_symbols
            if str(symbol).upper() != "CASH" and not has_ohlc(data, str(symbol).upper())
        }
    )
    if not missing:
        return requested_model, ""

    warning = (
        f"Execution model {requested_model} requires adjusted OHLC data; "
        f"missing OHLC for {', '.join(missing)}. Fell back to {CLOSE_TO_CLOSE_SHIFTED}."
    )
    return CLOSE_TO_CLOSE_SHIFTED, warning


def asset_returns(
    data: pd.DataFrame,
    symbol: str,
    execution_model: str | None = None,
) -> pd.Series:
    symbol = str(symbol).upper()
    model = execution_model_from_data(data, execution_model)
    if symbol == "CASH":
        return pd.Series(0.0, index=data.index, name=f"daily_ret_{symbol}")
    if symbol not in data.columns:
        raise ValueError(f"Symbol is missing from price data: {symbol}")

    if model == CLOSE_TO_CLOSE_SHIFTED:
        returns = data[symbol].astype(float).pct_change(fill_method=None).fillna(0.0)
    elif model == NEXT_OPEN_TO_CLOSE:
        open_col = ohlc_column(symbol, "OPEN")
        close_col = ohlc_column(symbol, "CLOSE")
        returns = data[close_col].astype(float) / data[open_col].astype(float) - 1.0
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elif model == NEXT_OPEN_TO_NEXT_OPEN:
        open_col = ohlc_column(symbol, "OPEN")
        returns = data[open_col].astype(float).shift(-1) / data[open_col].astype(float) - 1.0
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        raise ValueError(f"Unsupported execution_model: {execution_model}")

    returns.name = f"daily_ret_{symbol}"
    return returns.astype(float)


def requested_execution_model_from_data(data: pd.DataFrame) -> str:
    return normalize_execution_model(data.attrs.get("requested_execution_model", data.attrs.get("execution_model")))
