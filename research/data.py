from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from .execution import OHLC_FIELDS, ohlc_column

logger = logging.getLogger(__name__)
CASH_SYMBOL = "CASH"


def add_cash_series(
    df: pd.DataFrame,
    symbol: str = CASH_SYMBOL,
    initial_value: float = 1.0,
    include_ohlc: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    cash_symbol = str(symbol).upper()
    out[cash_symbol] = float(initial_value)
    if include_ohlc:
        for field in OHLC_FIELDS:
            out[ohlc_column(cash_symbol, field)] = float(initial_value)
    return out


def _selected_price_columns(
    df: pd.DataFrame,
    symbols: Sequence[str],
    include_ohlc: bool,
) -> list[str]:
    columns: list[str] = []
    for symbol in symbols:
        if symbol in df.columns:
            columns.append(symbol)
        if include_ohlc:
            columns.extend(column for column in ohlc_column_set(symbol) if column in df.columns)
    return list(dict.fromkeys(columns))


def ohlc_column_set(symbol: str) -> list[str]:
    return [ohlc_column(symbol, field) for field in OHLC_FIELDS]


def load_prices(
    start: str = "2011-01-01",
    end: Optional[str] = None,
    symbols: Sequence[str] = ("TQQQ", "QQQ"),
    use_csv_if_exists: bool = True,
    cache_dir: str = "./price_cache",
    dropna: bool = True,
    allow_missing_symbols: bool = False,
    include_ohlc: bool = False,
) -> pd.DataFrame:
    """
    Download adjusted close prices, optionally including adjusted OHLC fields.
    """
    symbols = tuple(str(symbol).upper() for symbol in symbols)
    if not symbols:
        raise ValueError("At least one symbol is required.")
    non_cash_symbols = tuple(symbol for symbol in symbols if symbol != CASH_SYMBOL)
    wants_cash = CASH_SYMBOL in symbols

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_symbols = non_cash_symbols
    cache_name = (
        "tqqq_qqq_prices.csv"
        if cache_symbols == ("TQQQ", "QQQ") and not include_ohlc
        else f"{'_'.join(s.lower() for s in cache_symbols)}_{'ohlc' if include_ohlc else 'prices'}.csv"
    )
    file_path = cache_path / cache_name

    if non_cash_symbols and use_csv_if_exists and file_path.exists():
        df = pd.read_csv(file_path, parse_dates=["Date"]).set_index("Date")
        missing = set(non_cash_symbols) - set(df.columns)
        if missing and not allow_missing_symbols:
            raise ValueError(f"Cached price file is missing symbols: {sorted(missing)}")
        available_symbols = [symbol for symbol in non_cash_symbols if symbol in df.columns]
        if not available_symbols:
            raise ValueError(f"Cached price file contains none of the requested symbols: {file_path}")
        df = df[_selected_price_columns(df, available_symbols, include_ohlc)].sort_index()
        if dropna:
            df = df.dropna(subset=available_symbols)
        if wants_cash:
            df = add_cash_series(df, include_ohlc=include_ohlc)
            df = df[_selected_price_columns(df, symbols, include_ohlc)]
        logger.info("Loaded cached prices from %s", file_path)
        return df

    if not non_cash_symbols:
        end_value = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        index = pd.bdate_range(pd.Timestamp(start), end_value)
        df = pd.DataFrame(index=index)
        df.index.name = "Date"
        df = add_cash_series(df, include_ohlc=include_ohlc)
        return df[_selected_price_columns(df, symbols, include_ohlc)]

    logger.info("Downloading prices from Yahoo Finance...")
    raw = yf.download(
        list(non_cash_symbols),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise ValueError("No price data downloaded.")

    if isinstance(raw.columns, pd.MultiIndex):
        available_top = {
            str(value).upper(): value
            for value in raw.columns.get_level_values(0)
        }
        values = {}
        for symbol in non_cash_symbols:
            raw_symbol = available_top.get(symbol)
            if raw_symbol is not None and "Close" in raw[raw_symbol].columns:
                values[symbol] = raw[raw_symbol]["Close"]
                if include_ohlc:
                    for field in OHLC_FIELDS:
                        if field.title() in raw[raw_symbol].columns:
                            values[ohlc_column(symbol, field)] = raw[raw_symbol][field.title()]
            elif not allow_missing_symbols:
                raise ValueError(f"No close prices downloaded for symbol: {symbol}")
            else:
                logger.warning("Skipping missing downloaded symbol: %s", symbol)
        if not values:
            raise ValueError("No requested symbol close prices downloaded.")
        df = pd.DataFrame(values)
    elif len(non_cash_symbols) == 1 and "Close" in raw.columns:
        symbol = non_cash_symbols[0]
        values = {symbol: raw["Close"]}
        if include_ohlc:
            for field in OHLC_FIELDS:
                field_name = field.title()
                if field_name in raw.columns:
                    values[ohlc_column(symbol, field)] = raw[field_name]
        df = pd.DataFrame(values)
    else:
        raise ValueError("Unexpected yfinance column format.")

    if dropna:
        df = df.dropna()
    else:
        df = df.dropna(how="all")

    df.index.name = "Date"
    df.to_csv(file_path)
    if wants_cash:
        df = add_cash_series(df, include_ohlc=include_ohlc)
        df = df[_selected_price_columns(df, symbols, include_ohlc)]
    logger.info("Saved price cache to %s", file_path)
    return df


def make_synthetic_leveraged_returns(
    base_returns: pd.Series,
    leverage: float = 3.0,
    annual_expense_ratio: float = 0.0095,
    annual_financing_spread: float = 0.0,
) -> pd.Series:
    """
    Approximate daily leveraged ETF returns from base asset daily returns.
    """
    returns = pd.Series(base_returns, copy=True).astype(float)
    daily_expense_drag = float(annual_expense_ratio) / 252.0
    daily_financing_drag = max(float(leverage) - 1.0, 0.0) * float(annual_financing_spread) / 252.0
    synthetic = float(leverage) * returns - daily_expense_drag - daily_financing_drag
    synthetic.name = "synthetic_leveraged_return"
    return synthetic


def make_synthetic_leveraged_price(
    base_prices: pd.Series,
    leverage: float = 3.0,
    annual_expense_ratio: float = 0.0095,
    annual_financing_spread: float = 0.0,
    initial_value: float = 1.0,
) -> pd.Series:
    prices = pd.Series(base_prices, copy=True).astype(float).dropna()
    if prices.empty:
        raise ValueError("base_prices is empty.")

    base_returns = prices.pct_change().fillna(0.0)
    synthetic_returns = make_synthetic_leveraged_returns(
        base_returns=base_returns,
        leverage=leverage,
        annual_expense_ratio=annual_expense_ratio,
        annual_financing_spread=annual_financing_spread,
    )
    synthetic_returns.iloc[0] = 0.0
    synthetic_price = float(initial_value) * (1.0 + synthetic_returns).cumprod()
    synthetic_price = synthetic_price.replace([np.inf, -np.inf], np.nan).dropna()
    synthetic_price.name = "synthetic_leveraged_price"
    return synthetic_price
