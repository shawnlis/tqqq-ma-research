from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf

from .data import _selected_price_columns
from .execution import OHLC_FIELDS, ohlc_column
from .trading_calendar import nyse_trading_days_between


@dataclass(frozen=True)
class MarketDataRefreshResult:
    attempted: bool
    refresh_success: bool
    source: str
    status: str
    cache_path: Path
    latest_price_date: str
    calendar_stale_days: int
    trading_stale_days: int
    stale_days: int
    max_allowed_stale_days: int
    refreshed_rows: int
    error: str
    generated_at: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "refresh_success": self.refresh_success,
            "source": self.source,
            "status": self.status,
            "cache_path": str(self.cache_path),
            "latest_price_date": self.latest_price_date,
            "calendar_stale_days": self.calendar_stale_days,
            "trading_stale_days": self.trading_stale_days,
            "stale_days": self.stale_days,
            "max_allowed_stale_days": self.max_allowed_stale_days,
            "refreshed_rows": self.refreshed_rows,
            "error": self.error,
            "generated_at": self.generated_at,
        }


def _cache_path(cache_dir: str, symbols: tuple[str, ...]) -> Path:
    cache_symbols = tuple(symbol for symbol in symbols if symbol != "CASH")
    cache_name = (
        "tqqq_qqq_ohlc.csv"
        if cache_symbols == ("TQQQ", "QQQ")
        else f"{'_'.join(symbol.lower() for symbol in cache_symbols)}_ohlc.csv"
    )
    return Path(cache_dir) / cache_name


def _download_ohlc(symbols: tuple[str, ...], start: str) -> pd.DataFrame:
    raw = yf.download(
        list(symbols),
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    values: Dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        available_top = {str(value).upper(): value for value in raw.columns.get_level_values(0)}
        for symbol in symbols:
            raw_symbol = available_top.get(symbol)
            if raw_symbol is None or "Close" not in raw[raw_symbol].columns:
                continue
            values[symbol] = raw[raw_symbol]["Close"]
            for field in OHLC_FIELDS:
                field_name = field.title()
                if field_name in raw[raw_symbol].columns:
                    values[ohlc_column(symbol, field)] = raw[raw_symbol][field_name]
    elif len(symbols) == 1 and "Close" in raw.columns:
        symbol = symbols[0]
        values[symbol] = raw["Close"]
        for field in OHLC_FIELDS:
            field_name = field.title()
            if field_name in raw.columns:
                values[ohlc_column(symbol, field)] = raw[field_name]
    if not values:
        return pd.DataFrame()
    out = pd.DataFrame(values).dropna(how="all")
    out.index.name = "Date"
    return out


def _latest_stale_days(frame: pd.DataFrame, as_of_date: Optional[pd.Timestamp]) -> tuple[str, int, int]:
    if frame.empty:
        return "", 999999, 999999
    latest = pd.Timestamp(frame.index.max()).normalize()
    as_of = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.today().normalize()
    calendar_stale_days = int(max((as_of - latest).days, 0))
    trading_stale_days = nyse_trading_days_between(latest, as_of)
    return latest.date().isoformat(), calendar_stale_days, trading_stale_days


def refresh_recent_market_data(
    config: Dict[str, Any],
    *,
    data_csv: Optional[Path] = None,
    as_of_date: Optional[pd.Timestamp] = None,
) -> MarketDataRefreshResult:
    generated_at = datetime.now(timezone.utc).isoformat()
    symbols = tuple(dict.fromkeys([str(config.get("target_symbol", "TQQQ")).upper(), "QQQ"]))
    cache_dir = str(config.get("cache_dir", "./price_cache"))
    cache_path = _cache_path(cache_dir, symbols)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    max_stale = int(config.get("max_allowed_stale_days", config.get("stale_data_warning_days", 1)))

    if data_csv is not None:
        data = pd.read_csv(data_csv, parse_dates=["Date"]).set_index("Date").sort_index()
        latest, calendar_stale_days, trading_stale_days = _latest_stale_days(data, as_of_date)
        return MarketDataRefreshResult(
            attempted=False,
            refresh_success=False,
            source="data_csv",
            status="ok" if trading_stale_days <= max_stale else "stale_data",
            cache_path=Path(data_csv),
            latest_price_date=latest,
            calendar_stale_days=calendar_stale_days,
            trading_stale_days=trading_stale_days,
            stale_days=trading_stale_days,
            max_allowed_stale_days=max_stale,
            refreshed_rows=0,
            error="",
            generated_at=generated_at,
        )

    cached = pd.DataFrame()
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date").sort_index()

    attempted = bool(config.get("auto_refresh_market_data", False))
    refreshed_rows = 0
    error = ""
    refresh_success = False
    source = "cache"
    combined = cached.copy()

    if attempted:
        try:
            as_of = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.today().normalize()
            lookback_start = (as_of - pd.Timedelta(days=int(config.get("refresh_lookback_days", 10)))).date().isoformat()
            refreshed = _download_ohlc(symbols, lookback_start)
            refreshed_rows = int(len(refreshed))
            if not refreshed.empty:
                combined = pd.concat([cached, refreshed]).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined[_selected_price_columns(combined, symbols, include_ohlc=True)]
                combined.to_csv(cache_path)
                refresh_success = True
                source = "refresh"
            else:
                error = "refresh returned no rows"
                source = "failed_refresh"
        except Exception as exc:
            error = str(exc)
            source = "failed_refresh"

    latest, calendar_stale_days, trading_stale_days = _latest_stale_days(combined, as_of_date)
    status = "ok" if trading_stale_days <= max_stale else "stale_data"
    return MarketDataRefreshResult(
        attempted=attempted,
        refresh_success=refresh_success,
        source=source,
        status=status,
        cache_path=cache_path,
        latest_price_date=latest,
        calendar_stale_days=calendar_stale_days,
        trading_stale_days=trading_stale_days,
        stale_days=trading_stale_days,
        max_allowed_stale_days=max_stale,
        refreshed_rows=refreshed_rows,
        error=error,
        generated_at=generated_at,
    )
