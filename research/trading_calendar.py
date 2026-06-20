from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Iterable

import pandas as pd


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current + timedelta(days=7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=None)
def _minimal_nyse_holidays(year: int) -> frozenset[date]:
    """Minimal full-day NYSE holiday fallback.

    Prefer the installed exchange calendar when available. This fallback covers
    standard recent NYSE full-day holidays but intentionally does not model
    unscheduled market closures or early closes.
    """

    return frozenset(
        {
            _observed_fixed_holiday(year, 1, 1),
            _nth_weekday(year, 1, 0, 3),
            _nth_weekday(year, 2, 0, 3),
            _easter_sunday(year) - timedelta(days=2),
            _last_weekday(year, 5, 0),
            _observed_fixed_holiday(year, 6, 19),
            _observed_fixed_holiday(year, 7, 4),
            _nth_weekday(year, 9, 0, 1),
            _nth_weekday(year, 11, 3, 4),
            _observed_fixed_holiday(year, 12, 25),
        }
    )


def _fallback_nyse_sessions(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in _minimal_nyse_holidays(current.year):
            yield current
        current += timedelta(days=1)


def nyse_trading_days_between(latest_price_date: pd.Timestamp, as_of_date: pd.Timestamp) -> int:
    """Count completed NYSE sessions after latest_price_date and before as_of_date."""

    latest = pd.Timestamp(latest_price_date).normalize().date()
    as_of = pd.Timestamp(as_of_date).normalize().date()
    if as_of <= latest:
        return 0
    start = latest + timedelta(days=1)
    end = as_of - timedelta(days=1)
    if end < start:
        return 0

    try:
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS")
        sessions = calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return int(len(sessions))
    except Exception:
        return int(sum(1 for _ in _fallback_nyse_sessions(start, end)))

