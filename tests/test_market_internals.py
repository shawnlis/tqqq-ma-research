import numpy as np
import pandas as pd

from research.strategies.market_internals import (
    MarketInternalsParams,
    build_market_internals_signal,
    market_internals_availability,
)


def _market_prices(rows: int = 260) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=rows)
    step = np.arange(rows, dtype=float)
    qqq = 100.0 + step * 0.20
    spy = 100.0 + step * 0.15
    return pd.DataFrame(
        {
            "QQQ": qqq,
            "SPY": spy,
            "RSP": spy * (1.0 + step * 0.0003),
            "IWM": spy * (1.0 + step * 0.0002),
            "HYG": 80.0 + step * 0.08,
            "LQD": 100.0 + step * 0.02,
            "TLT": 100.0 - step * 0.02,
            "IEF": 90.0 + step * 0.01,
            "XLK": spy * (1.0 + step * 0.0004),
            "SMH": qqq * (1.0 + step * 0.0005),
            "^VIX": np.full(rows, 18.0),
        },
        index=index,
    )


def test_market_internals_risk_score_uses_available_components() -> None:
    prices = _market_prices()
    params = MarketInternalsParams(
        use_market_internals=True,
        signal_window=20,
        risk_on_threshold=0.60,
        risk_off_threshold=0.40,
    )

    signal = build_market_internals_signal(prices, params)

    assert "market_internals_risk_score" in signal.columns
    assert "mi_credit_risk" in signal.columns
    assert "mi_semiconductor_strength" in signal.columns
    assert signal["market_internals_component_count"].iloc[-1] >= 8
    assert signal["market_internals_risk_score"].iloc[-1] > 0.80


def test_market_internals_missing_symbols_are_reported_and_neutral_safe() -> None:
    prices = _market_prices()[["QQQ", "SPY"]]
    params = MarketInternalsParams(use_market_internals=True, signal_window=20)

    signal = build_market_internals_signal(prices, params)
    status = market_internals_availability(prices, max_required_window=200)

    skipped = {item["symbol"]: item["reason"] for item in status["skipped_symbols"]}
    assert skipped["RSP"] == "missing"
    assert skipped["HYG"] == "missing"
    assert signal["market_internals_risk_score"].notna().all()
