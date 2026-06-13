import pandas as pd
import pytest

import research.backtest as backtest
from research.strategies.ma_strategy import Params


def _prices(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"TQQQ": values, "QQQ": values}, index=index)


def test_transaction_cost_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _prices([100.0, 100.0, 100.0, 100.0])
    signal = pd.Series([1.0, 1.0, 0.0, 0.0], index=data.index)
    monkeypatch.setattr(backtest, "build_signal", lambda tqqq, qqq, params: signal)

    result = backtest.backtest_with_positions(
        data=data,
        params=Params(short=1, long=2, ma_type="sma"),
        transaction_cost_bps=100.0,
    )

    assert result["position"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert result["turnover"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert result["ret"].tolist() == pytest.approx([0.0, -0.01, 0.0, -0.01])


def test_position_shift_prevents_lookahead(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _prices([100.0, 110.0, 121.0])
    signal = pd.Series([0.0, 1.0, 1.0], index=data.index)
    monkeypatch.setattr(backtest, "build_signal", lambda tqqq, qqq, params: signal)

    result = backtest.backtest_with_positions(
        data=data,
        params=Params(short=1, long=2, ma_type="sma"),
        transaction_cost_bps=0.0,
    )

    assert result.loc[data.index[1], "daily_ret"] == pytest.approx(0.10)
    assert result.loc[data.index[1], "position"] == 0.0
    assert result.loc[data.index[1], "ret"] == 0.0
    assert result.loc[data.index[2], "position"] == 1.0


def test_next_open_to_close_execution_uses_shifted_position(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _prices([100.0, 105.0, 110.0])
    data["TQQQ_OPEN"] = [100.0, 200.0, 300.0]
    data["TQQQ_CLOSE"] = [100.0, 210.0, 330.0]
    signal = pd.Series([1.0, 1.0, 1.0], index=data.index)
    monkeypatch.setattr(backtest, "build_signal", lambda tqqq, qqq, params: signal)

    result = backtest.backtest_with_positions(
        data=data,
        params=Params(short=1, long=2, ma_type="sma"),
        transaction_cost_bps=0.0,
        execution_model="next_open_to_close",
    )

    assert result.loc[data.index[0], "position"] == 0.0
    assert result.loc[data.index[1], "position"] == 1.0
    assert result.loc[data.index[1], "daily_ret"] == pytest.approx(0.05)
    assert result.loc[data.index[1], "ret"] == pytest.approx(0.05)
    assert result.loc[data.index[2], "daily_ret"] == pytest.approx(0.10)
    assert result.loc[data.index[2], "ret"] == pytest.approx(0.10)
    assert result.loc[data.index[1], "execution_model"] == "next_open_to_close"


def test_next_open_to_next_open_execution_uses_open_to_next_open_return(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _prices([100.0, 105.0, 110.0])
    data["TQQQ_OPEN"] = [100.0, 200.0, 300.0]
    data["TQQQ_CLOSE"] = [100.0, 210.0, 330.0]
    signal = pd.Series([1.0, 1.0, 1.0], index=data.index)
    monkeypatch.setattr(backtest, "build_signal", lambda tqqq, qqq, params: signal)

    result = backtest.backtest_with_positions(
        data=data,
        params=Params(short=1, long=2, ma_type="sma"),
        transaction_cost_bps=0.0,
        execution_model="next_open_to_next_open",
    )

    assert result.loc[data.index[1], "position"] == 1.0
    assert result.loc[data.index[1], "daily_ret"] == pytest.approx(0.50)
    assert result.loc[data.index[1], "ret"] == pytest.approx(0.50)
    assert result.loc[data.index[2], "daily_ret"] == pytest.approx(0.0)
    assert result.loc[data.index[1], "execution_model"] == "next_open_to_next_open"


def test_execution_model_falls_back_without_ohlc_and_records_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _prices([100.0, 110.0, 121.0])
    signal = pd.Series([1.0, 1.0, 1.0], index=data.index)
    monkeypatch.setattr(backtest, "build_signal", lambda tqqq, qqq, params: signal)

    result = backtest.backtest_with_positions(
        data=data,
        params=Params(short=1, long=2, ma_type="sma"),
        transaction_cost_bps=0.0,
        execution_model="next_open_to_close",
    )

    assert result.loc[data.index[1], "execution_model"] == "close_to_close_shifted"
    assert "missing OHLC" in result.loc[data.index[1], "execution_model_warning"]
    assert result.loc[data.index[1], "ret"] == pytest.approx(0.10)


def test_risk_off_asset_gets_unused_capital_and_transaction_costs(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _prices([100.0, 100.0, 100.0])
    data["SHY"] = [100.0, 100.0, 100.0]
    signal = pd.Series([1.0, 0.0, 0.0], index=data.index)
    monkeypatch.setattr(backtest, "build_signal", lambda tqqq, qqq, params: signal)

    result = backtest.backtest_with_positions(
        data=data,
        params=Params(
            short=1,
            long=2,
            ma_type="sma",
            risk_off_symbol="SHY",
            risk_off_weight=1.0,
        ),
        transaction_cost_bps=100.0,
    )

    assert result["tqqq_weight"].tolist() == [0.0, 1.0, 0.0]
    assert result["risk_off_weight"].tolist() == [1.0, 0.0, 1.0]
    assert result["turnover_tqqq"].tolist() == [0.0, 1.0, 1.0]
    assert result["turnover_risk_off"].tolist() == [1.0, 1.0, 1.0]
    assert result["ret"].tolist() == pytest.approx([-0.01, -0.02, -0.02])
