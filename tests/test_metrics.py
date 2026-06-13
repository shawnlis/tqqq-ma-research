import pandas as pd
import pytest

from research.metrics import annualized_return, max_drawdown, summarize_performance


def test_max_drawdown() -> None:
    equity = pd.Series([1.0, 1.2, 0.9, 1.3])

    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_annualized_return() -> None:
    equity = pd.Series([100.0, 121.0])

    assert annualized_return(equity, periods_per_year=1) == pytest.approx(0.21)


def _bt(returns: list[float], dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ret": returns,
            "position": 1.0,
            "turnover": 0.0,
        },
        index=dates,
    )


def test_objectives_rank_known_outperformer_above_benchmark() -> None:
    dates = pd.bdate_range("2020-01-01", periods=260)
    benchmark_ret = pd.Series(0.001, index=dates)
    outperformer = _bt([0.002] * len(dates), dates)
    laggard = _bt([0.0005] * len(dates), dates)

    out = summarize_performance(outperformer, benchmark_ret)
    lag = summarize_performance(laggard, benchmark_ret)

    assert out["objective_final_ratio"] > 1.0
    assert out["objective_final_ratio"] > lag["objective_final_ratio"]
    assert out["objective_raw_outperformance_with_survival"] > lag["objective_raw_outperformance_with_survival"]


def test_objectives_penalize_high_sharpe_strategy_that_loses_final_equity() -> None:
    dates = pd.bdate_range("2020-01-01", periods=260)
    benchmark_ret = pd.Series(([0.02, -0.005] * 130), index=dates)
    low_vol_loser = _bt(([0.0006, 0.0004] * 130), dates)

    metrics = summarize_performance(low_vol_loser, benchmark_ret)

    assert metrics["sharpe"] > metrics["bench_sharpe"]
    assert metrics["final_equity_ratio"] < 1.0
    assert metrics["objective_raw_outperformance_with_survival"] < 0.0
    assert metrics["objective_final_ratio"] < 1.0


def test_objectives_use_same_period_benchmark_dates() -> None:
    dates = pd.bdate_range("2020-01-01", periods=5)
    bt = _bt([0.0, 0.01, 0.01, 0.01, 0.01], dates)
    benchmark_ret = pd.Series(
        [0.0, 0.01, 0.01, 0.01, 0.01, 0.99],
        index=pd.bdate_range("2020-01-01", periods=6),
    )

    metrics = summarize_performance(bt, benchmark_ret)

    assert metrics["objective_final_ratio"] == pytest.approx(1.0)

    missing_same_period = benchmark_ret.drop(dates[2])
    with pytest.raises(ValueError, match="benchmark_ret is missing"):
        summarize_performance(bt, missing_same_period)
