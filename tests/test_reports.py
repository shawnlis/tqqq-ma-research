import pandas as pd
import pytest

from research.reports import compare_to_benchmark


def test_compare_to_benchmark_uses_only_stitched_dates() -> None:
    stitched_dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    stitched = pd.DataFrame(
        {
            "ret": [0.0, 0.05],
            "equity": [1.0, 1.05],
        },
        index=stitched_dates,
    )
    price_data = pd.DataFrame(
        {"TQQQ": [100.0, 200.0, 220.0, 999.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
    )

    summary, yearly = compare_to_benchmark(stitched, price_data)
    row = summary.iloc[0]

    assert row["start_date"] == "2020-01-02"
    assert row["end_date"] == "2020-01-03"
    assert row["benchmark_final_equity"] == pytest.approx(1.10)
    assert row["final_equity_ratio"] == pytest.approx(1.05 / 1.10)
    assert bool(row["raw_outperformance_pass"]) is False
    assert list(yearly.columns) == [
        "year",
        "strategy_return",
        "benchmark_return",
        "strategy_beats_benchmark",
    ]
    assert yearly.loc[0, "year"] == 2020
    assert yearly.loc[0, "benchmark_return"] == pytest.approx(0.10)
    assert yearly.loc[0, "strategy_return"] == pytest.approx(0.05)
