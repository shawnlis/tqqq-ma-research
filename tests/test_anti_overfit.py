import pandas as pd
import pytest

from research.anti_overfit import block_bootstrap_summary, subperiod_summary


def test_subperiod_summary_uses_only_stitched_dates_for_benchmark() -> None:
    stitched = pd.DataFrame(
        {
            "ret": [0.0, 0.05],
            "equity": [1.0, 1.05],
        },
        index=pd.to_datetime(["2023-01-03", "2023-01-04"]),
    )
    price_data = pd.DataFrame(
        {"TQQQ": [1.0, 100.0, 110.0, 999999.0]},
        index=pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
    )

    summary = subperiod_summary(stitched, price_data, "TQQQ")
    row = summary.loc[summary["period"] == "2023-present"].iloc[0]

    assert row["status"] == "ok"
    assert row["start_date"] == "2023-01-03"
    assert row["end_date"] == "2023-01-04"
    assert row["benchmark_final_equity"] == pytest.approx(1.10)
    assert row["final_equity_ratio"] == pytest.approx(1.05 / 1.10)


def test_block_bootstrap_runs_paired_same_period_returns() -> None:
    dates = pd.bdate_range("2023-01-03", periods=8)
    stitched = pd.DataFrame(
        {
            "ret": [0.0, 0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.02],
            "equity": [1.0, 1.02, 1.0098, 1.040094, 1.040094, 1.05049494, 1.0294850412, 1.050074742024],
        },
        index=dates,
    )
    price_data = pd.DataFrame(
        {"TQQQ": [100.0, 101.0, 100.0, 104.0, 104.0, 105.0, 103.0, 104.0]},
        index=dates,
    )

    summary = block_bootstrap_summary(
        stitched,
        price_data,
        "TQQQ",
        block_sizes=(2,),
        n_bootstrap=10,
        seed=7,
    )

    assert summary.loc[0, "status"] == "ok"
    assert summary.loc[0, "block_size"] == 2
    assert 0.0 <= summary.loc[0, "probability_strategy_beats_benchmark"] <= 1.0
