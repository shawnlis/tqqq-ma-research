import pandas as pd
import pytest

from research.yearly_contribution import compute_one_year_contribution


def test_one_year_contribution_identifies_dominated_result() -> None:
    yearly = pd.DataFrame(
        [
            {"year": 2020, "strategy_return": 0.50, "benchmark_return": 0.00},
            {"year": 2021, "strategy_return": 0.02, "benchmark_return": 0.00},
            {"year": 2022, "strategy_return": 0.01, "benchmark_return": 0.00},
        ]
    )

    audit = compute_one_year_contribution(yearly, metadata={"family": "VolTarget"})

    assert set(audit["audit_status"]) == {"ok"}
    assert bool(audit["one_year_contributes_more_than_60pct"].iloc[0])
    assert int(audit["largest_contribution_year"].iloc[0]) == 2020
    assert float(audit["largest_single_year_contribution_share"].iloc[0]) > 0.60


def test_one_year_contribution_passes_diversified_result() -> None:
    yearly = pd.DataFrame(
        [
            {"year": 2020, "strategy_return": 0.10, "benchmark_return": 0.00},
            {"year": 2021, "strategy_return": 0.10, "benchmark_return": 0.00},
            {"year": 2022, "strategy_return": 0.10, "benchmark_return": 0.00},
        ]
    )

    audit = compute_one_year_contribution(yearly)

    assert set(audit["audit_status"]) == {"ok"}
    assert not bool(audit["one_year_contributes_more_than_60pct"].iloc[0])
    assert float(audit["largest_single_year_contribution_share"].iloc[0]) == pytest.approx(1.0 / 3.0)
    assert audit["contribution_to_total_log_excess"].sum() == pytest.approx(1.0)


def test_one_year_contribution_handles_negative_total_excess_return() -> None:
    yearly = pd.DataFrame(
        [
            {"year": 2020, "strategy_return": 0.00, "benchmark_return": 0.10},
            {"year": 2021, "strategy_return": 0.02, "benchmark_return": 0.10},
        ]
    )

    audit = compute_one_year_contribution(yearly)

    assert set(audit["audit_status"]) == {"non_positive_total_log_excess"}
    assert not bool(audit["one_year_contributes_more_than_60pct"].iloc[0])
    assert float(audit["largest_single_year_contribution_share"].iloc[0]) == 0.0
    assert float(audit["total_log_excess_return"].iloc[0]) < 0.0
    assert audit["contribution_to_total_log_excess"].notna().all()

