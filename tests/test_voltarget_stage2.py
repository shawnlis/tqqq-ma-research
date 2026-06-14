from pathlib import Path

import pandas as pd

from research.voltarget_stage2 import (
    evaluate_return_slice,
    filter_out_test_windows_overlapping_year,
    write_voltarget_stage2_sensitivity_outputs,
)


def _stitched(dates: list[str], strategy_returns: list[float], benchmark_returns: list[float]) -> pd.DataFrame:
    index = pd.to_datetime(dates)
    df = pd.DataFrame(
        {
            "ret": strategy_returns,
            "daily_ret_tqqq": benchmark_returns,
        },
        index=index,
    )
    df.index.name = "Date"
    df["equity"] = (1.0 + df["ret"]).cumprod()
    return df


def _write_candidate(
    output_dir: Path,
    *,
    strategy_returns: list[float],
    benchmark_returns: list[float],
) -> None:
    output_dir.mkdir(parents=True)
    stitched = _stitched(
        ["2020-06-30", "2021-06-30", "2022-06-30", "2023-06-30"],
        strategy_returns,
        benchmark_returns,
    )
    stitched.to_csv(output_dir / "stitched_equity.csv")
    pd.DataFrame(
        [
            {"test_start": "2020-01-01", "test_end": "2020-12-31"},
            {"test_start": "2021-01-01", "test_end": "2021-12-31"},
            {"test_start": "2022-01-01", "test_end": "2022-12-31"},
            {"test_start": "2023-01-01", "test_end": "2023-12-31"},
        ]
    ).to_csv(output_dir / "walk_forward_windows.csv", index=False)


def test_posthoc_exclude_year_recomputes_strategy_and_benchmark_on_identical_dates() -> None:
    stitched = _stitched(
        ["2021-01-04", "2022-01-03", "2023-01-03"],
        [0.10, 0.50, 0.20],
        [0.00, 0.20, 0.10],
    )

    row = evaluate_return_slice(
        stitched,
        family="VolTarget",
        category="tqqq_voltarget",
        variant="standard_5y_1y__10bps",
        mode="exclude_2022_returns_posthoc",
        mask=stitched.index.year != 2022,
        excluded_year=2022,
    )

    assert row["same_strategy_benchmark_dates"] is True
    assert row["date_count"] == 2
    assert row["start_date"] == "2021-01-04"
    assert row["end_date"] == "2023-01-03"
    assert row["strategy_final_equity"] == 1.10 * 1.20
    assert row["benchmark_final_equity"] == 1.00 * 1.10


def test_exclude_2022_test_windows_removes_2022_test_windows() -> None:
    stitched = _stitched(
        ["2021-06-30", "2022-06-30", "2023-06-30"],
        [0.10, 0.40, 0.15],
        [0.05, -0.20, 0.10],
    )
    windows = pd.DataFrame(
        [
            {"test_start": "2021-01-01", "test_end": "2021-12-31"},
            {"test_start": "2022-01-01", "test_end": "2022-12-31"},
            {"test_start": "2023-01-01", "test_end": "2023-12-31"},
        ]
    )

    filtered = filter_out_test_windows_overlapping_year(stitched, windows, 2022)

    assert 2022 not in set(filtered.index.year)
    assert set(filtered.index.year) == {2021, 2023}


def test_stage2_outputs_and_report_include_2022_sensitivity(tmp_path: Path) -> None:
    plain_dir = tmp_path / "plain"
    governor_dir = tmp_path / "governor"
    comparator_dir = tmp_path / "comparator"
    _write_candidate(
        plain_dir,
        strategy_returns=[0.05, 0.08, 0.80, 0.10],
        benchmark_returns=[0.04, 0.06, -0.40, 0.08],
    )
    _write_candidate(
        governor_dir,
        strategy_returns=[0.06, 0.10, 0.90, 0.14],
        benchmark_returns=[0.04, 0.06, -0.40, 0.08],
    )
    _write_candidate(
        comparator_dir,
        strategy_returns=[0.03, 0.05, 0.20, 0.07],
        benchmark_returns=[0.04, 0.06, -0.40, 0.08],
    )
    summary = pd.DataFrame(
        [
            {
                "family": "VolTargetTQQQStrategy",
                "category": "tqqq_voltarget",
                "experiment_name": "plain",
                "config_path": "configs/vol_target_tqqq.yaml",
                "baseline_variant": "standard_5y_1y__10bps",
            },
            {
                "family": "VolTarget + DrawdownGovernor",
                "category": "tqqq_voltarget_governor",
                "experiment_name": "governor",
                "config_path": "configs/vol_target_governor.yaml",
                "baseline_variant": "standard_5y_1y__10bps",
            },
            {
                "family": "CoreOverlay + Rebound comparator",
                "category": "secondary_comparator",
                "experiment_name": "comparator",
                "config_path": "configs/core_overlay_with_rebound.yaml",
                "baseline_variant": "standard_5y_1y__10bps",
            },
        ]
    )
    variants = pd.DataFrame(
        [
            {
                "family": "VolTargetTQQQStrategy",
                "category": "tqqq_voltarget",
                "experiment_name": "plain",
                "config_path": "configs/vol_target_tqqq.yaml",
                "variant": "standard_5y_1y__10bps",
                "walk_forward_variant": "standard_5y_1y",
                "transaction_cost_bps": 10.0,
                "execution_model": "close_to_close_shifted",
                "status": "ok",
                "output_dir": str(plain_dir),
            },
            {
                "family": "VolTarget + DrawdownGovernor",
                "category": "tqqq_voltarget_governor",
                "experiment_name": "governor",
                "config_path": "configs/vol_target_governor.yaml",
                "variant": "standard_5y_1y__10bps",
                "walk_forward_variant": "standard_5y_1y",
                "transaction_cost_bps": 10.0,
                "execution_model": "close_to_close_shifted",
                "status": "ok",
                "output_dir": str(governor_dir),
            },
            {
                "family": "CoreOverlay + Rebound comparator",
                "category": "secondary_comparator",
                "experiment_name": "comparator",
                "config_path": "configs/core_overlay_with_rebound.yaml",
                "variant": "standard_5y_1y__10bps",
                "walk_forward_variant": "standard_5y_1y",
                "transaction_cost_bps": 10.0,
                "execution_model": "close_to_close_shifted",
                "status": "ok",
                "output_dir": str(comparator_dir),
            },
        ]
    )

    result = write_voltarget_stage2_sensitivity_outputs(
        output_dir=tmp_path,
        summary=summary,
        variants=variants,
    )

    assert (tmp_path / "voltarget_stage2_summary.csv").exists()
    assert (tmp_path / "voltarget_stage2_report.md").exists()
    assert (tmp_path / "leave_one_year_out.csv").exists()
    assert (tmp_path / "subperiod_summary.csv").exists()
    assert (tmp_path / "year_dominance_audit.csv").exists()
    stage2 = result["summary"]
    assert set(stage2["category"]) == {
        "tqqq_voltarget",
        "tqqq_voltarget_governor",
        "secondary_comparator",
    }
    assert stage2["one_year_dominated"].astype(bool).any()
    report = (tmp_path / "voltarget_stage2_report.md").read_text(encoding="utf-8")
    assert "Does VolTarget beat TQQQ excluding 2022?" in report
    assert "Does the governor add value outside 2022?" in report

