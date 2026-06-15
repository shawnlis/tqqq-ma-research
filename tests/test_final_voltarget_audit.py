from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.final_voltarget_audit import (
    build_exposure_audit,
    build_fair_benchmark_audit,
    build_yearly_contribution_audit,
    classify_final_candidate,
    prepare_audit_curves,
    run_final_voltarget_audit,
)


def _write_fixture(tmp_path: Path, *, include_exposure: bool = True, include_fair: bool = True) -> tuple[Path, Path]:
    input_dir = tmp_path / "stage3"
    candidate_dir = input_dir / "candidate"
    candidate_dir.mkdir(parents=True)

    candidate = {
        "family": "VolTargetTQQQStrategy",
        "category": "tqqq_voltarget",
        "is_primary_candidate": True,
        "classification": "crash_control_candidate",
        "baseline_output_dir": "candidate",
        "full_period_ratio_vs_tqqq": 3.5,
        "ex_2022_ratio_vs_tqqq": 2.5,
        "post_2022_ratio_vs_tqqq": 1.1,
        "ratio_vs_same_avg_exposure_constant": 3.3,
        "ratio_vs_simple_voltarget_no_trend": 1.5,
        "ex_2022_ratio_vs_simple_voltarget_no_trend": 0.7,
        "one_year_dominance_share": 0.54,
        "run_deep_tournament_recommendation": "defer_deep_tournament",
    }
    pd.DataFrame([candidate]).to_csv(input_dir / "voltarget_stage3_summary.csv", index=False)
    pd.DataFrame([candidate]).drop(columns=["baseline_output_dir"]).to_csv(
        input_dir / "candidate_decision_table.csv",
        index=False,
    )
    pd.DataFrame([candidate]).to_csv(input_dir / "robust_alpha_audit.csv", index=False)
    if include_fair:
        pd.DataFrame(
            [
                {
                    "family": "VolTargetTQQQStrategy",
                    "category": "tqqq_voltarget",
                    "ratio_vs_same_avg_exposure_constant": 3.3,
                    "ratio_vs_simple_voltarget_no_trend": 1.5,
                    "ex_2022_ratio_vs_same_avg_exposure_constant": 0.8,
                    "ex_2022_ratio_vs_simple_voltarget_no_trend": 0.7,
                }
            ]
        ).to_csv(input_dir / "fair_ex_2022_audit.csv", index=False)

    stitched = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2021-01-04",
                    "2022-01-03",
                    "2022-01-04",
                    "2023-01-03",
                ]
            ),
            "equity": [1.0, 1.2, 1.1, 1.5, 1.0, 2.0],
            "daily_ret_tqqq": [0.0, 0.1, -0.2, 0.5, -0.4, 0.8],
        }
    )
    if include_exposure:
        stitched["position"] = [0.5, 1.1, 1.3, 1.6, 2.1, 0.8]
    stitched.to_csv(candidate_dir / "stitched_equity.csv", index=False)
    pd.DataFrame(
        [
            {
                "benchmark_symbol": "TQQQ",
                "strategy_final_equity": 2.0,
                "benchmark_final_equity": 1.0,
                "final_equity_ratio": 2.0,
            }
        ]
    ).to_csv(candidate_dir / "same_period_benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {"year": 2020, "strategy_return": 0.20, "benchmark_return": 0.10},
            {"year": 2021, "strategy_return": -0.08333333333333333, "benchmark_return": -0.20},
            {"year": 2022, "strategy_return": -0.3333333333333333, "benchmark_return": -0.10},
            {"year": 2023, "strategy_return": 1.0, "benchmark_return": 0.8},
        ]
    ).to_csv(candidate_dir / "yearly_returns.csv", index=False)
    return input_dir, candidate_dir


def test_relative_equity_equals_strategy_equity_divided_by_benchmark_equity() -> None:
    stitched = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "equity": [1.0, 2.0, 3.0],
            "daily_ret_tqqq": [0.0, 1.0, 0.5],
        }
    )

    curves = prepare_audit_curves(stitched)

    assert curves["relative_equity"].tolist() == pytest.approx(
        (curves["strategy_equity"] / curves["tqqq_equity"]).tolist()
    )


def test_yearly_contribution_sums_correctly(tmp_path: Path) -> None:
    input_dir, candidate_dir = _write_fixture(tmp_path)
    stitched = pd.read_csv(candidate_dir / "stitched_equity.csv")
    curves = prepare_audit_curves(stitched)
    yearly = build_yearly_contribution_audit(curves, pd.read_csv(candidate_dir / "yearly_returns.csv"))

    assert yearly["share_of_total_log_excess_contribution"].sum() == pytest.approx(1.0)


def test_exposure_thresholds_are_computed_correctly(tmp_path: Path) -> None:
    _, candidate_dir = _write_fixture(tmp_path)
    stitched = pd.read_csv(candidate_dir / "stitched_equity.csv")
    curves = prepare_audit_curves(stitched)

    _, summary = build_exposure_audit(stitched, curves)

    assert summary["pct_days_exposure_gt_1_0"] == pytest.approx(4 / 6)
    assert summary["pct_days_exposure_gt_1_25"] == pytest.approx(3 / 6)
    assert summary["pct_days_exposure_gt_1_5"] == pytest.approx(2 / 6)
    assert summary["pct_days_exposure_gt_2_0"] == pytest.approx(1 / 6)


def test_missing_exposure_column_is_handled_gracefully(tmp_path: Path) -> None:
    _, candidate_dir = _write_fixture(tmp_path, include_exposure=False)
    stitched = pd.read_csv(candidate_dir / "stitched_equity.csv")
    curves = prepare_audit_curves(stitched)

    daily, summary = build_exposure_audit(stitched, curves)

    assert bool(summary["exposure_available"]) is False
    assert summary["message"] == "exposure column unavailable"
    assert "exposure" not in daily.columns


def test_fair_benchmark_missing_artifacts_are_reported_not_guessed(tmp_path: Path) -> None:
    input_dir, candidate_dir = _write_fixture(tmp_path, include_fair=False)
    candidate = pd.read_csv(input_dir / "voltarget_stage3_summary.csv").iloc[0]

    fair = build_fair_benchmark_audit(input_dir, candidate_dir, candidate, {"fair_ex_2022": None, "fair_leverage": None})

    assert "missing_artifact" in set(fair["status"])
    missing = fair[fair["benchmark"].eq("simple_voltarget_no_trend") & fair["period"].eq("ex_2022")]
    assert missing.iloc[0]["status"] == "missing_artifact"


def test_final_classification_remains_crash_control_when_no_trend_ex_2022_below_one(tmp_path: Path) -> None:
    input_dir, candidate_dir = _write_fixture(tmp_path)
    candidate = pd.read_csv(input_dir / "voltarget_stage3_summary.csv").iloc[0]
    fair = build_fair_benchmark_audit(
        input_dir,
        candidate_dir,
        candidate,
        {"fair_ex_2022": pd.read_csv(input_dir / "fair_ex_2022_audit.csv"), "fair_leverage": None},
    )

    assert classify_final_candidate(candidate, fair) == "crash_control_candidate"


def test_report_writes_all_required_sections(tmp_path: Path) -> None:
    input_dir, _ = _write_fixture(tmp_path)
    output_dir = tmp_path / "audit"

    result = run_final_voltarget_audit(input_dir, output_dir)

    report = (output_dir / "final_voltarget_audit_report.md").read_text(encoding="utf-8")
    for section in [
        "Direct Answers",
        "Final Classification",
        "Key Metrics",
        "Charts",
        "Not Claims",
    ]:
        assert section in report
    assert result.final_classification == "crash_control_candidate"
    assert (output_dir / "final_voltarget_audit_summary.csv").exists()

