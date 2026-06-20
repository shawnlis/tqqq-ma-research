from __future__ import annotations

import pandas as pd

from research.voltarget_stage3 import (
    _write_stage3_report,
    build_stage3_robust_alpha_audit,
    classify_stage3_candidate,
)


def _base_row(**overrides):
    row = {
        "family": "VolTargetTQQQStrategy",
        "category": "tqqq_voltarget",
        "full_period_ratio_vs_tqqq": 1.4,
        "ex_2022_ratio_vs_tqqq": 1.1,
        "worst_leave_one_year_ratio": 1.05,
        "ratio_vs_same_avg_exposure_constant": 1.1,
        "ratio_vs_simple_voltarget_no_trend": 1.05,
        "ex_2022_ratio_vs_same_avg_exposure_constant": 1.02,
        "ex_2022_ratio_vs_simple_voltarget_no_trend": 1.01,
        "one_year_dominance_share": 0.50,
        "max_drawdown_improvement_vs_tqqq": 0.08,
    }
    row.update(overrides)
    return row


def test_robust_alpha_candidate_requires_ex_2022_fair_benchmark_pass() -> None:
    row = _base_row(ex_2022_ratio_vs_simple_voltarget_no_trend=0.99)

    assert classify_stage3_candidate(row) != "robust_alpha_candidate"


def test_crash_control_candidate_triggers_when_2022_dominance_remains_high() -> None:
    row = _base_row(
        ex_2022_ratio_vs_simple_voltarget_no_trend=0.75,
        one_year_dominance_share=0.91,
        max_drawdown_improvement_vs_tqqq=0.12,
    )

    assert classify_stage3_candidate(row) == "crash_control_candidate"


def test_leverage_risk_budget_candidate_triggers_when_simple_no_trend_explains_result() -> None:
    row = _base_row(
        ex_2022_ratio_vs_simple_voltarget_no_trend=0.90,
        one_year_dominance_share=0.20,
        max_drawdown_improvement_vs_tqqq=0.01,
    )

    assert classify_stage3_candidate(row) == "leverage_risk_budget_candidate"


def test_stage3_report_includes_deep_tournament_run_or_defer_recommendation(tmp_path) -> None:
    row = _base_row(
        classification="crash_control_candidate",
        is_primary_candidate=True,
        run_deep_tournament_recommendation="defer_deep_tournament",
        decision_reason="Headline and TQQQ ex-2022 gates pass, but fair ex-2022 or one-year dominance gates still fail.",
        recommended_parameter_neighborhood="target_ann_vol=0.75",
        post_2022_ratio_vs_tqqq=0.90,
        turnover=10.0,
        cost_drag=0.1,
    )
    row["50bps_ratio_vs_tqqq"] = 1.2
    audit = pd.DataFrame(
        [
            row
        ]
    )
    report_path = tmp_path / "voltarget_stage3_report.md"

    _write_stage3_report(report_path, audit)

    report = report_path.read_text(encoding="utf-8")
    assert "Should full VolTarget deep tournament be run?" in report
    assert "defer the deep tournament" in report


def test_drawdown_governor_cannot_be_primary_if_it_fails_ex_2022_fair_checks() -> None:
    row = _base_row(
        family="VolTarget + DrawdownGovernor comparator",
        category="tqqq_voltarget_governor",
        ex_2022_ratio_vs_same_avg_exposure_constant=0.50,
        ex_2022_ratio_vs_simple_voltarget_no_trend=0.60,
    )

    assert classify_stage3_candidate(row) != "robust_alpha_candidate"


def test_build_stage3_audit_marks_governor_comparator_as_non_primary(monkeypatch) -> None:
    summary = pd.DataFrame(
        [
            {
                "family": "VolTarget + DrawdownGovernor comparator",
                "category": "tqqq_voltarget_governor",
                "experiment_name": "x",
                "baseline_variant": "standard_5y_1y__10bps",
            }
        ]
    )
    variants = pd.DataFrame(
        [
            {
                "family": "VolTarget + DrawdownGovernor comparator",
                "category": "tqqq_voltarget_governor",
                "experiment_name": "x",
                "variant": "standard_5y_1y__10bps",
                "status": "ok",
                "output_dir": "unused",
            }
        ]
    )

    def fake_candidate_rows(_summary, _variants):
        return pd.DataFrame(
            [
                {
                    **_base_row(
                        family="VolTarget + DrawdownGovernor comparator",
                        category="tqqq_voltarget_governor",
                        ex_2022_ratio_vs_same_avg_exposure_constant=0.40,
                        ex_2022_ratio_vs_simple_voltarget_no_trend=0.50,
                    ),
                    "experiment_name": "x",
                    "baseline_variant": "standard_5y_1y__10bps",
                    "baseline_output_dir": "unused",
                    "is_primary_candidate": False,
                    "50bps_ratio_vs_tqqq": 1.1,
                    "worst_removed_year": 2022,
                    "dominant_year": 2022,
                    "turnover": 1.0,
                    "cost_drag": 0.1,
                    "recommended_parameter_neighborhood": "",
                    "final_equity_ratio": 1.4,
                    "final_equity_ratio_ex_2022": 1.1,
                    "min_leave_one_year_out_ratio": 1.05,
                    "max_drawdown_improvement_ex_2022": 0.08,
                    "one_year_dominated": True,
                    "largest_contribution_year": 2022,
                    "largest_single_year_contribution_share": 0.9,
                }
            ]
        )

    def fake_fair_summary(candidates):
        return pd.DataFrame(
            [
                {
                    "family": "VolTarget + DrawdownGovernor comparator",
                    "category": "tqqq_voltarget_governor",
                    "baseline_output_dir": "unused",
                    "ratio_vs_same_avg_exposure_constant_tqqq": 1.1,
                    "ratio_vs_simple_voltarget_no_trend": 1.1,
                    "ex_2022_ratio_vs_same_avg_exposure_constant_tqqq": 0.40,
                    "ex_2022_ratio_vs_simple_voltarget_no_trend": 0.50,
                }
            ]
        )

    monkeypatch.setattr("research.voltarget_stage3._candidate_summary_rows", fake_candidate_rows)
    monkeypatch.setattr("research.voltarget_stage3.build_fair_leverage_summary_for_candidates", fake_fair_summary)

    audit = build_stage3_robust_alpha_audit(summary, variants)

    assert bool(audit.iloc[0]["is_primary_candidate"]) is False
    assert audit.iloc[0]["classification"] == "reject"
