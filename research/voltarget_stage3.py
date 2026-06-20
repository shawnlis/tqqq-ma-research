from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .voltarget_fair_leverage import build_fair_leverage_summary_for_candidates
from .voltarget_stage2 import evaluate_return_slice, yearly_returns_from_stitched
from .yearly_contribution import candidate_contribution_summary, compute_one_year_contribution


PRIMARY_CATEGORY = "tqqq_voltarget"
GOVERNOR_CATEGORY = "tqqq_voltarget_governor"
VOLTARGET_STAGE3_CATEGORIES = {PRIMARY_CATEGORY, GOVERNOR_CATEGORY}

ROBUST_ALPHA_COLUMNS = [
    "family",
    "category",
    "experiment_name",
    "baseline_variant",
    "baseline_output_dir",
    "is_primary_candidate",
    "classification",
    "full_period_ratio_vs_tqqq",
    "ex_2022_ratio_vs_tqqq",
    "post_2022_ratio_vs_tqqq",
    "ratio_vs_same_avg_exposure_constant",
    "ratio_vs_simple_voltarget_no_trend",
    "ex_2022_ratio_vs_same_avg_exposure_constant",
    "ex_2022_ratio_vs_simple_voltarget_no_trend",
    "50bps_ratio_vs_tqqq",
    "worst_leave_one_year_ratio",
    "worst_removed_year",
    "one_year_dominance_share",
    "dominant_year",
    "max_drawdown_improvement_vs_tqqq",
    "turnover",
    "cost_drag",
    "run_deep_tournament_recommendation",
    "recommended_parameter_neighborhood",
    "decision_reason",
]


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _load_stitched(output_dir: Path) -> pd.DataFrame:
    stitched = _read_csv(output_dir / "stitched_equity.csv", index_col=0, parse_dates=True)
    if not stitched.empty:
        stitched.index.name = "Date"
        if "equity" not in stitched.columns and "ret" in stitched.columns:
            stitched["equity"] = (1.0 + stitched["ret"].astype(float)).cumprod()
    return stitched


def _candidate_variant(summary_row: pd.Series, variants: pd.DataFrame) -> Optional[pd.Series]:
    if variants.empty:
        return None
    family = str(summary_row.get("family", ""))
    category = str(summary_row.get("category", ""))
    experiment_name = str(summary_row.get("experiment_name", ""))
    baseline_variant = str(summary_row.get("baseline_variant", "standard_5y_1y__10bps") or "standard_5y_1y__10bps")
    mask = (
        (variants.get("family", pd.Series(dtype=str)).astype(str) == family)
        & (variants.get("category", pd.Series(dtype=str)).astype(str) == category)
        & (variants.get("variant", pd.Series(dtype=str)).astype(str) == baseline_variant)
        & (variants.get("status", pd.Series(dtype=str)).astype(str) == "ok")
    )
    if experiment_name and "experiment_name" in variants.columns:
        mask &= variants["experiment_name"].astype(str) == experiment_name
    matches = variants[mask]
    return matches.iloc[0] if not matches.empty else None


def _variant_ratio(
    variants: pd.DataFrame,
    *,
    family: str,
    category: str,
    variant: str,
) -> float:
    if variants.empty:
        return np.nan
    rows = variants[
        (variants.get("family", pd.Series(dtype=str)).astype(str) == family)
        & (variants.get("category", pd.Series(dtype=str)).astype(str) == category)
        & (variants.get("variant", pd.Series(dtype=str)).astype(str) == variant)
        & (variants.get("status", pd.Series(dtype=str)).astype(str) == "ok")
    ]
    if rows.empty:
        return np.nan
    return _safe_float(rows.iloc[0].get("final_equity_ratio"))


def _leave_one_year_metrics(
    stitched: pd.DataFrame,
    *,
    family: str,
    category: str,
    variant: str,
) -> tuple[float, Any]:
    if stitched.empty:
        return np.nan, ""
    rows = []
    for year in sorted(int(year) for year in pd.Index(stitched.index.year).unique()):
        row = evaluate_return_slice(
            stitched,
            family=family,
            category=category,
            variant=variant,
            mode="exclude_each_year_posthoc",
            mask=stitched.index.year != year,
            excluded_year=year,
        )
        rows.append(row)
    leave = pd.DataFrame(rows)
    if leave.empty:
        return np.nan, ""
    leave["final_equity_ratio"] = pd.to_numeric(leave["final_equity_ratio"], errors="coerce")
    valid = leave.dropna(subset=["final_equity_ratio"])
    if valid.empty:
        return np.nan, ""
    idx = valid["final_equity_ratio"].idxmin()
    return float(valid.loc[idx, "final_equity_ratio"]), valid.loc[idx, "excluded_year"]


def _selected_parameter_neighborhood(output_dir: Path) -> str:
    windows = _read_csv(output_dir / "walk_forward_windows.csv")
    if windows.empty:
        return ""
    columns = [
        "target_ann_vol",
        "realized_vol_window",
        "min_exposure",
        "max_exposure",
        "trend_window",
        "momentum_window",
        "crash_exposure",
    ]
    parts = []
    for column in columns:
        if column not in windows.columns:
            continue
        values = pd.to_numeric(windows[column], errors="coerce").dropna()
        if values.empty:
            continue
        unique = sorted(float(value) for value in values.unique())
        parts.append(f"{column}=" + "/".join(f"{value:g}" for value in unique))
    return "; ".join(parts)


def classify_stage3_candidate(row: Dict[str, Any]) -> str:
    category = str(row.get("category", ""))
    full_ratio = _safe_float(row.get("full_period_ratio_vs_tqqq"))
    ex_2022 = _safe_float(row.get("ex_2022_ratio_vs_tqqq"))
    worst_leave = _safe_float(row.get("worst_leave_one_year_ratio"))
    same_avg = _safe_float(row.get("ratio_vs_same_avg_exposure_constant"))
    simple = _safe_float(row.get("ratio_vs_simple_voltarget_no_trend"))
    ex_same_avg = _safe_float(row.get("ex_2022_ratio_vs_same_avg_exposure_constant"))
    ex_simple = _safe_float(row.get("ex_2022_ratio_vs_simple_voltarget_no_trend"))
    dominance = _safe_float(row.get("one_year_dominance_share"))
    dd_improvement = _safe_float(row.get("max_drawdown_improvement_vs_tqqq"))

    if (
        category == PRIMARY_CATEGORY
        and full_ratio > 1.0
        and ex_2022 > 1.0
        and worst_leave > 1.0
        and same_avg > 1.0
        and simple > 1.0
        and ex_simple > 1.0
        and pd.notna(dominance)
        and dominance <= 0.60
    ):
        return "robust_alpha_candidate"

    fair_ex_2022_fails = (
        (pd.notna(ex_same_avg) and ex_same_avg <= 1.0)
        or (pd.notna(ex_simple) and ex_simple <= 1.0)
    )
    dominance_high = pd.notna(dominance) and dominance > 0.60
    drawdown_help = pd.notna(dd_improvement) and dd_improvement >= 0.05
    if (
        category == PRIMARY_CATEGORY
        and full_ratio > 1.0
        and ex_2022 >= 0.95
        and drawdown_help
        and (fair_ex_2022_fails or dominance_high)
    ):
        return "crash_control_candidate"

    if (
        category == PRIMARY_CATEGORY
        and full_ratio > 1.0
        and same_avg > 1.0
        and pd.notna(ex_simple)
        and ex_simple <= 1.0
    ):
        return "leverage_risk_budget_candidate"

    return "reject"


def stage3_decision_reason(row: Dict[str, Any]) -> str:
    classification = str(row.get("classification", ""))
    if classification == "robust_alpha_candidate":
        return "Plain VolTarget passes TQQQ, fair benchmark, ex-2022, leave-one-year, and dominance gates."
    if classification == "crash_control_candidate":
        return "Headline and TQQQ ex-2022 gates pass, but fair ex-2022 or one-year dominance gates still fail."
    if classification == "leverage_risk_budget_candidate":
        return "The candidate beats TQQQ and same-average exposure, but SimpleVolTargetNoTrend explains ex-2022 behavior."
    if str(row.get("category", "")) == GOVERNOR_CATEGORY:
        return "DrawdownGovernor is comparator-only and fails primary fair/ex-2022 attribution gates."
    return "Required robust-alpha gates are not met."


def _candidate_summary_rows(summary: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for _, candidate in summary.iterrows():
        category = str(candidate.get("category", ""))
        if category not in VOLTARGET_STAGE3_CATEGORIES:
            continue
        variant = _candidate_variant(candidate, variants)
        if variant is None:
            continue
        output_dir = Path(str(variant.get("output_dir", "")))
        stitched = _load_stitched(output_dir)
        family = str(candidate.get("family", ""))
        baseline_variant = str(candidate.get("baseline_variant", "standard_5y_1y__10bps") or "standard_5y_1y__10bps")
        normal = evaluate_return_slice(
            stitched,
            family=family,
            category=category,
            variant=baseline_variant,
            mode="normal_full_period",
        )
        ex_2022 = evaluate_return_slice(
            stitched,
            family=family,
            category=category,
            variant=baseline_variant,
            mode="exclude_2022_returns_posthoc",
            mask=stitched.index.year != 2022 if not stitched.empty else None,
            excluded_year=2022,
        )
        post_2022 = evaluate_return_slice(
            stitched,
            family=family,
            category=category,
            variant=baseline_variant,
            mode="post_2022_only",
            mask=stitched.index.year >= 2023 if not stitched.empty else None,
        )
        worst_leave, worst_year = _leave_one_year_metrics(
            stitched,
            family=family,
            category=category,
            variant=baseline_variant,
        )
        yearly = yearly_returns_from_stitched(stitched)
        contribution = compute_one_year_contribution(
            yearly,
            metadata={
                "family": family,
                "category": category,
                "variant": baseline_variant,
                "output_dir": str(output_dir),
            },
        )
        dominance = candidate_contribution_summary(contribution)
        rows.append(
            {
                "family": family,
                "category": category,
                "experiment_name": str(candidate.get("experiment_name", "")),
                "baseline_variant": baseline_variant,
                "baseline_output_dir": str(output_dir),
                "is_primary_candidate": category == PRIMARY_CATEGORY,
                "final_equity_ratio": _safe_float(normal.get("final_equity_ratio")),
                "final_equity_ratio_ex_2022": _safe_float(ex_2022.get("final_equity_ratio")),
                "min_leave_one_year_out_ratio": worst_leave,
                "max_drawdown_improvement_ex_2022": _safe_float(ex_2022.get("max_drawdown_improvement")),
                "one_year_dominated": bool(dominance.get("performance_dominated_by_one_year", False)),
                "largest_contribution_year": dominance.get("dominant_year", ""),
                "largest_single_year_contribution_share": _safe_float(dominance.get("dominant_year_excess_share")),
                "full_period_ratio_vs_tqqq": _safe_float(normal.get("final_equity_ratio")),
                "ex_2022_ratio_vs_tqqq": _safe_float(ex_2022.get("final_equity_ratio")),
                "post_2022_ratio_vs_tqqq": _safe_float(post_2022.get("final_equity_ratio")),
                "50bps_ratio_vs_tqqq": _variant_ratio(
                    variants,
                    family=family,
                    category=category,
                    variant="standard_5y_1y__50bps",
                ),
                "worst_leave_one_year_ratio": worst_leave,
                "worst_removed_year": worst_year,
                "one_year_dominance_share": _safe_float(dominance.get("dominant_year_excess_share")),
                "dominant_year": dominance.get("dominant_year", ""),
                "max_drawdown_improvement_vs_tqqq": _safe_float(normal.get("max_drawdown_improvement")),
                "turnover": float(stitched["turnover"].sum()) if not stitched.empty and "turnover" in stitched.columns else np.nan,
                "cost_drag": float(stitched["cost"].sum()) if not stitched.empty and "cost" in stitched.columns else np.nan,
                "recommended_parameter_neighborhood": _selected_parameter_neighborhood(output_dir),
            }
        )
    return pd.DataFrame(rows)


def build_stage3_robust_alpha_audit(summary: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    candidates = _candidate_summary_rows(summary, variants)
    if candidates.empty:
        return pd.DataFrame(columns=ROBUST_ALPHA_COLUMNS)
    stale_fair_columns = [
        "ratio_vs_same_avg_exposure_constant",
        "ratio_vs_simple_voltarget_no_trend",
        "ex_2022_ratio_vs_same_avg_exposure_constant",
        "ex_2022_ratio_vs_simple_voltarget_no_trend",
    ]
    candidates = candidates.drop(columns=[column for column in stale_fair_columns if column in candidates.columns])
    fair = build_fair_leverage_summary_for_candidates(candidates)
    fair_columns = [
        "family",
        "category",
        "baseline_output_dir",
        "ratio_vs_same_avg_exposure_constant_tqqq",
        "ratio_vs_simple_voltarget_no_trend",
        "ex_2022_ratio_vs_same_avg_exposure_constant_tqqq",
        "ex_2022_ratio_vs_simple_voltarget_no_trend",
    ]
    merged = candidates.merge(
        fair[[column for column in fair_columns if column in fair.columns]],
        on=["family", "category", "baseline_output_dir"],
        how="left",
    )
    merged = merged.rename(
        columns={
            "ratio_vs_same_avg_exposure_constant_tqqq": "ratio_vs_same_avg_exposure_constant",
            "ratio_vs_simple_voltarget_no_trend": "ratio_vs_simple_voltarget_no_trend",
            "ex_2022_ratio_vs_same_avg_exposure_constant_tqqq": "ex_2022_ratio_vs_same_avg_exposure_constant",
            "ex_2022_ratio_vs_simple_voltarget_no_trend": "ex_2022_ratio_vs_simple_voltarget_no_trend",
        }
    )
    classifications = []
    reasons = []
    recommendations = []
    for _, row in merged.iterrows():
        payload = row.to_dict()
        classification = classify_stage3_candidate(payload)
        payload["classification"] = classification
        classifications.append(classification)
        reasons.append(stage3_decision_reason(payload))
        recommendations.append("run_deep_tournament" if classification == "robust_alpha_candidate" else "defer_deep_tournament")
    merged["classification"] = classifications
    merged["decision_reason"] = reasons
    merged["run_deep_tournament_recommendation"] = recommendations
    return merged.reindex(columns=ROBUST_ALPHA_COLUMNS)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    table = df[[column for column in columns if column in df.columns]].copy()
    table = table.where(pd.notna(table), "")
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in table.columns) + " |")
    return "\n".join(lines)


def _answer(audit: pd.DataFrame, column: str, threshold: float = 1.0) -> str:
    primary = audit[audit["category"].astype(str) == PRIMARY_CATEGORY] if not audit.empty else pd.DataFrame()
    if primary.empty or column not in primary.columns:
        return "inconclusive"
    value = _safe_float(primary.iloc[0].get(column))
    if not pd.notna(value):
        return "inconclusive"
    return "yes" if value > threshold else "no"


def _write_stage3_report(path: Path, audit: pd.DataFrame) -> None:
    primary = audit[audit["category"].astype(str) == PRIMARY_CATEGORY] if not audit.empty else pd.DataFrame()
    primary_row = primary.iloc[0] if not primary.empty else pd.Series(dtype=object)
    primary_classification = str(primary_row.get("classification", "inconclusive"))
    run_deep = primary_classification == "robust_alpha_candidate"
    if run_deep:
        run_text = "yes; expand only the reported primary VolTarget parameter neighborhood."
    else:
        run_text = "no; defer the deep tournament until fair ex-2022 and dominance gates improve."

    lines = [
        "# VolTarget Stage 3 Robust-Alpha Gate",
        "",
        "This report is a validation and attribution gate. It does not change strategy logic and is not an investment recommendation.",
        "",
        "## Direct Answers",
        f"- Is plain VolTarget robust alpha or mainly crash-control? `{primary_classification}`.",
        f"- Does it still beat fair benchmarks ex-2022? same-average: {_answer(audit, 'ex_2022_ratio_vs_same_avg_exposure_constant')}; SimpleVolTargetNoTrend: {_answer(audit, 'ex_2022_ratio_vs_simple_voltarget_no_trend')}.",
        f"- Is 2022 still dominant after narrowing the grid? {'yes' if _safe_float(primary_row.get('one_year_dominance_share')) > 0.60 else 'no'}.",
        f"- Does post-2022 weakness persist? {'yes' if _safe_float(primary_row.get('post_2022_ratio_vs_tqqq')) < 1.0 else 'no'}.",
        f"- Should full VolTarget deep tournament be run? {run_text}",
        f"- If yes, which exact parameter neighborhood should be expanded? `{primary_row.get('recommended_parameter_neighborhood', '')}`.",
        f"- If no, what classification should the strategy receive? `{primary_classification}`.",
        "",
        "## Candidate Decision Table",
        _markdown_table(
            audit,
            [
                "family",
                "is_primary_candidate",
                "classification",
                "run_deep_tournament_recommendation",
                "decision_reason",
            ],
        ),
        "",
        "## Robust Alpha Audit",
        _markdown_table(
            audit,
            [
                "family",
                "full_period_ratio_vs_tqqq",
                "ex_2022_ratio_vs_tqqq",
                "post_2022_ratio_vs_tqqq",
                "50bps_ratio_vs_tqqq",
                "worst_leave_one_year_ratio",
                "one_year_dominance_share",
                "max_drawdown_improvement_vs_tqqq",
                "turnover",
                "cost_drag",
            ],
        ),
        "",
        "## Fair Ex-2022 Audit",
        _markdown_table(
            audit,
            [
                "family",
                "ratio_vs_same_avg_exposure_constant",
                "ratio_vs_simple_voltarget_no_trend",
                "ex_2022_ratio_vs_same_avg_exposure_constant",
                "ex_2022_ratio_vs_simple_voltarget_no_trend",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_voltarget_stage3_outputs(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    variants: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_stage3_robust_alpha_audit(summary, variants)
    fair_columns = [
        "family",
        "category",
        "ratio_vs_same_avg_exposure_constant",
        "ratio_vs_simple_voltarget_no_trend",
        "ex_2022_ratio_vs_same_avg_exposure_constant",
        "ex_2022_ratio_vs_simple_voltarget_no_trend",
    ]
    decision_columns = [
        "family",
        "category",
        "is_primary_candidate",
        "classification",
        "run_deep_tournament_recommendation",
        "recommended_parameter_neighborhood",
        "decision_reason",
    ]
    robust_columns = [
        column
        for column in ROBUST_ALPHA_COLUMNS
        if column not in {"ratio_vs_same_avg_exposure_constant", "ratio_vs_simple_voltarget_no_trend", "ex_2022_ratio_vs_same_avg_exposure_constant", "ex_2022_ratio_vs_simple_voltarget_no_trend"}
    ]
    robust_alpha = audit[[column for column in robust_columns if column in audit.columns]].copy()
    fair_ex_2022 = audit[[column for column in fair_columns if column in audit.columns]].copy()
    decision = audit[[column for column in decision_columns if column in audit.columns]].copy()

    audit.to_csv(output_dir / "voltarget_stage3_summary.csv", index=False)
    robust_alpha.to_csv(output_dir / "robust_alpha_audit.csv", index=False)
    fair_ex_2022.to_csv(output_dir / "fair_ex_2022_audit.csv", index=False)
    decision.to_csv(output_dir / "candidate_decision_table.csv", index=False)
    _write_stage3_report(output_dir / "voltarget_stage3_report.md", audit)
    return {
        "summary": audit,
        "robust_alpha": robust_alpha,
        "fair_ex_2022": fair_ex_2022,
        "decision": decision,
    }
