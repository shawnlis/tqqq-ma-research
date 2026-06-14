from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


STAGE_SUMMARY_COLUMNS = [
    "family",
    "category",
    "stage_role",
    "best_variant",
    "best_final_equity_ratio",
    "best_robustness_score",
    "cost_10_final_equity_ratio",
    "cost_25_final_equity_ratio",
    "cost_50_final_equity_ratio",
    "walk_forward_5_1_final_equity_ratio",
    "walk_forward_3_1_final_equity_ratio",
    "walk_forward_7_1_final_equity_ratio",
    "baseline_strategy_max_dd",
    "worst_strategy_max_dd",
    "selected_max_exposure_values",
    "max_selected_max_exposure",
    "max_exposure_gt_1_required_to_beat_tqqq",
    "max_exposure_1_beats_tqqq",
    "ratio_versus_same_max_constant_tqqq",
    "max_drawdown_difference_vs_same_max_constant_tqqq",
    "performance_dominated_by_one_year",
    "dominant_year",
    "dominant_year_excess_share",
    "status",
    "rejection_reason",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _first_ok(group: pd.DataFrame, variant: str) -> Optional[pd.Series]:
    if group.empty:
        return None
    variant_values = (
        group["variant"].astype(str)
        if "variant" in group.columns
        else pd.Series("", index=group.index, dtype=str)
    )
    status_values = (
        group["status"].astype(str)
        if "status" in group.columns
        else pd.Series("", index=group.index, dtype=str)
    )
    rows = group[(variant_values == variant) & (status_values == "ok")]
    if rows.empty:
        return None
    return rows.iloc[0]


def _ratio_for(group: pd.DataFrame, wf_variant: str, cost_bps: float) -> float:
    if group.empty:
        return np.nan
    wf_values = (
        group["walk_forward_variant"].astype(str)
        if "walk_forward_variant" in group.columns
        else pd.Series("", index=group.index, dtype=str)
    )
    cost_values = (
        pd.to_numeric(group["transaction_cost_bps"], errors="coerce")
        if "transaction_cost_bps" in group.columns
        else pd.Series(np.nan, index=group.index, dtype=float)
    )
    status_values = (
        group["status"].astype(str)
        if "status" in group.columns
        else pd.Series("", index=group.index, dtype=str)
    )
    rows = group[
        (wf_values == wf_variant)
        & (cost_values == float(cost_bps))
        & (status_values == "ok")
    ]
    ratios = pd.to_numeric(rows.get("final_equity_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
    return float(ratios.iloc[0]) if not ratios.empty else np.nan


def _role(category: str) -> str:
    category = str(category)
    if category == "tqqq_voltarget":
        return "voltarget"
    if category == "tqqq_voltarget_governor":
        return "governor"
    if category == "secondary_comparator":
        return "core_overlay_comparator"
    return category or "unknown"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _selected_max_exposure(output_dir: Path) -> tuple[str, float]:
    wf = _read_csv(output_dir / "walk_forward_windows.csv")
    if wf.empty or "max_exposure" not in wf.columns:
        return "", np.nan
    values = pd.to_numeric(wf["max_exposure"], errors="coerce").dropna()
    if values.empty:
        return "", np.nan
    unique_values = sorted(float(value) for value in values.unique())
    return ";".join(f"{value:g}" for value in unique_values), float(max(unique_values))


def _same_max_fairness(output_dir: Path) -> Dict[str, float]:
    fairness = _read_csv(output_dir / "constant_leverage_benchmark_summary.csv")
    if fairness.empty or "is_same_max_exposure_benchmark" not in fairness.columns:
        return {
            "ratio_versus_same_max_constant_tqqq": np.nan,
            "max_drawdown_difference_vs_same_max_constant_tqqq": np.nan,
        }
    mask = fairness["is_same_max_exposure_benchmark"].map(_truthy)
    row = fairness[mask].iloc[0] if mask.any() else fairness.iloc[0]
    return {
        "ratio_versus_same_max_constant_tqqq": _safe_float(
            row.get("ratio_versus_same_max_constant_tqqq")
        ),
        "max_drawdown_difference_vs_same_max_constant_tqqq": _safe_float(
            row.get("max_drawdown_difference_vs_same_max_constant_tqqq")
        ),
    }


def _baseline_output_path(baseline: Optional[pd.Series]) -> Optional[Path]:
    if baseline is None:
        return None
    value = str(baseline.get("output_dir", "")).strip()
    if not value or value.lower() == "nan":
        return None
    return Path(value)


def _year_dominance(output_dir: Path) -> Dict[str, Any]:
    yearly = _read_csv(output_dir / "yearly_returns.csv")
    if yearly.empty or not {"year", "strategy_return", "benchmark_return"}.issubset(yearly.columns):
        return {
            "performance_dominated_by_one_year": "",
            "dominant_year": "",
            "dominant_year_excess_share": np.nan,
        }
    excess = pd.to_numeric(yearly["strategy_return"], errors="coerce") - pd.to_numeric(
        yearly["benchmark_return"],
        errors="coerce",
    )
    positive = excess[excess > 0.0]
    if positive.empty or positive.sum() <= 0.0:
        return {
            "performance_dominated_by_one_year": False,
            "dominant_year": "",
            "dominant_year_excess_share": 0.0,
        }
    idx = positive.idxmax()
    share = float(positive.loc[idx] / positive.sum())
    return {
        "performance_dominated_by_one_year": bool(share > 0.60),
        "dominant_year": str(yearly.loc[idx, "year"]),
        "dominant_year_excess_share": share,
    }


def _max_exposure_gate(baseline_ratio: float, selected_values: str, selected_max: float) -> tuple[str, str]:
    if not pd.notna(baseline_ratio) or baseline_ratio <= 1.0:
        return "not_applicable_no_raw_outperformance", "false"
    if not selected_values or not pd.notna(selected_max):
        return "unknown_no_selected_params", "unknown"
    selected = [float(value) for value in selected_values.split(";") if value]
    if selected and max(selected) <= 1.0:
        return "no", "true"
    if selected and min(selected) > 1.0:
        return "yes_selected_path_uses_gt_1", "false"
    return "mixed_selected_windows", "unknown"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    table = df[[column for column in columns if column in df.columns]].fillna("")
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in table.columns) + " |")
    return "\n".join(lines)


def summarize_voltarget_stage(output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    summary = _read_csv(output_dir / "tournament_summary.csv")
    variants = _read_csv(output_dir / "tournament_variant_results.csv")
    if summary.empty:
        raise FileNotFoundError(f"Missing tournament_summary.csv in {output_dir}")

    rows: list[Dict[str, Any]] = []
    for _, strategy in summary.iterrows():
        family = str(strategy.get("family", ""))
        category = str(strategy.get("category", ""))
        group = variants[
            (variants.get("family", "").astype(str) == family)
            & (variants.get("category", "").astype(str) == category)
        ].copy() if not variants.empty else pd.DataFrame()
        ok = group[group.get("status", "").astype(str) == "ok"].copy() if not group.empty else pd.DataFrame()
        variant_ratios = (
            pd.to_numeric(ok["final_equity_ratio"], errors="coerce")
            if not ok.empty and "final_equity_ratio" in ok.columns
            else pd.Series(dtype=float)
        )
        valid_variant_ratios = variant_ratios.dropna()
        if not valid_variant_ratios.empty:
            best = ok.loc[valid_variant_ratios.idxmax()]
        else:
            best = pd.Series(dtype=object)

        baseline = _first_ok(group, "standard_5y_1y__10bps") if not group.empty else None
        output_path = _baseline_output_path(baseline)
        selected_values, selected_max = _selected_max_exposure(output_path) if output_path else ("", np.nan)
        fairness = (
            _same_max_fairness(output_path)
            if output_path
            else {
                "ratio_versus_same_max_constant_tqqq": np.nan,
                "max_drawdown_difference_vs_same_max_constant_tqqq": np.nan,
            }
        )
        dominance = (
            _year_dominance(output_path)
            if output_path
            else {
                "performance_dominated_by_one_year": "",
                "dominant_year": "",
                "dominant_year_excess_share": np.nan,
            }
        )
        baseline_ratio = _safe_float(strategy.get("baseline_final_equity_ratio"))
        max_gate, max_1_beats = _max_exposure_gate(baseline_ratio, selected_values, selected_max)

        rows.append(
            {
                "family": family,
                "category": category,
                "stage_role": _role(category),
                "best_variant": str(best.get("variant", strategy.get("baseline_variant", ""))),
                "best_final_equity_ratio": _safe_float(best.get("final_equity_ratio", strategy.get("baseline_final_equity_ratio"))),
                "best_robustness_score": _safe_float(strategy.get("robustness_score")),
                "cost_10_final_equity_ratio": _safe_float(strategy.get("cost_10_final_equity_ratio")),
                "cost_25_final_equity_ratio": _safe_float(strategy.get("cost_25_final_equity_ratio")),
                "cost_50_final_equity_ratio": _safe_float(strategy.get("cost_50_final_equity_ratio")),
                "walk_forward_5_1_final_equity_ratio": _ratio_for(group, "standard_5y_1y", 10.0),
                "walk_forward_3_1_final_equity_ratio": _ratio_for(group, "alternate_3y_1y", 10.0),
                "walk_forward_7_1_final_equity_ratio": _ratio_for(group, "alternate_7y_1y", 10.0),
                "baseline_strategy_max_dd": _safe_float(strategy.get("baseline_strategy_max_dd")),
                "worst_strategy_max_dd": _safe_float(strategy.get("worst_strategy_max_dd")),
                "selected_max_exposure_values": selected_values,
                "max_selected_max_exposure": selected_max,
                "max_exposure_gt_1_required_to_beat_tqqq": max_gate,
                "max_exposure_1_beats_tqqq": max_1_beats,
                **fairness,
                **dominance,
                "status": "ok" if _safe_float(strategy.get("successful_variants")) > 0 else "no_successful_variants",
                "rejection_reason": str(strategy.get("rejection_reason", "")),
            }
        )

    stage = pd.DataFrame(rows, columns=STAGE_SUMMARY_COLUMNS)
    stage.to_csv(output_dir / "voltarget_stage_summary.csv", index=False)
    _write_stage_report(output_dir / "voltarget_stage_report.md", stage)
    print(stage.to_string(index=False))
    print(f"\nVolTarget stage summary written to: {output_dir / 'voltarget_stage_summary.csv'}")
    return stage


def _write_stage_report(path: Path, stage: pd.DataFrame) -> None:
    best = stage.sort_values("best_final_equity_ratio", ascending=False).head(1)
    best_line = "No completed Stage rows were available."
    if not best.empty:
        row = best.iloc[0]
        best_line = (
            f"Best final_equity_ratio: `{row['best_final_equity_ratio']}` "
            f"from `{row['family']}` / `{row['best_variant']}`."
        )
    lines = [
        "# VolTarget Stage Validation Summary",
        "",
        "This report summarizes staged VolTarget tournament outputs only. It is not an investment recommendation.",
        "",
        best_line,
        "",
        "## Family Summary",
        _markdown_table(
            stage,
            [
                "family",
                "stage_role",
                "best_final_equity_ratio",
                "best_robustness_score",
                "baseline_strategy_max_dd",
                "worst_strategy_max_dd",
                "max_exposure_gt_1_required_to_beat_tqqq",
                "max_exposure_1_beats_tqqq",
                "performance_dominated_by_one_year",
            ],
        ),
        "",
        "## 10/25/50 bps Comparison",
        _markdown_table(
            stage,
            [
                "family",
                "cost_10_final_equity_ratio",
                "cost_25_final_equity_ratio",
                "cost_50_final_equity_ratio",
            ],
        ),
        "",
        "## 5/1, 3/1, 7/1 Walk-Forward Comparison",
        _markdown_table(
            stage,
            [
                "family",
                "walk_forward_5_1_final_equity_ratio",
                "walk_forward_3_1_final_equity_ratio",
                "walk_forward_7_1_final_equity_ratio",
            ],
        ),
        "",
        "## Same-Max-Exposure Constant Benchmark",
        _markdown_table(
            stage,
            [
                "family",
                "selected_max_exposure_values",
                "ratio_versus_same_max_constant_tqqq",
                "max_drawdown_difference_vs_same_max_constant_tqqq",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
