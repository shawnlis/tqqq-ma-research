from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .metrics import annualized_return, calmar_ratio, max_drawdown, sharpe_ratio
from .yearly_contribution import (
    ONE_YEAR_CONTRIBUTION_COLUMNS,
    candidate_contribution_summary,
    compute_one_year_contribution,
)


STAGE2_SUMMARY_COLUMNS = [
    "family",
    "category",
    "experiment_name",
    "config_path",
    "baseline_variant",
    "baseline_output_dir",
    "final_equity_ratio",
    "final_equity_ratio_ex_2022",
    "min_leave_one_year_out_ratio",
    "worst_removed_year",
    "max_drawdown_improvement_ex_2022",
    "final_equity_ratio_ex_2022_test_windows",
    "final_equity_ratio_pre_2022",
    "final_equity_ratio_post_2022",
    "final_equity_ratio_2022_only",
    "final_equity_ratio_2020_only",
    "final_equity_ratio_2023_only",
    "pass_full_period",
    "pass_ex_2022",
    "pass_leave_one_year",
    "pass_soft_ex_2022",
    "one_year_dominated",
    "largest_contribution_year",
    "largest_single_year_contribution_share",
    "governor_adds_value_outside_2022",
    "status",
    "error",
]

SUBPERIOD_COLUMNS = [
    "family",
    "category",
    "variant",
    "mode",
    "excluded_year",
    "start_date",
    "end_date",
    "date_count",
    "strategy_final_equity",
    "benchmark_final_equity",
    "final_equity_ratio",
    "strategy_cagr",
    "benchmark_cagr",
    "excess_cagr",
    "strategy_sharpe",
    "benchmark_sharpe",
    "strategy_max_dd",
    "benchmark_max_dd",
    "max_drawdown_improvement",
    "strategy_calmar",
    "benchmark_calmar",
    "years_strategy_beats_benchmark",
    "total_years",
    "same_strategy_benchmark_dates",
    "status",
    "error",
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
    return stitched


def _benchmark_return_column(stitched: pd.DataFrame) -> str:
    for column in ("daily_ret_tqqq", "daily_ret_TQQQ", "benchmark_return"):
        if column in stitched.columns:
            return column
    raise ValueError("stitched equity is missing daily TQQQ benchmark returns.")


def _empty_metric_row(
    *,
    family: str,
    category: str,
    variant: str,
    mode: str,
    excluded_year: Any = "",
    status: str = "no_data",
    error: str = "",
) -> Dict[str, Any]:
    row = {column: "" for column in SUBPERIOD_COLUMNS}
    row.update(
        {
            "family": family,
            "category": category,
            "variant": variant,
            "mode": mode,
            "excluded_year": excluded_year,
            "date_count": 0,
            "same_strategy_benchmark_dates": True,
            "status": status,
            "error": error,
        }
    )
    for column in (
        "strategy_final_equity",
        "benchmark_final_equity",
        "final_equity_ratio",
        "strategy_cagr",
        "benchmark_cagr",
        "excess_cagr",
        "strategy_sharpe",
        "benchmark_sharpe",
        "strategy_max_dd",
        "benchmark_max_dd",
        "max_drawdown_improvement",
        "strategy_calmar",
        "benchmark_calmar",
        "years_strategy_beats_benchmark",
        "total_years",
    ):
        row[column] = np.nan
    return row


def evaluate_return_slice(
    stitched: pd.DataFrame,
    *,
    family: str,
    category: str,
    variant: str,
    mode: str,
    mask: Optional[pd.Series] = None,
    excluded_year: Any = "",
) -> Dict[str, Any]:
    if stitched.empty:
        return _empty_metric_row(
            family=family,
            category=category,
            variant=variant,
            mode=mode,
            excluded_year=excluded_year,
            error="stitched equity is empty",
        )
    if "ret" not in stitched.columns:
        return _empty_metric_row(
            family=family,
            category=category,
            variant=variant,
            mode=mode,
            excluded_year=excluded_year,
            status="invalid_data",
            error="stitched equity is missing ret",
        )
    try:
        benchmark_col = _benchmark_return_column(stitched)
    except ValueError as exc:
        return _empty_metric_row(
            family=family,
            category=category,
            variant=variant,
            mode=mode,
            excluded_year=excluded_year,
            status="invalid_data",
            error=str(exc),
        )

    data = stitched.sort_index().copy()
    if mask is not None:
        aligned = pd.Series(mask, index=stitched.index).reindex(data.index).fillna(False).astype(bool)
        data = data.loc[aligned].copy()
    if data.empty:
        return _empty_metric_row(
            family=family,
            category=category,
            variant=variant,
            mode=mode,
            excluded_year=excluded_year,
            error="period contains no rows",
        )

    strategy_ret = data["ret"].astype(float)
    benchmark_ret = data[benchmark_col].astype(float)
    strategy_equity = (1.0 + strategy_ret).cumprod()
    benchmark_equity = (1.0 + benchmark_ret).cumprod()
    strategy_final = float(strategy_equity.iloc[-1])
    benchmark_final = float(benchmark_equity.iloc[-1])
    strategy_cagr = annualized_return(strategy_equity)
    benchmark_cagr = annualized_return(benchmark_equity)
    strategy_dd = max_drawdown(strategy_equity)
    benchmark_dd = max_drawdown(benchmark_equity)
    yearly = pd.DataFrame(
        {
            "strategy_return": strategy_ret,
            "benchmark_return": benchmark_ret,
        },
        index=data.index,
    ).groupby(data.index.year).agg(lambda returns: float((1.0 + returns).prod() - 1.0))
    years_strategy_beats = int((yearly["strategy_return"] > yearly["benchmark_return"]).sum())
    return {
        "family": family,
        "category": category,
        "variant": variant,
        "mode": mode,
        "excluded_year": excluded_year,
        "start_date": data.index[0].date().isoformat(),
        "end_date": data.index[-1].date().isoformat(),
        "date_count": int(len(data)),
        "strategy_final_equity": strategy_final,
        "benchmark_final_equity": benchmark_final,
        "final_equity_ratio": strategy_final / benchmark_final if benchmark_final != 0.0 else np.nan,
        "strategy_cagr": strategy_cagr,
        "benchmark_cagr": benchmark_cagr,
        "excess_cagr": strategy_cagr - benchmark_cagr
        if pd.notna(strategy_cagr) and pd.notna(benchmark_cagr)
        else np.nan,
        "strategy_sharpe": sharpe_ratio(strategy_ret),
        "benchmark_sharpe": sharpe_ratio(benchmark_ret),
        "strategy_max_dd": strategy_dd,
        "benchmark_max_dd": benchmark_dd,
        "max_drawdown_improvement": strategy_dd - benchmark_dd,
        "strategy_calmar": calmar_ratio(strategy_cagr, strategy_dd),
        "benchmark_calmar": calmar_ratio(benchmark_cagr, benchmark_dd),
        "years_strategy_beats_benchmark": years_strategy_beats,
        "total_years": int(len(yearly)),
        "same_strategy_benchmark_dates": True,
        "status": "ok",
        "error": "",
    }


def filter_out_test_windows_overlapping_year(
    stitched: pd.DataFrame,
    windows: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    if stitched.empty or windows.empty or not {"test_start", "test_end"}.issubset(windows.columns):
        return stitched.iloc[0:0].copy()
    target_start = pd.Timestamp(year=int(year), month=1, day=1)
    target_end = pd.Timestamp(year=int(year), month=12, day=31)
    keep = pd.Series(False, index=stitched.index)
    for _, window in windows.iterrows():
        test_start = pd.Timestamp(window["test_start"])
        test_end = pd.Timestamp(window["test_end"])
        overlaps = test_start <= target_end and test_end >= target_start
        if overlaps:
            continue
        keep |= (stitched.index >= test_start) & (stitched.index <= test_end)
    return stitched.loc[keep].copy()


def yearly_returns_from_stitched(stitched: pd.DataFrame) -> pd.DataFrame:
    if stitched.empty:
        return pd.DataFrame(columns=["year", "strategy_return", "benchmark_return"])
    benchmark_col = _benchmark_return_column(stitched)
    data = stitched.sort_index().copy()
    yearly = pd.DataFrame(
        {
            "strategy_return": data["ret"].astype(float),
            "benchmark_return": data[benchmark_col].astype(float),
        },
        index=data.index,
    ).groupby(data.index.year).agg(lambda returns: float((1.0 + returns).prod() - 1.0))
    yearly.index.name = "year"
    return yearly.reset_index()


def _candidate_variant(summary_row: pd.Series, variants: pd.DataFrame) -> Optional[pd.Series]:
    if variants.empty:
        return None
    family = str(summary_row.get("family", ""))
    category = str(summary_row.get("category", ""))
    experiment_name = str(summary_row.get("experiment_name", ""))
    variant_name = str(summary_row.get("baseline_variant", "standard_5y_1y__10bps") or "standard_5y_1y__10bps")
    mask = (
        (variants.get("family", pd.Series(dtype=str)).astype(str) == family)
        & (variants.get("category", pd.Series(dtype=str)).astype(str) == category)
        & (variants.get("variant", pd.Series(dtype=str)).astype(str) == variant_name)
        & (variants.get("status", pd.Series(dtype=str)).astype(str) == "ok")
    )
    if experiment_name and "experiment_name" in variants.columns:
        mask &= variants["experiment_name"].astype(str) == experiment_name
    matches = variants[mask]
    return matches.iloc[0] if not matches.empty else None


def _mode_rows_for_candidate(
    *,
    family: str,
    category: str,
    variant: str,
    stitched: pd.DataFrame,
    windows: pd.DataFrame,
) -> list[Dict[str, Any]]:
    years = sorted(int(year) for year in pd.Index(stitched.index.year).unique())
    rows = [
        evaluate_return_slice(
            stitched,
            family=family,
            category=category,
            variant=variant,
            mode="normal_full_period",
        ),
        evaluate_return_slice(
            stitched,
            family=family,
            category=category,
            variant=variant,
            mode="exclude_2022_returns_posthoc",
            mask=stitched.index.year != 2022,
            excluded_year=2022,
        ),
    ]
    filtered_windows = filter_out_test_windows_overlapping_year(stitched, windows, 2022)
    rows.append(
        evaluate_return_slice(
            filtered_windows,
            family=family,
            category=category,
            variant=variant,
            mode="exclude_2022_test_windows",
            excluded_year=2022,
        )
    )
    for mode, mask in [
        ("pre_2022_only", stitched.index.year <= 2021),
        ("post_2022_only", stitched.index.year >= 2023),
        ("2022_only", stitched.index.year == 2022),
        ("2020_only", stitched.index.year == 2020),
        ("2023_only", stitched.index.year == 2023),
    ]:
        rows.append(
            evaluate_return_slice(
                stitched,
                family=family,
                category=category,
                variant=variant,
                mode=mode,
                mask=mask,
            )
        )
    for year in years:
        rows.append(
            evaluate_return_slice(
                stitched,
                family=family,
                category=category,
                variant=variant,
                mode="exclude_each_year_posthoc",
                mask=stitched.index.year != year,
                excluded_year=year,
            )
        )
    return rows


def _metric_for(rows: pd.DataFrame, mode: str, column: str) -> float:
    if rows.empty:
        return np.nan
    match = rows[(rows["mode"].astype(str) == mode) & (rows["status"].astype(str) == "ok")]
    if match.empty:
        return np.nan
    return _safe_float(match.iloc[0].get(column))


def _summary_from_candidate(
    *,
    summary_row: pd.Series,
    variant: Optional[pd.Series],
    subperiod_rows: pd.DataFrame,
    contribution: pd.DataFrame,
) -> Dict[str, Any]:
    family = str(summary_row.get("family", ""))
    category = str(summary_row.get("category", ""))
    baseline_variant = str(summary_row.get("baseline_variant", "standard_5y_1y__10bps") or "standard_5y_1y__10bps")
    if variant is None:
        return {
            "family": family,
            "category": category,
            "experiment_name": str(summary_row.get("experiment_name", "")),
            "config_path": str(summary_row.get("config_path", "")),
            "baseline_variant": baseline_variant,
            "baseline_output_dir": "",
            "status": "missing_baseline_variant",
            "error": f"missing ok variant {baseline_variant}",
        }

    leave = subperiod_rows[
        (subperiod_rows["mode"].astype(str) == "exclude_each_year_posthoc")
        & (subperiod_rows["status"].astype(str) == "ok")
    ].copy()
    leave["final_equity_ratio"] = pd.to_numeric(leave["final_equity_ratio"], errors="coerce")
    if leave.empty or leave["final_equity_ratio"].dropna().empty:
        min_leave = np.nan
        worst_removed_year: Any = ""
    else:
        idx = leave["final_equity_ratio"].idxmin()
        min_leave = float(leave.loc[idx, "final_equity_ratio"])
        worst_removed_year = int(float(leave.loc[idx, "excluded_year"]))

    final_ratio = _metric_for(subperiod_rows, "normal_full_period", "final_equity_ratio")
    ex_2022 = _metric_for(subperiod_rows, "exclude_2022_returns_posthoc", "final_equity_ratio")
    ex_2022_dd_improvement = _metric_for(subperiod_rows, "exclude_2022_returns_posthoc", "max_drawdown_improvement")
    dominance = candidate_contribution_summary(contribution)
    row = {
        "family": family,
        "category": category,
        "experiment_name": str(summary_row.get("experiment_name", "")),
        "config_path": str(summary_row.get("config_path", "")),
        "baseline_variant": baseline_variant,
        "baseline_output_dir": str(variant.get("output_dir", "")),
        "final_equity_ratio": final_ratio,
        "final_equity_ratio_ex_2022": ex_2022,
        "min_leave_one_year_out_ratio": min_leave,
        "worst_removed_year": worst_removed_year,
        "max_drawdown_improvement_ex_2022": ex_2022_dd_improvement,
        "final_equity_ratio_ex_2022_test_windows": _metric_for(
            subperiod_rows, "exclude_2022_test_windows", "final_equity_ratio"
        ),
        "final_equity_ratio_pre_2022": _metric_for(subperiod_rows, "pre_2022_only", "final_equity_ratio"),
        "final_equity_ratio_post_2022": _metric_for(subperiod_rows, "post_2022_only", "final_equity_ratio"),
        "final_equity_ratio_2022_only": _metric_for(subperiod_rows, "2022_only", "final_equity_ratio"),
        "final_equity_ratio_2020_only": _metric_for(subperiod_rows, "2020_only", "final_equity_ratio"),
        "final_equity_ratio_2023_only": _metric_for(subperiod_rows, "2023_only", "final_equity_ratio"),
        "pass_full_period": bool(pd.notna(final_ratio) and final_ratio > 1.0),
        "pass_ex_2022": bool(pd.notna(ex_2022) and ex_2022 > 1.0),
        "pass_leave_one_year": bool(pd.notna(min_leave) and min_leave > 1.0),
        "pass_soft_ex_2022": bool(
            pd.notna(ex_2022)
            and ex_2022 > 0.95
            and pd.notna(ex_2022_dd_improvement)
            and ex_2022_dd_improvement > 0.20
        ),
        "one_year_dominated": bool(dominance.get("performance_dominated_by_one_year", False)),
        "largest_contribution_year": dominance.get("dominant_year", ""),
        "largest_single_year_contribution_share": dominance.get("dominant_year_excess_share", np.nan),
        "governor_adds_value_outside_2022": "",
        "status": "ok",
        "error": "",
    }
    return row


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    table = df[[column for column in columns if column in df.columns]].head(max_rows).astype(object)
    table = table.where(pd.notna(table), "")
    if table.empty:
        return "_No requested columns available._"
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in table.columns) + " |")
    return "\n".join(lines)


def _answer(summary: pd.DataFrame, category: str, column: str, threshold: float = 1.0) -> str:
    rows = summary[(summary["category"].astype(str) == category) & (summary["status"].astype(str) == "ok")]
    if rows.empty or column not in rows.columns:
        return "inconclusive"
    value = _safe_float(rows.iloc[0].get(column))
    if not pd.notna(value):
        return "inconclusive"
    return "yes" if value > threshold else "no"


def _write_stage2_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    leave_one_year: pd.DataFrame,
    subperiod: pd.DataFrame,
    dominance: pd.DataFrame,
) -> None:
    governor = summary[summary["category"].astype(str) == "tqqq_voltarget_governor"]
    plain = summary[summary["category"].astype(str) == "tqqq_voltarget"]
    if not governor.empty and not plain.empty:
        governor_ex = _safe_float(governor.iloc[0].get("final_equity_ratio_ex_2022"))
        plain_ex = _safe_float(plain.iloc[0].get("final_equity_ratio_ex_2022"))
        governor_adds = bool(pd.notna(governor_ex) and pd.notna(plain_ex) and governor_ex > plain_ex)
    else:
        governor_adds = False

    destroyed = summary[summary["min_leave_one_year_out_ratio"].apply(_safe_float) <= 1.0]
    destroyed_text = (
        "No single-year removal destroyed every candidate result."
        if destroyed.empty
        else "; ".join(
            f"{row['family']}: removing {row['worst_removed_year']} left ratio {float(row['min_leave_one_year_out_ratio']):.6f}"
            for _, row in destroyed.iterrows()
            if pd.notna(_safe_float(row.get("min_leave_one_year_out_ratio")))
        )
    )

    lines = [
        "# VolTarget Stage 2 2022 Sensitivity Report",
        "",
        "This report is a validation/audit artifact. It does not change strategy logic and is not an investment recommendation.",
        "",
        "## Direct Answers",
        f"- Does VolTarget beat TQQQ excluding 2022? {_answer(summary, 'tqqq_voltarget', 'final_equity_ratio_ex_2022')}.",
        f"- Does VolTarget + Governor beat TQQQ excluding 2022? {_answer(summary, 'tqqq_voltarget_governor', 'final_equity_ratio_ex_2022')}.",
        f"- Does VolTarget beat TQQQ if every single year is removed one at a time? {_answer(summary, 'tqqq_voltarget', 'min_leave_one_year_out_ratio')}.",
        f"- Does VolTarget + Governor beat TQQQ if every single year is removed one at a time? {_answer(summary, 'tqqq_voltarget_governor', 'min_leave_one_year_out_ratio')}.",
        f"- Which year removal destroys the result? {destroyed_text}",
        f"- Does the strategy still beat TQQQ in 2016-2021? {_answer(summary, 'tqqq_voltarget', 'final_equity_ratio_pre_2022')}.",
        f"- Does the strategy still beat TQQQ in 2023-2026? {_answer(summary, 'tqqq_voltarget', 'final_equity_ratio_post_2022')}.",
        f"- Does the governor add value outside 2022? {'yes' if governor_adds else 'no'}.",
        f"- Is the strategy still one-year dominated after Stage 2? {'yes' if bool(summary['one_year_dominated'].map(_truthy).any()) else 'no'}.",
        "",
        "## Stage 2 Summary",
        _markdown_table(
            summary,
            [
                "family",
                "category",
                "final_equity_ratio",
                "final_equity_ratio_ex_2022",
                "min_leave_one_year_out_ratio",
                "worst_removed_year",
                "pass_full_period",
                "pass_ex_2022",
                "pass_leave_one_year",
                "pass_soft_ex_2022",
                "one_year_dominated",
                "largest_contribution_year",
                "largest_single_year_contribution_share",
                "governor_adds_value_outside_2022",
            ],
        ),
        "",
        "## Leave-One-Year-Out",
        _markdown_table(
            leave_one_year.sort_values(["family", "final_equity_ratio"]) if not leave_one_year.empty else leave_one_year,
            ["family", "excluded_year", "final_equity_ratio", "strategy_max_dd", "benchmark_max_dd", "status"],
            max_rows=40,
        ),
        "",
        "## Subperiods",
        _markdown_table(
            subperiod,
            ["family", "mode", "final_equity_ratio", "max_drawdown_improvement", "years_strategy_beats_benchmark", "total_years", "status"],
            max_rows=50,
        ),
        "",
        "## Year Dominance Audit",
        _markdown_table(
            dominance,
            [
                "family",
                "year",
                "yearly_strategy_return",
                "yearly_tqqq_return",
                "contribution_to_total_log_excess",
                "largest_single_year_contribution_share",
                "one_year_contributes_more_than_60pct",
                "audit_status",
            ],
            max_rows=50,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_voltarget_stage2_sensitivity_outputs(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    variants: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[Dict[str, Any]] = []
    subperiod_rows: list[Dict[str, Any]] = []
    dominance_rows: list[Dict[str, Any]] = []

    for _, candidate in summary.iterrows():
        variant = _candidate_variant(candidate, variants)
        if variant is None:
            summary_rows.append(
                _summary_from_candidate(
                    summary_row=candidate,
                    variant=None,
                    subperiod_rows=pd.DataFrame(columns=SUBPERIOD_COLUMNS),
                    contribution=pd.DataFrame(columns=ONE_YEAR_CONTRIBUTION_COLUMNS),
                )
            )
            continue

        family = str(candidate.get("family", ""))
        category = str(candidate.get("category", ""))
        variant_name = str(variant.get("variant", "standard_5y_1y__10bps"))
        candidate_output_dir = Path(str(variant.get("output_dir", "")))
        stitched = _load_stitched(candidate_output_dir)
        windows = _read_csv(candidate_output_dir / "walk_forward_windows.csv")
        candidate_subperiod = pd.DataFrame(
            _mode_rows_for_candidate(
                family=family,
                category=category,
                variant=variant_name,
                stitched=stitched,
                windows=windows,
            ),
            columns=SUBPERIOD_COLUMNS,
        )
        yearly = yearly_returns_from_stitched(stitched) if not stitched.empty else pd.DataFrame()
        contribution = compute_one_year_contribution(
            yearly,
            metadata={
                "family": family,
                "category": category,
                "experiment_name": str(candidate.get("experiment_name", "")),
                "config_path": str(candidate.get("config_path", "")),
                "variant": variant_name,
                "walk_forward_variant": str(variant.get("walk_forward_variant", "")),
                "transaction_cost_bps": variant.get("transaction_cost_bps", ""),
                "execution_model": str(variant.get("execution_model", "")),
                "output_dir": str(candidate_output_dir),
            },
        )
        subperiod_rows.extend(candidate_subperiod.to_dict("records"))
        dominance_rows.extend(contribution.to_dict("records"))
        summary_rows.append(
            _summary_from_candidate(
                summary_row=candidate,
                variant=variant,
                subperiod_rows=candidate_subperiod,
                contribution=contribution,
            )
        )

    stage2_summary = pd.DataFrame(summary_rows, columns=STAGE2_SUMMARY_COLUMNS)
    subperiod = pd.DataFrame(subperiod_rows, columns=SUBPERIOD_COLUMNS)
    leave_one_year = subperiod[subperiod["mode"].astype(str) == "exclude_each_year_posthoc"].copy()
    dominance = pd.DataFrame(dominance_rows, columns=ONE_YEAR_CONTRIBUTION_COLUMNS)

    governor = stage2_summary[stage2_summary["category"].astype(str) == "tqqq_voltarget_governor"]
    plain = stage2_summary[stage2_summary["category"].astype(str) == "tqqq_voltarget"]
    if not governor.empty and not plain.empty:
        governor_ex = _safe_float(governor.iloc[0].get("final_equity_ratio_ex_2022"))
        plain_ex = _safe_float(plain.iloc[0].get("final_equity_ratio_ex_2022"))
        value = bool(pd.notna(governor_ex) and pd.notna(plain_ex) and governor_ex > plain_ex)
        stage2_summary.loc[
            stage2_summary["category"].astype(str) == "tqqq_voltarget_governor",
            "governor_adds_value_outside_2022",
        ] = value

    stage2_summary.to_csv(output_dir / "voltarget_stage2_summary.csv", index=False)
    leave_one_year.to_csv(output_dir / "leave_one_year_out.csv", index=False)
    subperiod.to_csv(output_dir / "subperiod_summary.csv", index=False)
    dominance.to_csv(output_dir / "year_dominance_audit.csv", index=False)
    _write_stage2_report(
        output_dir / "voltarget_stage2_report.md",
        summary=stage2_summary,
        leave_one_year=leave_one_year,
        subperiod=subperiod,
        dominance=dominance,
    )
    return {
        "summary": stage2_summary,
        "leave_one_year": leave_one_year,
        "subperiod": subperiod,
        "dominance": dominance,
    }
