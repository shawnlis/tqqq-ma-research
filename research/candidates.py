from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .baseline_gate import enforce_baseline_gate
from .experiments import load_yaml_file
from .reports import _markdown_table


ALLOWED_REJECTION_REASONS = {
    "failed raw outperformance",
    "overfit",
    "too cost-sensitive",
    "drawdown too severe",
    "rebound miss",
    "insufficient history",
    "data unavailable",
}

OUTPUT_COLUMNS = [
    "family",
    "category",
    "experiment_name",
    "config_path",
    "strategy_name",
    "comparison_benchmark_symbol",
    "standard_variant",
    "standard_final_equity_ratio",
    "worst_alternate_final_equity_ratio",
    "cost_10_final_equity_ratio",
    "cost_50_final_equity_ratio",
    "standard_strategy_max_dd",
    "standard_benchmark_max_dd",
    "drawdown_gap_vs_benchmark",
    "parameter_stability_neighborhood_median_final_equity_ratio",
    "parameter_stability_source",
    "single_year_excess_return_share",
    "dominant_excess_year",
    "yearly_returns_source",
    "accepted_final_candidate",
    "primary_rejection_reason",
    "rejection_details",
]


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _safe_name(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
    return name or "strategy"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _identity(row: pd.Series) -> Dict[str, str]:
    return {
        "family": str(row.get("family", "")),
        "category": str(row.get("category", "")),
        "experiment_name": str(row.get("experiment_name", "")),
        "config_path": str(row.get("config_path", "")),
        "strategy_name": str(row.get("strategy_name", "")),
    }


def _filter_identity(df: pd.DataFrame, info: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for column, value in info.items():
        if column not in out.columns or value == "":
            continue
        matches = out[out[column].astype(str) == value]
        if matches.empty:
            return matches
        out = matches
    return out


def _candidate_dirs(tournament_dir: Path, info: Dict[str, str]) -> List[Path]:
    dirs: List[Path] = []
    for key in ("experiment_name", "family"):
        value = info.get(key, "")
        if value:
            dirs.extend(
                [
                    tournament_dir / "experiments" / _safe_name(value),
                    tournament_dir / _safe_name(value),
                ]
            )

    config_path = info.get("config_path", "")
    if config_path:
        path = Path(config_path)
        if path.exists():
            try:
                config = load_yaml_file(path)
                output_dir = config.get("output_dir")
                if output_dir:
                    dirs.append(Path(output_dir))
            except Exception:
                pass

    seen = set()
    unique_dirs = []
    for directory in dirs:
        resolved = str(directory)
        if resolved not in seen:
            seen.add(resolved)
            unique_dirs.append(directory)
    return unique_dirs


def _load_artifact(
    tournament_dir: Path,
    info: Dict[str, str],
    filenames: Iterable[str],
) -> Tuple[pd.DataFrame, str]:
    for filename in filenames:
        path = tournament_dir / filename
        df = _read_csv(path)
        if not df.empty:
            filtered = _filter_identity(df, info)
            if not filtered.empty:
                return filtered, str(path)

    for directory in _candidate_dirs(tournament_dir, info):
        for filename in filenames:
            path = directory / filename
            df = _read_csv(path)
            if not df.empty:
                filtered = _filter_identity(df, info)
                if not filtered.empty:
                    return filtered, str(path)

    return pd.DataFrame(), ""


def _first_ok(
    variants: pd.DataFrame,
    *,
    walk_forward_variant: str,
    cost_bps: float,
) -> Optional[pd.Series]:
    if variants.empty:
        return None
    cost = pd.to_numeric(variants.get("transaction_cost_bps"), errors="coerce")
    mask = (
        variants.get("status", "").astype(str).eq("ok")
        & variants.get("walk_forward_variant", "").astype(str).eq(walk_forward_variant)
        & np.isclose(cost.astype(float), float(cost_bps), equal_nan=False)
    )
    rows = variants.loc[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def _standard_variant_name(variants: pd.DataFrame) -> str:
    names = variants.get("walk_forward_variant", pd.Series(dtype=str)).astype(str).unique()
    for preferred in ("standard_5y_1y", "standard"):
        if preferred in names:
            return preferred
    for name in names:
        if "standard" in name.lower():
            return str(name)
    return "standard_5y_1y"


def _worst_alternate_ratio(variants: pd.DataFrame, standard_name: str) -> Tuple[float, int]:
    if variants.empty:
        return np.nan, 0
    cost = pd.to_numeric(variants.get("transaction_cost_bps"), errors="coerce")
    ok = variants[
        variants.get("status", "").astype(str).eq("ok")
        & ~variants.get("walk_forward_variant", "").astype(str).eq(standard_name)
        & np.isclose(cost.astype(float), 10.0, equal_nan=False)
    ].copy()
    ratios = pd.to_numeric(ok.get("final_equity_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
    if ratios.empty:
        return np.nan, 0
    return float(ratios.min()), int(len(ratios))


def _parameter_stability_score(tournament_dir: Path, info: Dict[str, str]) -> Tuple[float, str]:
    df, source = _load_artifact(
        tournament_dir,
        info,
        (
            "parameter_stability.csv",
            "candidate_parameter_stability.csv",
            "final_candidate_parameter_stability.csv",
        ),
    )
    if df.empty:
        return np.nan, ""

    if "metric" in df.columns:
        metric_filter = df["metric"].astype(str).str.contains("final_equity", case=False, na=False)
        if metric_filter.any():
            df = df.loc[metric_filter].copy()

    preferred_columns = (
        "parameter_stability_neighborhood_median_final_equity_ratio",
        "neighborhood_median_final_equity_ratio",
        "median_final_equity_ratio",
        "neighborhood_mean",
        "selected_metric",
        "final_equity_ratio",
    )
    for column in preferred_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            return float(values.median()), f"{source}:{column}"
    return np.nan, source


def _year_concentration(tournament_dir: Path, info: Dict[str, str]) -> Tuple[float, Any, str, int]:
    df, source = _load_artifact(
        tournament_dir,
        info,
        (
            "soxl_vs_tqqq_yearly_returns.csv",
            "yearly_returns.csv",
            "same_period_yearly_returns.csv",
            "tournament_yearly_returns.csv",
        ),
    )
    if df.empty:
        return np.nan, "", "", 0
    required = {"strategy_return", "benchmark_return"}
    if not required.issubset(df.columns):
        return np.nan, "", source, int(len(df))

    returns = df.copy()
    returns["excess_return"] = (
        pd.to_numeric(returns["strategy_return"], errors="coerce")
        - pd.to_numeric(returns["benchmark_return"], errors="coerce")
    )
    returns = returns.dropna(subset=["excess_return"])
    if returns.empty:
        return np.nan, "", source, 0

    positive = returns[returns["excess_return"] > 0.0].copy()
    if positive.empty:
        return 1.0, "", source, int(len(returns))
    total_positive = float(positive["excess_return"].sum())
    if total_positive <= 0.0:
        return 1.0, "", source, int(len(returns))
    idx = positive["excess_return"].idxmax()
    share = float(positive.loc[idx, "excess_return"] / total_positive)
    dominant_year = positive.loc[idx, "year"] if "year" in positive.columns else ""
    return share, dominant_year, source, int(len(returns))


def _classify_data_error(group: pd.DataFrame) -> str:
    errors = " ".join(str(value).lower() for value in group.get("error", pd.Series(dtype=str)).dropna())
    if any(token in errors for token in ("missing", "download", "data", "symbol", "no close")):
        return "data unavailable"
    return "insufficient history"


def _evaluate_strategy(
    tournament_dir: Path,
    group: pd.DataFrame,
    summary_row: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    representative = group.iloc[0] if summary_row is None else summary_row
    info = _identity(representative)
    standard_name = _standard_variant_name(group)
    standard_10 = _first_ok(group, walk_forward_variant=standard_name, cost_bps=10.0)
    standard_50 = _first_ok(group, walk_forward_variant=standard_name, cost_bps=50.0)
    worst_alt, alternate_count = _worst_alternate_ratio(group, standard_name)

    stability_median, stability_source = _parameter_stability_score(tournament_dir, info)
    concentration, dominant_year, yearly_source, yearly_count = _year_concentration(tournament_dir, info)

    standard_ratio = _safe_float(standard_10.get("final_equity_ratio")) if standard_10 is not None else np.nan
    cost_50_ratio = _safe_float(standard_50.get("final_equity_ratio")) if standard_50 is not None else np.nan
    strategy_max_dd = _safe_float(standard_10.get("strategy_max_dd")) if standard_10 is not None else np.nan
    benchmark_max_dd = _safe_float(standard_10.get("benchmark_max_dd")) if standard_10 is not None else np.nan
    comparison_benchmark = ""
    if standard_10 is not None:
        comparison_benchmark = str(
            standard_10.get("benchmark_symbol", standard_10.get("tournament_benchmark_symbol", "TQQQ"))
        ).upper()
    drawdown_gap = (
        abs(strategy_max_dd) - abs(benchmark_max_dd)
        if pd.notna(strategy_max_dd) and pd.notna(benchmark_max_dd)
        else np.nan
    )

    failed_checks: List[str] = []
    primary_reason = ""
    if standard_10 is None:
        failed_checks.append("standard 5/1 10 bps row is missing or failed")
        primary_reason = _classify_data_error(group)
    elif comparison_benchmark and comparison_benchmark != "TQQQ":
        failed_checks.append(f"standard comparison benchmark is {comparison_benchmark}, not TQQQ")
        primary_reason = primary_reason or "data unavailable"
    if standard_50 is None:
        failed_checks.append("standard 5/1 50 bps row is missing or failed")
        primary_reason = primary_reason or _classify_data_error(group)
    if alternate_count == 0:
        failed_checks.append("alternate walk-forward 10 bps rows are missing or failed")
        primary_reason = primary_reason or "insufficient history"
    if pd.isna(stability_median):
        failed_checks.append("parameter stability neighborhood artifact is missing or unusable")
        primary_reason = primary_reason or "data unavailable"
    if pd.isna(concentration):
        failed_checks.append("yearly returns artifact is missing or unusable")
        primary_reason = primary_reason or "data unavailable"
    elif yearly_count < 3:
        failed_checks.append("fewer than three yearly return rows available")
        primary_reason = primary_reason or "insufficient history"

    if pd.notna(standard_ratio) and standard_ratio <= 1.0:
        failed_checks.append("standard walk-forward final_equity_ratio <= 1.0")
        primary_reason = primary_reason or "failed raw outperformance"
    if pd.notna(worst_alt) and worst_alt <= 0.95:
        failed_checks.append("worst alternate walk-forward final_equity_ratio <= 0.95")
        primary_reason = primary_reason or "overfit"
    if pd.notna(standard_ratio) and standard_ratio <= 1.0:
        failed_checks.append("10 bps final_equity_ratio <= 1.0")
        primary_reason = primary_reason or "too cost-sensitive"
    if pd.notna(cost_50_ratio) and cost_50_ratio <= 0.90:
        failed_checks.append("50 bps final_equity_ratio <= 0.90")
        primary_reason = primary_reason or "too cost-sensitive"
    if pd.notna(drawdown_gap) and drawdown_gap > 0.10:
        failed_checks.append("strategy max drawdown is worse than benchmark by more than 10 percentage points")
        primary_reason = primary_reason or "drawdown too severe"
    if pd.notna(stability_median) and stability_median < 0.95:
        failed_checks.append("parameter stability neighborhood median final_equity_ratio < 0.95")
        primary_reason = primary_reason or "overfit"
    if pd.notna(concentration) and concentration > 0.60:
        failed_checks.append("single year contributes more than 60% of positive excess return")
        primary_reason = primary_reason or "overfit"

    accepted = len(failed_checks) == 0
    if accepted:
        primary_reason = ""
    elif primary_reason not in ALLOWED_REJECTION_REASONS:
        primary_reason = "data unavailable"

    return {
        **info,
        "comparison_benchmark_symbol": comparison_benchmark,
        "standard_variant": standard_name,
        "standard_final_equity_ratio": standard_ratio,
        "worst_alternate_final_equity_ratio": worst_alt,
        "cost_10_final_equity_ratio": standard_ratio,
        "cost_50_final_equity_ratio": cost_50_ratio,
        "standard_strategy_max_dd": strategy_max_dd,
        "standard_benchmark_max_dd": benchmark_max_dd,
        "drawdown_gap_vs_benchmark": drawdown_gap,
        "parameter_stability_neighborhood_median_final_equity_ratio": stability_median,
        "parameter_stability_source": stability_source,
        "single_year_excess_return_share": concentration,
        "dominant_excess_year": dominant_year,
        "yearly_returns_source": yearly_source,
        "accepted_final_candidate": bool(accepted),
        "primary_rejection_reason": primary_reason,
        "rejection_details": "; ".join(failed_checks),
    }


def _strategy_groups(variant_results: pd.DataFrame) -> Iterable[Tuple[Tuple[Any, ...], pd.DataFrame]]:
    group_cols = [
        column
        for column in ("family", "category", "experiment_name", "config_path")
        if column in variant_results.columns
    ]
    if not group_cols:
        yield ("all",), variant_results
        return
    yield from variant_results.groupby(group_cols, dropna=False)


def extract_final_candidates(
    tournament_dir: Path,
    *,
    accept_baseline_regression: bool = False,
) -> pd.DataFrame:
    enforce_baseline_gate(
        accept_baseline_regression=accept_baseline_regression,
        action="extracting final candidates",
    )
    tournament_dir = Path(tournament_dir)
    summary_path = tournament_dir / "tournament_summary.csv"
    variant_results_path = tournament_dir / "tournament_variant_results.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing completed tournament summary: {summary_path}. "
            "Run a successful full tournament before extracting candidates."
        )
    if not variant_results_path.exists():
        raise FileNotFoundError(f"Missing tournament variant results: {variant_results_path}")

    variant_results = pd.read_csv(variant_results_path)
    summary = _read_csv(summary_path)
    if summary.empty:
        raise ValueError(f"Tournament summary is empty: {summary_path}")
    summary_index: Dict[Tuple[str, str, str, str], pd.Series] = {}
    if not summary.empty:
        for _, row in summary.iterrows():
            key = (
                str(row.get("family", "")),
                str(row.get("category", "")),
                str(row.get("experiment_name", "")),
                str(row.get("config_path", "")),
            )
            summary_index[key] = row

    rows: List[Dict[str, Any]] = []
    for key, group in _strategy_groups(variant_results):
        key_tuple = tuple(str(value) for value in key)
        summary_row = summary_index.get(key_tuple)
        rows.append(_evaluate_strategy(tournament_dir, group, summary_row=summary_row))

    results = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)
    candidates = results[results["accepted_final_candidate"]].copy()
    rejected = results[~results["accepted_final_candidate"]].copy()

    candidates.sort_values(
        ["standard_final_equity_ratio", "worst_alternate_final_equity_ratio"],
        ascending=[False, False],
    ).to_csv(tournament_dir / "final_candidates.csv", index=False)
    rejected.sort_values(["primary_rejection_reason", "family"]).to_csv(
        tournament_dir / "rejected_strategies.csv",
        index=False,
    )
    _write_candidate_report(tournament_dir, candidates, rejected, results)
    (tournament_dir / "candidate_extraction_config.json").write_text(
        json.dumps(
            {
                "command": "extract-candidates",
                "source_dir": str(tournament_dir),
                "rules": {
                    "standard_final_equity_ratio": "> 1.0 at 10 bps",
                    "worst_alternate_final_equity_ratio": "> 0.95 at 10 bps",
                    "cost_10_final_equity_ratio": "> 1.0",
                    "cost_50_final_equity_ratio": "> 0.90",
                    "drawdown_gap_vs_benchmark": "<= 0.10",
                    "parameter_stability_neighborhood_median_final_equity_ratio": ">= 0.95",
                    "single_year_excess_return_share": "<= 0.60",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _print_candidate_summary(candidates, rejected)
    return results


def _write_candidate_report(
    tournament_dir: Path,
    candidates: pd.DataFrame,
    rejected: pd.DataFrame,
    all_results: pd.DataFrame,
) -> Path:
    report_path = tournament_dir / "candidate_report.md"
    reason_counts = (
        rejected["primary_rejection_reason"].value_counts().rename_axis("reason").reset_index(name="count")
        if not rejected.empty and "primary_rejection_reason" in rejected.columns
        else pd.DataFrame(columns=["reason", "count"])
    )
    lines = [
        "# Final Candidate Extraction Report",
        "",
        "This report applies hard post-tournament filters to empirical backtest artifacts. It is not an investment recommendation.",
        "",
        "## Rules",
        "- Standard 5/1 walk-forward final_equity_ratio versus same-period TQQQ must be > 1.0 at 10 bps.",
        "- Worst alternate walk-forward final_equity_ratio must be > 0.95 at 10 bps.",
        "- Standard final_equity_ratio must be > 1.0 at 10 bps and > 0.90 at 50 bps.",
        "- Strategy max drawdown must not be worse than TQQQ by more than 10 percentage points.",
        "- Parameter-stability neighborhood median final_equity_ratio must be at least 0.95.",
        "- No single year may contribute more than 60% of positive excess return.",
        "",
        "## Final Candidates",
        _markdown_table(
            candidates,
            [
                "family",
                "standard_final_equity_ratio",
                "worst_alternate_final_equity_ratio",
                "cost_50_final_equity_ratio",
                "drawdown_gap_vs_benchmark",
                "parameter_stability_neighborhood_median_final_equity_ratio",
                "single_year_excess_return_share",
            ],
            max_rows=50,
        ),
        "",
        "## Rejection Summary",
        _markdown_table(reason_counts, max_rows=20),
        "",
        "## Rejected Strategies",
        _markdown_table(
            rejected,
            [
                "family",
                "primary_rejection_reason",
                "rejection_details",
                "standard_final_equity_ratio",
                "worst_alternate_final_equity_ratio",
                "cost_50_final_equity_ratio",
                "drawdown_gap_vs_benchmark",
            ],
            max_rows=100,
        ),
        "",
        "## Artifact Coverage",
        _markdown_table(
            all_results,
            [
                "family",
                "parameter_stability_source",
                "yearly_returns_source",
                "primary_rejection_reason",
            ],
            max_rows=100,
        ),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _print_candidate_summary(candidates: pd.DataFrame, rejected: pd.DataFrame) -> None:
    print("\n=== Final candidate extraction ===")
    print(f"FINAL CANDIDATES: {len(candidates)}")
    print(f"REJECTED STRATEGIES: {len(rejected)}")
    if not candidates.empty:
        print("TOP FINAL CANDIDATES:")
        for _, row in candidates.head(10).iterrows():
            ratio = _safe_float(row.get("standard_final_equity_ratio"))
            print(f"- {row.get('family')}: standard ratio={ratio:.6f}")
    if not rejected.empty:
        print("REJECTION REASONS:")
        counts = rejected["primary_rejection_reason"].value_counts()
        for reason, count in counts.items():
            print(f"- {reason}: {int(count)}")
