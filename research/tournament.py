from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .baseline_gate import enforce_baseline_gate
from .execution import CLOSE_TO_CLOSE_SHIFTED
from .experiments import (
    _asset_config,
    _is_vol_target_strategy,
    _load_price_data,
    _run_walk_forward,
    estimate_experiment_workload,
    load_yaml_file,
)
from .fairness import write_voltarget_fairness_outputs
from .reports import compare_to_benchmark, generate_tournament_report
from .yearly_contribution import (
    build_one_year_contribution_from_variants,
    markdown_one_year_contribution_table,
)
from .voltarget_stage2 import write_voltarget_stage2_sensitivity_outputs
from .voltarget_stage3 import write_voltarget_stage3_outputs


DEFAULT_WALK_FORWARD_VARIANTS = (
    {"name": "standard_5y_1y", "train_years": 5, "test_years": 1},
    {"name": "alternate_3y_1y", "train_years": 3, "test_years": 1},
    {"name": "alternate_7y_1y", "train_years": 7, "test_years": 1},
)
DEFAULT_COST_SCENARIOS = (10.0, 25.0, 50.0)
DEFAULT_EXECUTION_MODELS = (CLOSE_TO_CLOSE_SHIFTED,)
DEFAULT_MAX_ESTIMATED_RUNTIME_SECONDS = 3600.0
SUMMARY_COLUMNS = [
    "family",
    "category",
    "experiment_name",
    "config_path",
    "strategy_name",
    "selection_benchmark_symbol",
    "tournament_benchmark_symbol",
    "baseline_variant",
    "baseline_final_equity_ratio",
    "baseline_excess_cagr",
    "baseline_strategy_cagr",
    "baseline_strategy_max_dd",
    "baseline_strategy_calmar",
    "native_benchmark_symbol",
    "baseline_native_final_equity_ratio",
    "median_native_final_equity_ratio",
    "worst_native_final_equity_ratio",
    "median_final_equity_ratio",
    "worst_final_equity_ratio",
    "validation_variants_beating_tqqq",
    "successful_variants",
    "failed_variants",
    "total_variants",
    "variant_win_rate",
    "worst_strategy_max_dd",
    "cost_10_final_equity_ratio",
    "cost_25_final_equity_ratio",
    "cost_50_final_equity_ratio",
    "cost_sensitivity_25_vs_10",
    "cost_sensitivity_50_vs_10",
    "robustness_score",
    "accepted_candidate",
    "rejection_reason",
    "output_dir",
    "estimated_parameter_combinations",
    "estimated_walk_forward_windows",
    "estimated_evaluations",
    "estimated_runtime_seconds",
    "actual_runtime_seconds",
    "actual_seconds_per_estimated_evaluation",
]


def _resolve_config_path(tournament_path: Path, value: Any) -> Path:
    candidate = Path(str(value))
    if candidate.exists():
        return candidate
    return tournament_path.parent / candidate


def _strategy_entries(tournament_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = tournament_config.get("strategies") or tournament_config.get("configs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Tournament config must contain a non-empty strategies or configs list.")
    normalized = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"config": entry})
        elif isinstance(entry, dict):
            normalized.append(dict(entry))
        else:
            raise ValueError("Tournament strategy entries must be paths or mappings.")
    return normalized


def _walk_forward_variants(tournament_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = tournament_config.get("walk_forward_variants", DEFAULT_WALK_FORWARD_VARIANTS)
    variants = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("walk_forward_variants entries must be mappings.")
        train_years = int(item["train_years"])
        test_years = int(item.get("test_years", 1))
        name = str(item.get("name", f"{train_years}y_{test_years}y"))
        variants.append(
            {
                "name": name,
                "train_years": train_years,
                "test_years": test_years,
                "walk_forward_mode": str(item.get("walk_forward_mode", "rolling")),
            }
        )
    return variants


def _cost_scenarios(tournament_config: Dict[str, Any]) -> List[float]:
    return [float(value) for value in tournament_config.get("cost_scenarios", DEFAULT_COST_SCENARIOS)]


def _execution_models(tournament_config: Dict[str, Any]) -> List[str]:
    values = tournament_config.get("execution_models", DEFAULT_EXECUTION_MODELS)
    if isinstance(values, str):
        values = [values]
    models = [str(value) for value in values]
    return models or list(DEFAULT_EXECUTION_MODELS)


def _variant_name(
    wf_variant: Dict[str, Any],
    cost_bps: float,
    execution_model: str = CLOSE_TO_CLOSE_SHIFTED,
) -> str:
    name = f"{wf_variant['name']}__{float(cost_bps):g}bps"
    if str(execution_model) != CLOSE_TO_CLOSE_SHIFTED:
        name = f"{name}__{_safe_path_part(execution_model)}"
    return name


def _safe_path_part(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text.strip("_") or "variant"


def _variant_output_dir(base_config: Dict[str, Any], variant_name: str) -> Path:
    return Path(str(base_config.get("output_dir", "outputs/tournament/experiments/strategy"))) / _safe_path_part(variant_name)


def _write_variant_report(
    *,
    output_dir: Path,
    row: Dict[str, Any],
) -> None:
    lines = [
        f"# Tournament Variant Report: {row.get('experiment_name', output_dir.parent.name)}",
        "",
        "This report is a bounded tournament-gate artifact. It is not a final research conclusion or investment recommendation.",
        "",
        f"- Family: `{row.get('family', '')}`",
        f"- Variant: `{row.get('variant', '')}`",
        f"- Benchmark: `{row.get('tournament_benchmark_symbol', '')}`",
        f"- Status: `{row.get('status', '')}`",
        f"- Final equity ratio: `{row.get('final_equity_ratio', '')}`",
        f"- Excess CAGR: `{row.get('excess_cagr', '')}`",
        f"- Strategy max drawdown: `{row.get('strategy_max_dd', '')}`",
        f"- Output directory: `{output_dir}`",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_variant_success_artifacts(
    *,
    output_dir: Path,
    base_config: Dict[str, Any],
    row: Dict[str, Any],
    wf_table: pd.DataFrame,
    stitched: pd.DataFrame,
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    config_path_label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_error in ("error_summary.csv", "error_report.md"):
        (output_dir / stale_error).unlink(missing_ok=True)
    stitched.to_csv(output_dir / "stitched_equity.csv")
    wf_table.to_csv(output_dir / "walk_forward_windows.csv", index=False)
    summary.to_csv(output_dir / "same_period_benchmark_summary.csv", index=False)
    yearly.to_csv(output_dir / "yearly_returns.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "command": "run-tournament",
                "config_path": config_path_label,
                "variant": row.get("variant", ""),
                "family": row.get("family", ""),
                "category": row.get("category", ""),
                "tournament_benchmark_symbol": row.get("tournament_benchmark_symbol", ""),
                **base_config,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _write_variant_report(output_dir=output_dir, row=row)


def _write_variant_error_artifacts(
    *,
    output_dir: Path,
    row: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_success in (
        "stitched_equity.csv",
        "walk_forward_windows.csv",
        "same_period_benchmark_summary.csv",
        "yearly_returns.csv",
        "run_config.json",
        "report.md",
        "constant_leverage_benchmark_summary.csv",
        "voltarget_fairness_report.md",
    ):
        (output_dir / stale_success).unlink(missing_ok=True)
    for stale_native in output_dir.glob("soxl_vs_*"):
        if stale_native.is_file():
            stale_native.unlink(missing_ok=True)
    pd.DataFrame([row]).to_csv(output_dir / "error_summary.csv", index=False)
    lines = [
        f"# Tournament Variant Error: {row.get('experiment_name', output_dir.parent.name)}",
        "",
        "This variant failed during the bounded tournament gate.",
        "",
        f"- Family: `{row.get('family', '')}`",
        f"- Variant: `{row.get('variant', '')}`",
        f"- Status: `{row.get('status', '')}`",
        f"- Error: `{row.get('error', '')}`",
        "",
    ]
    (output_dir / "error_report.md").write_text("\n".join(lines), encoding="utf-8")


def _load_entry_config(tournament_path: Path, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    if "config" in entry:
        config_path = _resolve_config_path(tournament_path, entry["config"])
        config = load_yaml_file(config_path)
        config_path_label = str(config_path)
    else:
        config = dict(entry.get("experiment", entry))
        config_path_label = ""

    overrides = entry.get("overrides")
    if isinstance(overrides, dict):
        config.update(overrides)
    return config, config_path_label


def _run_tournament_variant(
    *,
    base_config: Dict[str, Any],
    data: pd.DataFrame,
    family: str,
    category: str,
    config_path_label: str,
    tournament_benchmark_symbol: str,
    wf_variant: Dict[str, Any],
    cost_bps: float,
    execution_model: str = CLOSE_TO_CLOSE_SHIFTED,
) -> Dict[str, Any]:
    variant_config = copy.deepcopy(base_config)
    variant_config["train_years"] = int(wf_variant["train_years"])
    variant_config["test_years"] = int(wf_variant["test_years"])
    variant_config["walk_forward_mode"] = str(wf_variant.get("walk_forward_mode", "rolling"))
    variant_config["transaction_cost_bps"] = float(cost_bps)
    variant_config["execution_model"] = str(execution_model)
    variant_config["anti_overfit_validation"] = {"enabled": False}
    variant_name = _variant_name(wf_variant, cost_bps, execution_model)
    variant_output_dir = _variant_output_dir(base_config, variant_name)
    estimate = estimate_experiment_workload(
        variant_config,
        include_full_experiment_overhead=False,
    )
    started = time.perf_counter()

    row: Dict[str, Any] = {
        "family": family,
        "category": category,
        "experiment_name": str(base_config.get("experiment_name", family)),
        "config_path": config_path_label,
        "strategy_name": str(base_config.get("strategy_name", "")),
        "selection_benchmark_symbol": str(base_config.get("benchmark_symbol", "")).upper(),
        "tournament_benchmark_symbol": tournament_benchmark_symbol,
        "variant": variant_name,
        "walk_forward_variant": str(wf_variant["name"]),
        "train_years": int(wf_variant["train_years"]),
        "test_years": int(wf_variant["test_years"]),
        "walk_forward_mode": str(wf_variant.get("walk_forward_mode", "rolling")),
        "transaction_cost_bps": float(cost_bps),
        "execution_model": str(execution_model),
        "status": "error",
        "error": "",
        "output_dir": str(variant_output_dir),
        **estimate,
    }

    try:
        wf_table, stitched, grid_size = _run_walk_forward(variant_config, data)
        actual_runtime_seconds = time.perf_counter() - started
        row["grid_size"] = int(grid_size)
        row["walk_forward_windows"] = int(len(wf_table))
        row["stitched_rows"] = int(len(stitched))
        row["actual_runtime_seconds"] = float(actual_runtime_seconds)
        estimated_evaluations = float(row.get("estimated_evaluations", np.nan))
        row["actual_seconds_per_estimated_evaluation"] = (
            float(actual_runtime_seconds / estimated_evaluations)
            if pd.notna(estimated_evaluations) and estimated_evaluations > 0
            else np.nan
        )
        if stitched.empty:
            row["status"] = "no_result"
            row["error"] = "variant produced no stitched equity"
            _write_variant_error_artifacts(output_dir=variant_output_dir, row=row)
            return row

        summary, yearly = compare_to_benchmark(
            stitched,
            data,
            benchmark_symbol=tournament_benchmark_symbol,
        )
        metrics = summary.iloc[0].to_dict()
        row.update(metrics)
        selection_benchmark = str(base_config.get("benchmark_symbol", "")).upper()
        if selection_benchmark and selection_benchmark != tournament_benchmark_symbol and selection_benchmark in data.columns:
            variant_output_dir.mkdir(parents=True, exist_ok=True)
            native_summary, native_yearly = compare_to_benchmark(
                stitched,
                data,
                benchmark_symbol=selection_benchmark,
            )
            native_metrics = native_summary.iloc[0].to_dict()
            row["native_benchmark_symbol"] = selection_benchmark
            row["native_final_equity_ratio"] = native_metrics.get("final_equity_ratio", np.nan)
            row["native_strategy_final_equity"] = native_metrics.get("strategy_final_equity", np.nan)
            row["native_benchmark_final_equity"] = native_metrics.get("benchmark_final_equity", np.nan)
            row["native_excess_cagr"] = native_metrics.get("excess_cagr", np.nan)
            lower = selection_benchmark.replace("^", "").lower()
            native_summary.to_csv(variant_output_dir / f"soxl_vs_{lower}_summary.csv", index=False)
            native_yearly.to_csv(variant_output_dir / f"soxl_vs_{lower}_yearly_returns.csv", index=False)
        if _is_vol_target_strategy(variant_config):
            write_voltarget_fairness_outputs(
                output_dir=variant_output_dir,
                stitched=stitched,
                price_data=data,
                transaction_cost_bps=float(cost_bps),
                wf_table=wf_table,
                config=variant_config,
                trade_asset=_asset_config(variant_config).trade_asset,
                execution_model=str(execution_model),
            )
        row["status"] = "ok"
        row["error"] = ""
        _write_variant_success_artifacts(
            output_dir=variant_output_dir,
            base_config=variant_config,
            row=row,
            wf_table=wf_table,
            stitched=stitched,
            summary=summary,
            yearly=yearly,
            config_path_label=config_path_label,
        )
        return row
    except Exception as exc:
        actual_runtime_seconds = time.perf_counter() - started
        row["actual_runtime_seconds"] = float(actual_runtime_seconds)
        row["error"] = str(exc)
        _write_variant_error_artifacts(output_dir=variant_output_dir, row=row)
        return row


def _safe_float(row: pd.Series, column: str) -> float:
    value = row.get(column, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _first_ok_variant(group: pd.DataFrame, variant: str) -> Optional[pd.Series]:
    rows = group[(group["variant"] == variant) & (group["status"] == "ok")]
    if rows.empty:
        return None
    return rows.iloc[0]


def _rejection_reason(
    *,
    baseline_ratio: float,
    cost_25_ratio: float,
    worst_ratio: float,
    successful_variants: int,
    failed_variants: int,
    variant_win_rate: float,
) -> str:
    reasons = []
    if successful_variants == 0:
        reasons.append("all variants failed")
    if failed_variants > 0:
        reasons.append(f"{failed_variants} variant run(s) failed")
    if not pd.notna(baseline_ratio) or baseline_ratio <= 1.0:
        reasons.append("standard 5/1 at 10 bps did not beat TQQQ")
    if not pd.notna(cost_25_ratio) or cost_25_ratio <= 1.0:
        reasons.append("standard 5/1 did not remain competitive at 25 bps")
    if pd.notna(worst_ratio) and worst_ratio <= 1.0:
        reasons.append("at least one validation variant failed raw TQQQ outperformance")
    if pd.notna(variant_win_rate) and variant_win_rate < 0.5:
        reasons.append("fewer than half of validation variants beat TQQQ")
    return "; ".join(reasons)


def _robustness_score(
    *,
    median_ratio: float,
    worst_ratio: float,
    variant_win_rate: float,
    worst_max_dd: float,
    cost_25_ratio: float,
    failed_variants: int,
    total_variants: int,
) -> float:
    if total_variants <= 0:
        return np.nan
    failure_penalty = float(failed_variants) / float(total_variants)
    drawdown_component = 1.0 + float(worst_max_dd) if pd.notna(worst_max_dd) else 0.0
    values = {
        "median_ratio": float(median_ratio) if pd.notna(median_ratio) else 0.0,
        "worst_ratio": float(worst_ratio) if pd.notna(worst_ratio) else 0.0,
        "variant_win_rate": float(variant_win_rate) if pd.notna(variant_win_rate) else 0.0,
        "drawdown_component": drawdown_component,
        "cost_25_ratio": float(cost_25_ratio) if pd.notna(cost_25_ratio) else 0.0,
    }
    return float(
        0.35 * values["median_ratio"]
        + 0.25 * values["worst_ratio"]
        + 0.15 * values["variant_win_rate"]
        + 0.10 * values["drawdown_component"]
        + 0.15 * values["cost_25_ratio"]
        - 0.25 * failure_penalty
    )


def summarize_tournament(variant_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if variant_results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    group_cols = ["family", "category", "experiment_name", "config_path"]
    for keys, group in variant_results.groupby(group_cols, dropna=False):
        family, category, experiment_name, config_path = keys
        ok = group[group["status"] == "ok"].copy()
        total_variants = int(len(group))
        successful_variants = int(len(ok))
        failed_variants = int((group["status"] != "ok").sum())

        ratio_series = pd.to_numeric(ok.get("final_equity_ratio", pd.Series(dtype=float)), errors="coerce")
        median_ratio = float(ratio_series.median()) if not ratio_series.dropna().empty else np.nan
        worst_ratio = float(ratio_series.min()) if not ratio_series.dropna().empty else np.nan
        native_ratio_series = pd.to_numeric(
            ok.get("native_final_equity_ratio", pd.Series(dtype=float)),
            errors="coerce",
        )
        median_native_ratio = (
            float(native_ratio_series.median()) if not native_ratio_series.dropna().empty else np.nan
        )
        worst_native_ratio = (
            float(native_ratio_series.min()) if not native_ratio_series.dropna().empty else np.nan
        )
        variants_beating = int((ratio_series > 1.0).sum()) if not ratio_series.dropna().empty else 0
        variant_win_rate = variants_beating / successful_variants if successful_variants else np.nan

        max_dd_series = pd.to_numeric(ok.get("strategy_max_dd", pd.Series(dtype=float)), errors="coerce")
        worst_max_dd = float(max_dd_series.min()) if not max_dd_series.dropna().empty else np.nan
        estimated_combinations = pd.to_numeric(
            group.get("estimated_parameter_combinations", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
        estimated_windows = pd.to_numeric(
            group.get("estimated_walk_forward_windows", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
        estimated_evaluations = pd.to_numeric(
            group.get("estimated_evaluations", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
        estimated_runtime = pd.to_numeric(
            group.get("estimated_runtime_seconds", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
        actual_runtime = pd.to_numeric(
            group.get("actual_runtime_seconds", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
        actual_seconds_per_eval = (
            float(actual_runtime / estimated_evaluations)
            if pd.notna(estimated_evaluations) and estimated_evaluations > 0
            else np.nan
        )

        baseline_variant = "standard_5y_1y__10bps"
        baseline = _first_ok_variant(group, baseline_variant)
        if baseline is None and not ok.empty:
            baseline = ok.sort_values("final_equity_ratio", ascending=False).iloc[0]
            baseline_variant = str(baseline["variant"])

        cost_10 = _first_ok_variant(group, "standard_5y_1y__10bps")
        cost_25 = _first_ok_variant(group, "standard_5y_1y__25bps")
        cost_50 = _first_ok_variant(group, "standard_5y_1y__50bps")
        cost_10_ratio = _safe_float(cost_10, "final_equity_ratio") if cost_10 is not None else np.nan
        cost_25_ratio = _safe_float(cost_25, "final_equity_ratio") if cost_25 is not None else np.nan
        cost_50_ratio = _safe_float(cost_50, "final_equity_ratio") if cost_50 is not None else np.nan
        sensitivity_25 = cost_25_ratio / cost_10_ratio if pd.notna(cost_25_ratio) and pd.notna(cost_10_ratio) and cost_10_ratio != 0.0 else np.nan
        sensitivity_50 = cost_50_ratio / cost_10_ratio if pd.notna(cost_50_ratio) and pd.notna(cost_10_ratio) and cost_10_ratio != 0.0 else np.nan

        baseline_ratio = _safe_float(baseline, "final_equity_ratio") if baseline is not None else np.nan
        rejection_reason = _rejection_reason(
            baseline_ratio=baseline_ratio,
            cost_25_ratio=cost_25_ratio,
            worst_ratio=worst_ratio,
            successful_variants=successful_variants,
            failed_variants=failed_variants,
            variant_win_rate=variant_win_rate,
        )
        robustness = _robustness_score(
            median_ratio=median_ratio,
            worst_ratio=worst_ratio,
            variant_win_rate=variant_win_rate,
            worst_max_dd=worst_max_dd,
            cost_25_ratio=cost_25_ratio,
            failed_variants=failed_variants,
            total_variants=total_variants,
        )

        rows.append(
            {
                "family": family,
                "category": category,
                "experiment_name": experiment_name,
                "config_path": config_path,
                "strategy_name": str(group["strategy_name"].iloc[0]),
                "selection_benchmark_symbol": str(group["selection_benchmark_symbol"].iloc[0]),
                "tournament_benchmark_symbol": str(group["tournament_benchmark_symbol"].iloc[0]),
                "baseline_variant": baseline_variant if baseline is not None else "",
                "baseline_final_equity_ratio": baseline_ratio,
                "baseline_excess_cagr": _safe_float(baseline, "excess_cagr") if baseline is not None else np.nan,
                "baseline_strategy_cagr": _safe_float(baseline, "strategy_cagr") if baseline is not None else np.nan,
                "baseline_strategy_max_dd": _safe_float(baseline, "strategy_max_dd") if baseline is not None else np.nan,
                "baseline_strategy_calmar": _safe_float(baseline, "strategy_calmar") if baseline is not None else np.nan,
                "native_benchmark_symbol": str(baseline.get("native_benchmark_symbol", "")) if baseline is not None else "",
                "baseline_native_final_equity_ratio": _safe_float(baseline, "native_final_equity_ratio") if baseline is not None else np.nan,
                "median_native_final_equity_ratio": median_native_ratio,
                "worst_native_final_equity_ratio": worst_native_ratio,
                "median_final_equity_ratio": median_ratio,
                "worst_final_equity_ratio": worst_ratio,
                "validation_variants_beating_tqqq": variants_beating,
                "successful_variants": successful_variants,
                "failed_variants": failed_variants,
                "total_variants": total_variants,
                "variant_win_rate": variant_win_rate,
                "worst_strategy_max_dd": worst_max_dd,
                "cost_10_final_equity_ratio": cost_10_ratio,
                "cost_25_final_equity_ratio": cost_25_ratio,
                "cost_50_final_equity_ratio": cost_50_ratio,
                "cost_sensitivity_25_vs_10": sensitivity_25,
                "cost_sensitivity_50_vs_10": sensitivity_50,
                "robustness_score": robustness,
                "accepted_candidate": rejection_reason == "",
                "rejection_reason": rejection_reason,
                "output_dir": "",
                "estimated_parameter_combinations": float(estimated_combinations),
                "estimated_walk_forward_windows": float(estimated_windows),
                "estimated_evaluations": float(estimated_evaluations),
                "estimated_runtime_seconds": float(estimated_runtime),
                "actual_runtime_seconds": float(actual_runtime),
                "actual_seconds_per_estimated_evaluation": actual_seconds_per_eval,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _failure_report(variant_results: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    failed_runs = variant_results[variant_results["status"] != "ok"].copy()
    if not failed_runs.empty:
        failed_runs["record_type"] = "failed_run"
        failed_runs["rejection_reason"] = failed_runs["error"]
    rejected = summary[summary["rejection_reason"].astype(str) != ""].copy()
    if not rejected.empty:
        rejected["record_type"] = "rejected_strategy"
        rejected["variant"] = ""
        rejected["status"] = "rejected"
        rejected["error"] = rejected["rejection_reason"]
    columns = list(
        dict.fromkeys(
            ["record_type", "family", "category", "experiment_name", "config_path", "variant", "status", "error", "rejection_reason"]
            + list(failed_runs.columns if not failed_runs.empty else [])
            + list(rejected.columns if not rejected.empty else [])
        )
    )
    frames = [frame for frame in (failed_runs, rejected) if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False).reindex(columns=columns) if frames else pd.DataFrame(columns=columns)


def _print_conclusion(summary: pd.DataFrame) -> None:
    print("\n=== Tournament conclusion ===")
    if summary.empty:
        print("No tournament summary rows were produced.")
        return

    best_raw = summary.sort_values("baseline_final_equity_ratio", ascending=False).iloc[0]
    best_risk = summary.sort_values("robustness_score", ascending=False).iloc[0]
    soxl = summary[summary["category"].astype(str).str.lower().str.contains("soxl")]
    best_soxl = soxl.sort_values("baseline_final_equity_ratio", ascending=False).iloc[0] if not soxl.empty else None
    rejected = summary[summary["rejection_reason"].astype(str) != ""]

    print(
        "BEST RAW OUTPERFORMER: "
        f"{best_raw['family']} | ratio={float(best_raw['baseline_final_equity_ratio']):.6f}"
    )
    print(
        "BEST RISK-ADJUSTED CANDIDATE: "
        f"{best_risk['family']} | robustness={float(best_risk['robustness_score']):.6f}"
    )
    if best_soxl is not None:
        print(
            "BEST SOXL CANDIDATE: "
            f"{best_soxl['family']} | ratio={float(best_soxl['baseline_final_equity_ratio']):.6f}"
        )
    else:
        print("BEST SOXL CANDIDATE: none run")

    print("STRATEGIES REJECTED AND WHY:")
    if rejected.empty:
        print("None.")
    else:
        for _, row in rejected.iterrows():
            print(f"- {row['family']}: {row['rejection_reason']}")


def _markdown_table(df: pd.DataFrame, columns: Optional[List[str]] = None, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No rows._"
    table = df.copy()
    if columns is not None:
        table = table[[column for column in columns if column in table.columns]]
    table = table.head(max_rows).fillna("")
    if table.empty:
        return "_No rows._"
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row.get(column, "")).replace("\n", " ") for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _sector_bet_mask(summary: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=bool)
    tokens = config.get("sector_bet_categories", ["soxl"])
    if isinstance(tokens, str):
        tokens = [tokens]
    lowered = [str(token).lower() for token in tokens]
    categories = summary.get("category", pd.Series("", index=summary.index)).astype(str).str.lower()
    return categories.map(lambda value: any(token in value for token in lowered))


def _write_ranked_aliases(output_dir: Path, summary: pd.DataFrame, config: Dict[str, Any]) -> None:
    if summary.empty:
        empty = pd.DataFrame(columns=SUMMARY_COLUMNS)
        empty.to_csv(output_dir / "ranked_by_final_equity_ratio.csv", index=False)
        empty.to_csv(output_dir / "ranked_by_robustness.csv", index=False)
        return

    sector_mask = _sector_bet_mask(summary, config) if bool(config.get("separate_sector_bets", False)) else pd.Series(False, index=summary.index)
    tqqq_summary = summary[~sector_mask].copy()
    sector_summary = summary[sector_mask].copy()
    if tqqq_summary.empty:
        tqqq_summary = summary.copy()

    tqqq_summary.sort_values("baseline_final_equity_ratio", ascending=False).to_csv(
        output_dir / "ranked_by_final_equity_ratio.csv",
        index=False,
    )
    tqqq_summary.sort_values("robustness_score", ascending=False).to_csv(
        output_dir / "ranked_by_robustness.csv",
        index=False,
    )

    if not sector_summary.empty:
        sector_summary.sort_values("baseline_final_equity_ratio", ascending=False).to_csv(
            output_dir / "sector_bet_ranked_by_final_equity_ratio.csv",
            index=False,
        )
        sector_summary.sort_values("robustness_score", ascending=False).to_csv(
            output_dir / "sector_bet_ranked_by_robustness.csv",
            index=False,
        )


def _write_voltarget_deep_report(
    *,
    output_dir: Path,
    config: Dict[str, Any],
    summary: pd.DataFrame,
    variants: pd.DataFrame,
    failures: pd.DataFrame,
) -> Optional[Path]:
    if not bool(config.get("voltarget_deep_report", False)) and str(config.get("report_type", "")) != "voltarget_deep":
        return None

    report_path = output_dir / "voltarget_deep_report.md"
    contribution = build_one_year_contribution_from_variants(summary, variants)
    contribution.to_csv(output_dir / "one_year_contribution.csv", index=False)
    sections = [
        ("TQQQ VolTarget candidates", "tqqq_voltarget"),
        ("VolTarget + Governor candidates", "tqqq_voltarget_governor"),
        ("MarketInternals VolTarget candidates", "tqqq_market_internals"),
        ("SOXL sector-bet candidates", "soxl_sector_bet"),
        ("CoreOverlay + Rebound comparator", "secondary_comparator"),
    ]
    lines = [
        f"# VolTarget Deep Validation Report: {config.get('tournament_name', output_dir.name)}",
        "",
        "This report summarizes empirical backtest outputs only. It is not an investment recommendation.",
        "",
        "## Objective",
        "Run a bounded deep-validation tournament focused on VolTarget families before any exhaustive tournament.",
        "",
        "## Ranking Scope",
        "`ranked_by_final_equity_ratio.csv` and `ranked_by_robustness.csv` exclude SOXL sector-bet rows. SOXL rows are kept in `sector_bet_ranked_by_final_equity_ratio.csv` and `sector_bet_ranked_by_robustness.csv`.",
        "",
    ]

    for title, category in sections:
        rows = summary[summary.get("category", pd.Series(dtype=str)).astype(str) == category].copy()
        lines.extend(
            [
                f"## {title}",
                _markdown_table(
                    rows.sort_values("baseline_final_equity_ratio", ascending=False) if not rows.empty else rows,
                    [
                        "family",
                        "baseline_final_equity_ratio",
                        "median_final_equity_ratio",
                        "worst_final_equity_ratio",
                        "baseline_strategy_max_dd",
                        "cost_10_final_equity_ratio",
                        "cost_25_final_equity_ratio",
                        "cost_50_final_equity_ratio",
                        "robustness_score",
                        "native_benchmark_symbol",
                        "baseline_native_final_equity_ratio",
                        "accepted_candidate",
                        "rejection_reason",
                    ],
                    max_rows=20,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Cost sensitivity",
            _markdown_table(
                summary.sort_values("baseline_final_equity_ratio", ascending=False) if not summary.empty else summary,
                [
                    "family",
                    "category",
                    "cost_10_final_equity_ratio",
                    "cost_25_final_equity_ratio",
                    "cost_50_final_equity_ratio",
                    "cost_sensitivity_25_vs_10",
                    "cost_sensitivity_50_vs_10",
                ],
                max_rows=30,
            ),
            "",
            "## Alternate walk-forward variants",
            _markdown_table(
                variants[
                    variants.get("walk_forward_variant", pd.Series(dtype=str)).astype(str)
                    != "standard_5y_1y"
                ].sort_values("final_equity_ratio", ascending=False)
                if not variants.empty and "final_equity_ratio" in variants.columns
                else variants,
                [
                    "family",
                    "category",
                    "variant",
                    "walk_forward_variant",
                    "walk_forward_mode",
                    "execution_model",
                    "transaction_cost_bps",
                    "final_equity_ratio",
                    "strategy_max_dd",
                    "native_final_equity_ratio",
                    "status",
                    "error",
                ],
                max_rows=40,
            ),
            "",
            "## Execution-model sensitivity",
            _markdown_table(
                variants.sort_values("final_equity_ratio", ascending=False)
                if not variants.empty and "final_equity_ratio" in variants.columns
                else variants,
                [
                    "family",
                    "category",
                    "variant",
                    "execution_model",
                    "final_equity_ratio",
                    "strategy_max_dd",
                    "status",
                    "error",
                ],
                max_rows=40,
            ),
            "",
            "## Failures and empirical rejections",
            _markdown_table(
                failures,
                ["record_type", "family", "category", "variant", "status", "error", "rejection_reason"],
                max_rows=40,
            ),
            "",
            "## One-year contribution audit",
            "Contribution is computed from yearly log excess return: `log(1 + strategy_return) - log(1 + TQQQ_return)`. A candidate is flagged when one year contributes more than 60% of positive total log excess return.",
            "",
            _markdown_table(
                contribution.drop_duplicates(
                    subset=["family", "category", "variant"],
                    keep="first",
                )
                if not contribution.empty
                else contribution,
                [
                    "family",
                    "category",
                    "variant",
                    "largest_contribution_year",
                    "largest_single_year_contribution_share",
                    "one_year_contributes_more_than_60pct",
                    "audit_status",
                ],
                max_rows=30,
            ),
            "",
            markdown_one_year_contribution_table(contribution, max_rows=40),
            "",
            "## Interpretation Guardrails",
            "- SOXL rows are sector-bet diagnostics and are not mixed into the TQQQ-only ranking files.",
            "- Passing this bounded tournament is not final candidate extraction; full validation artifacts are still required.",
            "- Results are sensitive to data vintage, execution model, transaction costs, and parameter grid design.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _tournament_dry_run_rows(
    *,
    tournament_path: Path,
    tournament_config: Dict[str, Any],
    entries: List[Dict[str, Any]],
    output_dir: Path,
    tournament_benchmark_symbol: str,
    variants: List[Dict[str, Any]],
    costs: List[float],
    execution_models: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        family = str(entry.get("family", entry.get("name", entry.get("config", "strategy"))))
        category = str(entry.get("category", "general"))
        try:
            base_config, config_path_label = _load_entry_config(tournament_path, entry)
            base_config.setdefault("experiment_name", family.lower().replace(" ", "_"))
            base_config.setdefault("output_dir", str(output_dir / "experiments" / base_config["experiment_name"]))
            total_estimated_combinations = 0
            total_estimated_windows = 0
            total_estimated_evaluations = 0
            total_estimated_runtime = 0.0
            for wf_variant in variants:
                for cost in costs:
                    for execution_model in execution_models:
                        variant_config = copy.deepcopy(base_config)
                        variant_config["train_years"] = int(wf_variant["train_years"])
                        variant_config["test_years"] = int(wf_variant["test_years"])
                        variant_config["walk_forward_mode"] = str(wf_variant.get("walk_forward_mode", "rolling"))
                        variant_config["transaction_cost_bps"] = float(cost)
                        variant_config["execution_model"] = str(execution_model)
                        variant_config["anti_overfit_validation"] = {"enabled": False}
                        estimate = estimate_experiment_workload(
                            variant_config,
                            include_full_experiment_overhead=False,
                        )
                        total_estimated_combinations += int(estimate.get("estimated_parameter_combinations", 0))
                        total_estimated_windows += int(estimate.get("estimated_walk_forward_windows", 0))
                        total_estimated_evaluations += int(estimate.get("estimated_evaluations", 0))
                        total_estimated_runtime += float(estimate.get("estimated_runtime_seconds", 0.0))
            rows.append(
                {
                    "status": "dry_run",
                    "family": family,
                    "category": category,
                    "experiment_name": str(base_config.get("experiment_name", "")),
                    "config_path": config_path_label,
                    "strategy_name": str(base_config.get("strategy_name", "")),
                    "symbols": ";".join(str(symbol).upper() for symbol in base_config.get("symbols", [])),
                    "selection_benchmark_symbol": str(base_config.get("benchmark_symbol", "")).upper(),
                    "tournament_benchmark_symbol": tournament_benchmark_symbol,
                    "output_dir": str(base_config.get("output_dir", "")),
                    "variant_count": len(variants) * len(costs) * len(execution_models),
                    "execution_models": ";".join(execution_models),
                    "estimated_parameter_combinations": total_estimated_combinations,
                    "estimated_walk_forward_windows": total_estimated_windows,
                    "estimated_evaluations": total_estimated_evaluations,
                    "estimated_runtime_seconds": total_estimated_runtime,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "status": "dry_run_error",
                    "family": family,
                    "category": category,
                    "config_path": str(entry.get("config", "")),
                    "tournament_benchmark_symbol": tournament_benchmark_symbol,
                    "variant_count": len(variants) * len(costs) * len(execution_models),
                    "execution_models": ";".join(execution_models),
                    "error": str(exc),
                }
            )
    summary = pd.DataFrame(rows)
    summary["tournament_total_configs"] = len(_strategy_entries(tournament_config))
    summary["tournament_selected_configs"] = len(entries)
    return summary


def run_tournament_config(
    tournament_path: Path,
    *,
    max_configs: Optional[int] = None,
    dry_run: bool = False,
    allow_long_run: bool = False,
    accept_baseline_regression: bool = False,
) -> pd.DataFrame:
    tournament_path = Path(tournament_path)
    tournament_config = load_yaml_file(tournament_path)
    tournament_name = str(tournament_config.get("tournament_name", tournament_path.stem))
    output_dir = Path(tournament_config.get("output_dir", Path("outputs") / tournament_name))
    output_dir.mkdir(parents=True, exist_ok=True)

    tournament_benchmark_symbol = str(tournament_config.get("benchmark_symbol", "TQQQ")).upper()
    variants = _walk_forward_variants(tournament_config)
    costs = _cost_scenarios(tournament_config)
    execution_models = _execution_models(tournament_config)
    entries = _strategy_entries(tournament_config)
    total_configs = len(entries)
    if max_configs is not None:
        entries = entries[: max(0, int(max_configs))]

    if dry_run:
        summary = _tournament_dry_run_rows(
            tournament_path=tournament_path,
            tournament_config=tournament_config,
            entries=entries,
            output_dir=output_dir,
            tournament_benchmark_symbol=tournament_benchmark_symbol,
            variants=variants,
            costs=costs,
            execution_models=execution_models,
        )
        summary.to_csv(output_dir / "tournament_dry_run_summary.csv", index=False)
        print(summary.to_string(index=False))
        print(f"\nDry-run summary written to: {output_dir / 'tournament_dry_run_summary.csv'}")
        return summary

    enforce_baseline_gate(
        accept_baseline_regression=accept_baseline_regression,
        action="running a full tournament",
    )

    guard_summary = _tournament_dry_run_rows(
        tournament_path=tournament_path,
        tournament_config=tournament_config,
        entries=entries,
        output_dir=output_dir,
        tournament_benchmark_symbol=tournament_benchmark_symbol,
        variants=variants,
        costs=costs,
        execution_models=execution_models,
    )
    max_estimated_runtime = float(
        tournament_config.get("max_estimated_runtime_seconds", DEFAULT_MAX_ESTIMATED_RUNTIME_SECONDS)
    )
    estimated_runtime = float(
        pd.to_numeric(guard_summary.get("estimated_runtime_seconds", pd.Series(dtype=float)), errors="coerce").sum()
    )
    guard_summary["max_estimated_runtime_seconds"] = max_estimated_runtime
    guard_summary["allow_long_run"] = bool(allow_long_run)
    guard_summary.to_csv(output_dir / "tournament_runtime_guard_summary.csv", index=False)
    if estimated_runtime > max_estimated_runtime and not allow_long_run:
        raise RuntimeError(
            "Tournament estimated runtime exceeds safety threshold: "
            f"{estimated_runtime:.3f}s > {max_estimated_runtime:.3f}s. "
            "Run with --dry-run, use a smaller config such as configs/tournament_gate.yaml, "
            "or pass --allow-long-run intentionally."
        )

    rows: List[Dict[str, Any]] = []
    tournament_started = time.perf_counter()
    for entry in entries:
        family = str(entry.get("family", entry.get("name", entry.get("config", "strategy"))))
        category = str(entry.get("category", "general"))
        config_path_label = ""
        try:
            base_config, config_path_label = _load_entry_config(tournament_path, entry)
            base_config.setdefault("experiment_name", family.lower().replace(" ", "_"))
            base_config.setdefault("output_dir", str(output_dir / "experiments" / base_config["experiment_name"]))
            data = _load_price_data(base_config)
            if tournament_benchmark_symbol not in data.columns:
                raise ValueError(
                    f"Tournament benchmark {tournament_benchmark_symbol} is missing from data for {family}."
                )
            for wf_variant in variants:
                for cost in costs:
                    for execution_model in execution_models:
                        rows.append(
                            _run_tournament_variant(
                                base_config=base_config,
                                data=data,
                                family=family,
                                category=category,
                                config_path_label=config_path_label,
                                tournament_benchmark_symbol=tournament_benchmark_symbol,
                                wf_variant=wf_variant,
                                cost_bps=cost,
                                execution_model=execution_model,
                            )
                        )
        except Exception as exc:
            for wf_variant in variants:
                for cost in costs:
                    for execution_model in execution_models:
                        rows.append(
                            {
                                "family": family,
                                "category": category,
                                "experiment_name": family.lower().replace(" ", "_"),
                                "config_path": config_path_label or str(entry.get("config", "")),
                                "strategy_name": "",
                                "selection_benchmark_symbol": "",
                                "tournament_benchmark_symbol": tournament_benchmark_symbol,
                                "variant": _variant_name(wf_variant, float(cost), str(execution_model)),
                                "walk_forward_variant": str(wf_variant["name"]),
                                "train_years": int(wf_variant["train_years"]),
                                "test_years": int(wf_variant["test_years"]),
                                "walk_forward_mode": str(wf_variant.get("walk_forward_mode", "rolling")),
                                "transaction_cost_bps": float(cost),
                                "execution_model": str(execution_model),
                                "status": "error",
                                "error": str(exc),
                            }
                        )

    variant_results = pd.DataFrame(rows)
    summary = summarize_tournament(variant_results)
    summary["output_dir"] = str(output_dir)
    summary["tournament_total_configs"] = total_configs
    summary["tournament_selected_configs"] = len(entries)
    summary["tournament_actual_runtime_seconds"] = time.perf_counter() - tournament_started
    failures = _failure_report(variant_results, summary)

    variant_results.to_csv(output_dir / "tournament_variant_results.csv", index=False)
    summary.to_csv(output_dir / "tournament_summary.csv", index=False)
    summary.sort_values("baseline_final_equity_ratio", ascending=False).to_csv(
        output_dir / "tournament_ranked_by_final_equity_ratio.csv",
        index=False,
    )
    summary.sort_values("robustness_score", ascending=False).to_csv(
        output_dir / "tournament_ranked_by_robustness.csv",
        index=False,
    )
    failures.to_csv(output_dir / "tournament_failures.csv", index=False)
    _write_ranked_aliases(output_dir, summary, tournament_config)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "command": "run-tournament",
                "config_path": str(tournament_path),
                **tournament_config,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    generate_tournament_report(output_dir=output_dir, config=tournament_config)
    _write_voltarget_deep_report(
        output_dir=output_dir,
        config=tournament_config,
        summary=summary,
        variants=variant_results,
        failures=failures,
    )
    if bool(tournament_config.get("voltarget_stage2_report", False)) or str(
        tournament_config.get("report_type", "")
    ) == "voltarget_stage2_2022_sensitivity":
        write_voltarget_stage2_sensitivity_outputs(
            output_dir=output_dir,
            summary=summary,
            variants=variant_results,
        )
    if bool(tournament_config.get("voltarget_stage3_report", False)) or str(
        tournament_config.get("report_type", "")
    ) == "voltarget_stage3_robust_alpha":
        write_voltarget_stage3_outputs(
            output_dir=output_dir,
            summary=summary,
            variants=variant_results,
        )

    _print_conclusion(summary)
    print(f"\nTournament outputs written to: {output_dir}")
    return summary
