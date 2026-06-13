from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .reports import _markdown_table


SUMMARY_COLUMNS = [
    "strategy_config",
    "config_name",
    "strategy_name",
    "config_path",
    "output_dir",
    "status",
    "classification",
    "date_range",
    "start_date",
    "end_date",
    "stitched_rows",
    "strategy_final_equity",
    "benchmark_final_equity",
    "final_equity_ratio",
    "excess_cagr",
    "strategy_max_dd",
    "benchmark_max_dd",
    "yearly_wins",
    "total_years",
    "raw_outperformance_pass",
    "objective_used",
    "transaction_cost_bps",
    "benchmark_date_alignment",
    "position_shift_audit",
    "transaction_cost_audit",
    "output_contract_status",
    "dominant_excess_year",
    "dominant_excess_year_share",
    "one_year_drives_excess_return",
    "notes",
]


@dataclass(frozen=True)
class ControlledRunSpec:
    strategy_config: str
    output_dir: Path
    config_path: Optional[Path] = None
    summary_filename: str = "same_period_benchmark_summary.csv"
    yearly_filenames: Tuple[str, ...] = ("yearly_returns.csv", "same_period_yearly_returns.csv")
    stitched_filenames: Tuple[str, ...] = ("stitched_equity.csv", "regime_walk_forward_stitched_equity.csv")
    run_config_filename: str = "run_config.json"


DEFAULT_CONTROLLED_RUNS: Tuple[ControlledRunSpec, ...] = (
    ControlledRunSpec(
        strategy_config="run-existing-regime",
        output_dir=Path("ma_search_output_v11"),
        yearly_filenames=("same_period_yearly_returns.csv", "yearly_returns.csv"),
        stitched_filenames=("regime_walk_forward_stitched_equity.csv", "stitched_equity.csv"),
    ),
    ControlledRunSpec(
        strategy_config="configs/regime_tqqq.yaml",
        config_path=Path("configs/regime_tqqq.yaml"),
        output_dir=Path("outputs/regime_tqqq"),
    ),
    ControlledRunSpec(
        strategy_config="configs/core_overlay_tqqq.yaml",
        config_path=Path("configs/core_overlay_tqqq.yaml"),
        output_dir=Path("outputs/core_overlay_tqqq"),
    ),
    ControlledRunSpec(
        strategy_config="configs/core_overlay_with_rebound.yaml",
        config_path=Path("configs/core_overlay_with_rebound.yaml"),
        output_dir=Path("outputs/core_overlay_with_rebound"),
    ),
    ControlledRunSpec(
        strategy_config="configs/vol_target_tqqq.yaml",
        config_path=Path("configs/vol_target_tqqq.yaml"),
        output_dir=Path("outputs/vol_target_tqqq"),
    ),
)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _first_existing(base_dir: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        path = base_dir / name
        if path.exists():
            return path
    return None


def _read_mapping(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _objective_from_metadata(run_config: Dict[str, Any], config: Dict[str, Any]) -> str:
    for key in ("objective", "train_objective"):
        value = run_config.get(key)
        if value:
            return str(value)
    for key in ("objective", "train_objective"):
        value = config.get(key)
        if value:
            return str(value)
    objectives = config.get("objectives")
    if isinstance(objectives, list) and objectives:
        return str(objectives[0])
    return ""


def _cost_from_metadata(run_config: Dict[str, Any], config: Dict[str, Any]) -> float:
    for source in (run_config, config):
        if "transaction_cost_bps" in source:
            return _as_float(source.get("transaction_cost_bps"))
    return float("nan")


def _config_name(spec: ControlledRunSpec, config: Dict[str, Any]) -> str:
    value = config.get("experiment_name")
    if value:
        return str(value)
    if spec.config_path:
        return Path(spec.config_path).stem
    return spec.strategy_config


def _strategy_name(spec: ControlledRunSpec, config: Dict[str, Any]) -> str:
    value = config.get("strategy_name")
    if value:
        return str(value)
    if spec.strategy_config == "run-existing-regime":
        return "existing_regime"
    return ""


def _read_first_summary(path: Path) -> Dict[str, Any]:
    summary = pd.read_csv(path)
    if summary.empty:
        raise ValueError(f"Summary file is empty: {path}")
    return summary.iloc[0].to_dict()


def _date_alignment(summary: Dict[str, Any], stitched_path: Optional[Path]) -> str:
    if stitched_path is None or not stitched_path.exists():
        return "missing_stitched_equity"
    try:
        stitched = pd.read_csv(stitched_path)
        if stitched.empty:
            return "missing_stitched_equity"
        date_column = "Date" if "Date" in stitched.columns else stitched.columns[0]
        dates = pd.to_datetime(stitched[date_column], errors="coerce").dropna()
        if dates.empty:
            return "missing_stitched_dates"
        expected_start = pd.to_datetime(summary.get("start_date"), errors="coerce")
        expected_end = pd.to_datetime(summary.get("end_date"), errors="coerce")
        expected_rows = _as_int(summary.get("stitched_rows"))
        starts_match = pd.notna(expected_start) and dates.min().normalize() == expected_start.normalize()
        ends_match = pd.notna(expected_end) and dates.max().normalize() == expected_end.normalize()
        rows_match = len(dates) == expected_rows
        return "pass" if starts_match and ends_match and rows_match else "fail"
    except Exception as exc:
        return f"error: {exc}"


def _position_shift_audit(stitched_path: Optional[Path]) -> str:
    if stitched_path is None or not stitched_path.exists():
        return "missing_stitched_equity"
    try:
        stitched = pd.read_csv(stitched_path, nrows=200)
    except Exception as exc:
        return f"error: {exc}"
    if "execution_model" in stitched.columns:
        models = {str(value) for value in stitched["execution_model"].dropna().unique() if str(value)}
        if "close_to_close_shifted" in models:
            return "shifted_execution_model"
    if {"position", "tqqq_weight", "trade_weight"}.intersection(stitched.columns):
        return "position_columns_present_review_code_for_shift"
    return "no_position_columns"


def _transaction_cost_audit(
    stitched_path: Optional[Path],
    transaction_cost_bps: float,
) -> str:
    if pd.isna(transaction_cost_bps):
        return "unknown_cost_setting"
    if transaction_cost_bps <= 0:
        return "zero_bps_configured"
    if stitched_path is None or not stitched_path.exists():
        return "missing_stitched_equity"
    try:
        stitched = pd.read_csv(stitched_path)
    except Exception as exc:
        return f"error: {exc}"
    if "cost" not in stitched.columns:
        return "cost_column_missing"
    total_cost = pd.to_numeric(stitched["cost"], errors="coerce").fillna(0.0).sum()
    return "costs_present" if total_cost > 0 else "no_costs_detected"


def _yearly_stats(
    yearly_path: Optional[Path],
    summary: Dict[str, Any],
) -> Tuple[int, int, Any, float, bool]:
    yearly_wins = _as_int(summary.get("years_strategy_beats_benchmark"))
    total_years = _as_int(summary.get("total_years"))
    dominant_year: Any = ""
    dominant_share = float("nan")
    one_year_drives = False

    if yearly_path is None or not yearly_path.exists():
        return yearly_wins, total_years, dominant_year, dominant_share, one_year_drives

    yearly = pd.read_csv(yearly_path)
    if yearly.empty:
        return yearly_wins, total_years, dominant_year, dominant_share, one_year_drives

    if "strategy_beats_benchmark" in yearly.columns:
        yearly_wins = int(yearly["strategy_beats_benchmark"].map(_as_bool).sum())
        total_years = int(len(yearly))

    required = {"year", "strategy_return", "benchmark_return"}
    if required.issubset(yearly.columns):
        relative = pd.to_numeric(yearly["strategy_return"], errors="coerce") - pd.to_numeric(
            yearly["benchmark_return"],
            errors="coerce",
        )
        positive = relative.clip(lower=0.0)
        total_positive = float(positive.sum())
        if total_positive > 0:
            idx = int(positive.idxmax())
            dominant_year = yearly.loc[idx, "year"]
            dominant_share = float(positive.loc[idx] / total_positive)
            one_year_drives = bool(dominant_share > 0.60)

    return yearly_wins, total_years, dominant_year, dominant_share, one_year_drives


def _classify(final_equity_ratio: float, raw_pass: bool, status: str, audit_notes: Sequence[str]) -> str:
    if status != "ok":
        return "infrastructure failure"
    if raw_pass or final_equity_ratio > 1.0:
        if any(note.startswith("audit_fail") for note in audit_notes):
            return "interesting but insufficient"
        return "candidate for tournament_gate validation"
    if pd.notna(final_equity_ratio) and final_equity_ratio >= 0.95:
        return "interesting but insufficient"
    return "empirical rejection"


def _failure_row(
    spec: ControlledRunSpec,
    summary_path: Path,
    reason: str,
    *,
    output_contract_status: str,
) -> Dict[str, Any]:
    config = _read_mapping(spec.config_path)
    return {
        "strategy_config": spec.strategy_config,
        "config_name": _config_name(spec, config),
        "strategy_name": _strategy_name(spec, config),
        "config_path": str(spec.config_path or ""),
        "output_dir": str(spec.output_dir),
        "status": output_contract_status,
        "classification": "infrastructure failure",
        "date_range": "",
        "start_date": "",
        "end_date": "",
        "stitched_rows": 0,
        "strategy_final_equity": np.nan,
        "benchmark_final_equity": np.nan,
        "final_equity_ratio": np.nan,
        "excess_cagr": np.nan,
        "strategy_max_dd": np.nan,
        "benchmark_max_dd": np.nan,
        "yearly_wins": 0,
        "total_years": 0,
        "raw_outperformance_pass": False,
        "objective_used": "",
        "transaction_cost_bps": np.nan,
        "benchmark_date_alignment": "not_available",
        "position_shift_audit": "not_available",
        "transaction_cost_audit": "not_available",
        "output_contract_status": output_contract_status,
        "dominant_excess_year": "",
        "dominant_excess_year_share": np.nan,
        "one_year_drives_excess_return": False,
        "notes": f"{reason}: {summary_path}",
    }


def _summarize_one(spec: ControlledRunSpec) -> Dict[str, Any]:
    output_dir = Path(spec.output_dir)
    summary_path = output_dir / spec.summary_filename
    if not summary_path.exists():
        return _failure_row(
            spec,
            summary_path,
            "missing same-period benchmark summary",
            output_contract_status="missing_same_period_benchmark_summary",
        )

    try:
        summary = _read_first_summary(summary_path)
    except Exception as exc:
        return _failure_row(
            spec,
            summary_path,
            f"invalid same-period benchmark summary ({exc})",
            output_contract_status="invalid_same_period_benchmark_summary",
        )

    run_config = _read_mapping(output_dir / spec.run_config_filename)
    config = _read_mapping(spec.config_path)
    objective = _objective_from_metadata(run_config, config)
    transaction_cost_bps = _cost_from_metadata(run_config, config)
    yearly_path = _first_existing(output_dir, spec.yearly_filenames)
    stitched_path = _first_existing(output_dir, spec.stitched_filenames)
    yearly_wins, total_years, dominant_year, dominant_share, one_year_drives = _yearly_stats(yearly_path, summary)
    output_contract_status = "pass" if stitched_path is not None and stitched_path.exists() else "missing_stitched_equity"

    final_equity_ratio = _as_float(summary.get("final_equity_ratio"))
    raw_pass = _as_bool(summary.get("raw_outperformance_pass")) or (
        pd.notna(final_equity_ratio) and final_equity_ratio > 1.0
    )
    date_alignment = _date_alignment(summary, stitched_path)
    position_audit = _position_shift_audit(stitched_path) if raw_pass else "not_audited_raw_fail"
    cost_audit = _transaction_cost_audit(stitched_path, transaction_cost_bps) if raw_pass else "not_audited_raw_fail"

    audit_notes: List[str] = []
    status = "ok" if output_contract_status == "pass" else output_contract_status
    if output_contract_status != "pass":
        audit_notes.append(f"audit_fail: {output_contract_status}")
    if raw_pass and date_alignment != "pass":
        audit_notes.append("audit_fail: benchmark date alignment")
    if raw_pass and position_audit not in {"shifted_execution_model", "position_columns_present_review_code_for_shift"}:
        audit_notes.append("audit_fail: shifted position evidence")
    if raw_pass and transaction_cost_bps > 0 and cost_audit != "costs_present":
        audit_notes.append("audit_fail: transaction cost evidence")
    if raw_pass and one_year_drives:
        audit_notes.append("audit_fail: one year drives excess return")

    classification = _classify(final_equity_ratio, raw_pass, status, audit_notes)
    start_date = str(summary.get("start_date", ""))
    end_date = str(summary.get("end_date", ""))
    notes = "; ".join(audit_notes)
    if not notes:
        notes = "standard summary found"

    return {
        "strategy_config": spec.strategy_config,
        "config_name": _config_name(spec, config),
        "strategy_name": _strategy_name(spec, config),
        "config_path": str(spec.config_path or ""),
        "output_dir": str(output_dir),
        "status": status,
        "classification": classification,
        "date_range": f"{start_date} to {end_date}",
        "start_date": start_date,
        "end_date": end_date,
        "stitched_rows": _as_int(summary.get("stitched_rows")),
        "strategy_final_equity": _as_float(summary.get("strategy_final_equity")),
        "benchmark_final_equity": _as_float(summary.get("benchmark_final_equity")),
        "final_equity_ratio": final_equity_ratio,
        "excess_cagr": _as_float(summary.get("excess_cagr")),
        "strategy_max_dd": _as_float(summary.get("strategy_max_dd")),
        "benchmark_max_dd": _as_float(summary.get("benchmark_max_dd")),
        "yearly_wins": yearly_wins,
        "total_years": total_years,
        "raw_outperformance_pass": raw_pass,
        "objective_used": objective,
        "transaction_cost_bps": transaction_cost_bps,
        "benchmark_date_alignment": date_alignment,
        "position_shift_audit": position_audit,
        "transaction_cost_audit": cost_audit,
        "output_contract_status": output_contract_status,
        "dominant_excess_year": dominant_year,
        "dominant_excess_year_share": dominant_share,
        "one_year_drives_excess_return": one_year_drives,
        "notes": notes,
    }


def _write_report(summary: pd.DataFrame, output_dir: Path) -> Path:
    report_path = output_dir / "controlled_run_report.md"
    candidates = summary[summary["classification"].eq("candidate for tournament_gate validation")]
    failures = summary[summary["classification"].eq("infrastructure failure")]
    raw_rejects = summary[summary["classification"].eq("empirical rejection")]
    interesting = summary[summary["classification"].eq("interesting but insufficient")]

    lines = [
        "# Controlled Real-Data Run Summary",
        "",
        "## Objective",
        "Summarize controlled real-data strategy runs and identify whether any cleared the first raw wealth gate versus same-period TQQQ.",
        "",
        "## Primary Gate",
        "`final_equity_ratio > 1.0` versus same-period TQQQ is required before full validation is warranted.",
        "",
        "## Verdict Table",
        _markdown_table(
            summary,
            columns=[
                "strategy_config",
                "config_name",
                "strategy_name",
                "classification",
                "date_range",
                "strategy_final_equity",
                "benchmark_final_equity",
                "final_equity_ratio",
                "excess_cagr",
                "strategy_max_dd",
                "benchmark_max_dd",
                "yearly_wins",
                "total_years",
                "raw_outperformance_pass",
                "output_contract_status",
                "objective_used",
                "transaction_cost_bps",
                "notes",
            ],
            max_rows=50,
        ),
        "",
        "## Candidate Audit",
    ]

    if candidates.empty:
        lines.extend(
            [
                "No completed controlled run cleared `final_equity_ratio > 1.0`; candidate-level audit was not triggered.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                _markdown_table(
                    candidates,
                    columns=[
                        "strategy_config",
                        "config_name",
                        "strategy_name",
                        "benchmark_date_alignment",
                        "position_shift_audit",
                        "transaction_cost_audit",
                        "output_contract_status",
                        "objective_used",
                        "dominant_excess_year",
                        "dominant_excess_year_share",
                        "one_year_drives_excess_return",
                    ],
                    max_rows=50,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Classification Counts",
            _markdown_table(
                summary["classification"].value_counts().rename_axis("classification").reset_index(name="count"),
                max_rows=20,
            ),
            "",
            "## Raw Rejections",
            _markdown_table(
                raw_rejects,
                columns=[
                    "strategy_config",
                    "config_name",
                    "strategy_name",
                    "final_equity_ratio",
                    "excess_cagr",
                    "strategy_max_dd",
                    "benchmark_max_dd",
                    "output_contract_status",
                ],
                max_rows=50,
            ),
            "",
            "## Interesting But Insufficient",
            _markdown_table(
                interesting,
                columns=["strategy_config", "config_name", "strategy_name", "final_equity_ratio", "output_contract_status", "notes"],
                max_rows=50,
            ),
            "",
            "## Infrastructure Failures",
            _markdown_table(
                failures,
                columns=["strategy_config", "config_name", "strategy_name", "output_dir", "output_contract_status", "notes"],
                max_rows=50,
            ),
            "",
            "## Limitations",
            "This report is a summary of saved backtest artifacts only. It does not make investment recommendations and does not replace full validation, cost sensitivity, anti-overfit checks, or code review.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def summarize_controlled_runs(
    *,
    output_dir: Path = Path("outputs/controlled_runs"),
    run_specs: Iterable[ControlledRunSpec] = DEFAULT_CONTROLLED_RUNS,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_summarize_one(spec) for spec in run_specs]
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary.to_csv(output_dir / "controlled_run_summary.csv", index=False)
    _write_report(summary, output_dir)
    print(summary.to_string(index=False))
    print(f"\nControlled run summary written to: {output_dir / 'controlled_run_summary.csv'}")
    print(f"Controlled run report written to: {output_dir / 'controlled_run_report.md'}")
    return summary
