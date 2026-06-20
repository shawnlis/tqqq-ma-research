from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .reports import _markdown_table


BASELINE_GATE_ENV = "RESEARCH_BASELINE_GATE_FILE"
DEFAULT_BASELINE_REGRESSION_DIR = Path("outputs") / "baseline_regression"
DEFAULT_BASELINE_GATE_FILE = DEFAULT_BASELINE_REGRESSION_DIR / "BASELINE_GATE_FAILED.txt"
ACCEPTANCE_FILENAME = "BASELINE_ACCEPTED_DATA_VINTAGE.txt"
ACCEPTANCE_SUMMARY_FILENAME = "baseline_acceptance_summary.csv"
ACCEPTANCE_REPORT_FILENAME = "baseline_acceptance_report.md"
FLOATING_NOISE_TOLERANCE = 1e-12


def baseline_gate_file() -> Path:
    override = os.environ.get(BASELINE_GATE_ENV)
    return Path(override) if override else DEFAULT_BASELINE_GATE_FILE


def baseline_regression_dir() -> Path:
    return baseline_gate_file().parent


def baseline_acceptance_file() -> Path:
    return baseline_regression_dir() / ACCEPTANCE_FILENAME


def baseline_acceptance_summary_file() -> Path:
    return baseline_regression_dir() / ACCEPTANCE_SUMMARY_FILENAME


def baseline_gate_failed() -> bool:
    return baseline_gate_file().exists()


def baseline_accepted() -> bool:
    return baseline_acceptance_file().exists()


def _acceptance_warning_text() -> str:
    summary = _read_csv(baseline_acceptance_summary_file())
    if summary.empty:
        return "v10 baseline regression accepted as data-vintage mismatch; exact numerical equivalence was not achieved."
    row = summary.iloc[0]
    reason = _text_value(row.get("reason", "data-vintage mismatch"))
    drift = _float_value(row.get("final_equity_relative_diff"))
    drift_text = f"{drift:.12g}" if pd.notna(drift) else "unknown"
    return (
        "v10 baseline regression accepted as data-vintage mismatch; "
        f"reason={reason}; final_equity_relative_diff={drift_text}; "
        "exact numerical equivalence was not achieved."
    )


def enforce_baseline_gate(*, accept_baseline_regression: bool = False, action: str = "run") -> None:
    gate = baseline_gate_file()
    acceptance = baseline_acceptance_file()
    if acceptance.exists():
        print(f"WARNING: {_acceptance_warning_text()} Proceeding with {action}.", file=sys.stderr)
        return
    if not gate.exists():
        return
    if accept_baseline_regression:
        print(
            f"WARNING: baseline regression gate is failed at {gate}; proceeding with explicit acceptance for {action}.",
            file=sys.stderr,
        )
        return
    raise RuntimeError(
        f"Baseline regression gate is failed at {gate}. "
        f"Resolve or formally accept the baseline regression before {action}. "
        "Pass --accept-baseline-regression to proceed intentionally."
    )


def check_blockers() -> pd.DataFrame:
    gate = baseline_gate_file()
    acceptance = baseline_acceptance_file()
    accepted = acceptance.exists()
    summary = _read_csv(baseline_acceptance_summary_file())
    reason = ""
    rel_diff: Any = ""
    if not summary.empty:
        row = summary.iloc[0]
        reason = _text_value(row.get("reason"))
        rel_diff = _float_value(row.get("final_equity_relative_diff"))
    rows = [
        {
            "blocker": "baseline_regression_gate",
            "status": "warning" if accepted else "failed" if gate.exists() else "ok",
            "path": str(acceptance if accepted else gate),
            "details": (
                f"Accepted data-vintage baseline regression. reason={reason}; "
                f"final_equity_relative_diff={rel_diff}"
                if accepted
                else "Baseline equivalence has not been accepted."
                if gate.exists()
                else ""
            ),
        }
    ]
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def baseline_acceptance_warning() -> str:
    return _acceptance_warning_text() if baseline_accepted() else ""


def _summary_value(df: pd.DataFrame, column: str, default: Any = np.nan) -> Any:
    if df.empty or column not in df.columns:
        return default
    return df[column].iloc[0]


def _summary_bool(df: pd.DataFrame, column: str) -> bool:
    return _bool_value(_summary_value(df, column, False))


def _nonnegative_small(value: Any, tolerance: float) -> bool:
    numeric = _float_value(value)
    return bool(pd.notna(numeric) and abs(numeric) <= float(tolerance))


def _positive_numeric(value: Any) -> bool:
    numeric = _float_value(value)
    return bool(pd.notna(numeric) and abs(numeric) > 0.0)


def _validate_acceptance_inputs(
    *,
    replay: pd.DataFrame,
    data_audit: pd.DataFrame,
    allocation: pd.DataFrame,
    baseline: pd.DataFrame,
    max_final_equity_rel_diff: float,
    require_weight_match: bool,
    weight_tolerance: float,
) -> Tuple[Dict[str, Any], list[str]]:
    failures: list[str] = []
    old_final = _float_value(_summary_value(replay, "old_final_equity"))
    replay_final = _float_value(_summary_value(replay, "replay_final_equity"))
    final_rel_diff = (
        abs(replay_final - old_final) / abs(old_final)
        if pd.notna(old_final) and pd.notna(replay_final) and old_final != 0.0
        else np.nan
    )

    date_range_matches = _summary_bool(baseline, "date_range_now_matches")
    row_count_matches = _summary_bool(baseline, "row_count_now_matches")
    params_replayed = _summary_bool(baseline, "old_selected_params_replayed_exactly")
    if not date_range_matches:
        failures.append("date range does not match")
    if not row_count_matches:
        failures.append("row count does not match")
    if not params_replayed:
        failures.append("old selected params were not replayed exactly")
    if not pd.notna(final_rel_diff) or final_rel_diff > float(max_final_equity_rel_diff):
        failures.append(
            f"final equity relative drift {final_rel_diff} exceeds threshold {max_final_equity_rel_diff}"
        )

    weight_columns = {
        "max_abs_tqqq_weight_diff": _summary_value(allocation, "max_abs_tqqq_weight_diff"),
        "max_abs_qqq_weight_diff": _summary_value(allocation, "max_abs_qqq_weight_diff"),
        "max_abs_position_diff": _summary_value(allocation, "max_abs_position_diff"),
        "max_abs_turnover_diff": _summary_value(allocation, "max_abs_turnover_diff"),
    }
    if require_weight_match:
        for column, value in weight_columns.items():
            if not _nonnegative_small(value, weight_tolerance):
                failures.append(f"{column} is not within floating-noise tolerance: {value}")

    data_match = _text_value(_summary_value(data_audit, "current_data_matches_old_baseline_returns", "")).lower()
    data_has_return_drift = _positive_numeric(_summary_value(data_audit, "max_abs_tqqq_return_diff")) or _positive_numeric(
        _summary_value(data_audit, "max_abs_qqq_return_diff")
    )
    if data_match != "no" or not data_has_return_drift:
        failures.append("baseline data audit does not prove price/return data mismatch")

    allocation_source = _text_value(_summary_value(allocation, "likely_source", "")).lower()
    allocation_data_mismatch = "data mismatch" in allocation_source or "return composition mismatch" in allocation_source
    allocation_bad_tokens = (
        "pre-shift signal mismatch",
        "shifted weight mismatch",
        "wrong old/new variant likely",
        "turnover/cost mismatch",
    )
    if not allocation_data_mismatch:
        failures.append("regime allocation audit does not classify source as data/return-composition mismatch")
    for token in allocation_bad_tokens:
        if token in allocation_source:
            failures.append(f"regime allocation audit includes disallowed source: {token}")

    metrics: Dict[str, Any] = {
        "accepted": len(failures) == 0,
        "date_range_matches": date_range_matches,
        "row_count_matches": row_count_matches,
        "old_selected_params_replayed_exactly": params_replayed,
        "old_final_equity": old_final,
        "replay_final_equity": replay_final,
        "final_equity_relative_diff": final_rel_diff,
        "max_final_equity_relative_diff_allowed": float(max_final_equity_rel_diff),
        "require_weight_match": bool(require_weight_match),
        "weight_tolerance": float(weight_tolerance),
        **weight_columns,
        "baseline_data_return_match_status": data_match,
        "max_abs_tqqq_return_diff": _float_value(_summary_value(data_audit, "max_abs_tqqq_return_diff")),
        "max_abs_qqq_return_diff": _float_value(_summary_value(data_audit, "max_abs_qqq_return_diff")),
        "allocation_likely_source": _text_value(_summary_value(allocation, "likely_source", "")),
        "regression_classification": _text_value(_summary_value(baseline, "regression_classification", "")),
        "likely_root_cause": _text_value(_summary_value(baseline, "likely_root_cause", "")),
    }
    return metrics, failures


def accept_baseline_regression(
    *,
    reason: str,
    max_final_equity_rel_diff: float,
    require_weight_match: bool = True,
    output_dir: Optional[Path] = None,
    replay_summary_path: Path = Path("outputs") / "replay" / "v10_regime_windows" / "replay_compare_summary.csv",
    baseline_data_summary_path: Path = Path("outputs") / "comparison" / "baseline_data_audit" / "baseline_data_audit_summary.csv",
    allocation_summary_path: Path = Path("outputs") / "comparison" / "regime_allocation_audit" / "regime_allocation_audit_summary.csv",
    baseline_regression_summary_path: Path = Path("outputs") / "baseline_regression" / "baseline_regression_summary.csv",
    weight_tolerance: float = FLOATING_NOISE_TOLERANCE,
) -> pd.DataFrame:
    target_dir = Path(output_dir) if output_dir is not None else baseline_regression_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    replay = _read_csv(replay_summary_path)
    data_audit = _read_csv(baseline_data_summary_path)
    allocation = _read_csv(allocation_summary_path)
    baseline = _read_csv(baseline_regression_summary_path)
    missing = [
        str(path)
        for path, df in (
            (replay_summary_path, replay),
            (baseline_data_summary_path, data_audit),
            (allocation_summary_path, allocation),
            (baseline_regression_summary_path, baseline),
        )
        if df.empty
    ]
    if missing:
        raise ValueError(f"Missing or empty baseline diagnostic summaries: {missing}")

    metrics, failures = _validate_acceptance_inputs(
        replay=replay,
        data_audit=data_audit,
        allocation=allocation,
        baseline=baseline,
        max_final_equity_rel_diff=max_final_equity_rel_diff,
        require_weight_match=require_weight_match,
        weight_tolerance=weight_tolerance,
    )
    if failures:
        failure_summary = pd.DataFrame([{**metrics, "reason": reason, "failure_reasons": "; ".join(failures)}])
        failure_summary.to_csv(target_dir / ACCEPTANCE_SUMMARY_FILENAME, index=False)
        _write_baseline_acceptance_report(target_dir, failure_summary, accepted=False)
        raise ValueError("Baseline regression acceptance failed: " + "; ".join(failures))

    timestamp = datetime.now(timezone.utc).isoformat()
    gate = baseline_gate_file()
    archived_gate_path = ""
    if gate.exists():
        archive_dir = target_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_gate = archive_dir / "BASELINE_GATE_FAILED_superseded.txt"
        archived_gate.write_text(gate.read_text(encoding="utf-8"), encoding="utf-8")
        gate.unlink()
        archived_gate_path = str(archived_gate)

    acceptance = {
        **metrics,
        "accepted": True,
        "accepted_at": timestamp,
        "reason": reason,
        "archived_gate_path": archived_gate_path,
        "acceptance_file": str(target_dir / ACCEPTANCE_FILENAME),
        "replay_summary_path": str(replay_summary_path),
        "baseline_data_summary_path": str(baseline_data_summary_path),
        "allocation_summary_path": str(allocation_summary_path),
        "baseline_regression_summary_path": str(baseline_regression_summary_path),
    }
    summary = pd.DataFrame([acceptance])
    summary.to_csv(target_dir / ACCEPTANCE_SUMMARY_FILENAME, index=False)
    _write_baseline_acceptance_report(target_dir, summary, accepted=True)
    (target_dir / ACCEPTANCE_FILENAME).write_text(
        "v10 baseline regression accepted as a data-vintage mismatch.\n"
        f"Accepted at: {timestamp}\n"
        f"Reason: {reason}\n"
        f"Final equity relative drift: {metrics['final_equity_relative_diff']:.12g}\n"
        "Allocation replay weights/positions/turnover matched within tolerance; exact numerical equivalence was not achieved.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nBaseline acceptance report written to: {target_dir / ACCEPTANCE_REPORT_FILENAME}")
    return summary


def _write_baseline_acceptance_report(output_dir: Path, summary: pd.DataFrame, *, accepted: bool) -> Path:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    lines = [
        "# Baseline Regression Acceptance",
        "",
        f"- Accepted: `{bool(accepted)}`",
        f"- Reason: {row.get('reason', '')}",
        f"- Final-equity relative drift: `{row.get('final_equity_relative_diff', '')}`",
        f"- Max allowed final-equity relative drift: `{row.get('max_final_equity_relative_diff_allowed', '')}`",
        f"- Allocation source: `{row.get('allocation_likely_source', '')}`",
        f"- Baseline data status: `{row.get('baseline_data_return_match_status', '')}`",
        "",
        "## Weight/Allocation Checks",
        f"- TQQQ weight max diff: `{row.get('max_abs_tqqq_weight_diff', '')}`",
        f"- QQQ weight max diff: `{row.get('max_abs_qqq_weight_diff', '')}`",
        f"- Position max diff: `{row.get('max_abs_position_diff', '')}`",
        f"- Turnover max diff: `{row.get('max_abs_turnover_diff', '')}`",
        "",
        "## Summary",
        _markdown_table(summary, max_rows=5),
        "",
    ]
    if not accepted:
        lines.extend(
            [
                "## Failure Reasons",
                str(row.get("failure_reasons", "")),
                "",
            ]
        )
    path = output_dir / ACCEPTANCE_REPORT_FILENAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _first_row(df: pd.DataFrame) -> pd.Series:
    return df.iloc[0] if not df.empty else pd.Series(dtype=object)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _float_value(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _text_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _compare_check(compare: pd.DataFrame, check: str) -> Optional[bool]:
    if compare.empty or "check" not in compare.columns or "passed" not in compare.columns:
        return None
    rows = compare[compare["check"].astype(str) == check]
    if rows.empty:
        return None
    return _bool_value(rows["passed"].iloc[0])


def _compare_final_value(compare: pd.DataFrame, column: str) -> float:
    if compare.empty or "check" not in compare.columns:
        return np.nan
    rows = compare[compare["check"].astype(str) == "final_equity"]
    if rows.empty or column not in rows.columns:
        return np.nan
    return _float_value(rows[column].iloc[0])


def _preferred_compare_path(path: Path) -> Path:
    explicit = Path(path)
    if explicit.exists():
        return explicit
    clipped = Path("outputs") / "comparison" / "baseline_clipped" / "compare_run_summary.csv"
    if clipped.exists():
        return clipped
    return explicit


def _params_replayed_exactly(old_windows_path: Path, replay_params_path: Path) -> Tuple[bool, str]:
    old = _read_csv(old_windows_path)
    replay = _read_csv(replay_params_path)
    if old.empty or replay.empty:
        return False, "missing old windows or replay parameter artifact"
    if len(old) != len(replay):
        return False, f"window row count differs: old={len(old)} replay={len(replay)}"

    param_columns = [
        "trend_window",
        "momentum_window",
        "vol_window",
        "trend_on",
        "trend_off",
        "mom_on",
        "mom_off",
        "vol_cap",
        "risk_on_leverage",
        "risk_off_qqq_position",
        "transition_position",
        "min_hold_days",
        "cooldown_days",
    ]
    for column in param_columns:
        old_values = old[column] if column in old.columns else pd.Series([0] * len(old))
        replay_values = replay[column] if column in replay.columns else pd.Series([np.nan] * len(replay))
        old_numeric = pd.to_numeric(old_values, errors="coerce").fillna(0.0)
        replay_numeric = pd.to_numeric(replay_values, errors="coerce").fillna(0.0)
        if not np.allclose(old_numeric.to_numpy(), replay_numeric.to_numpy(), atol=1e-12, equal_nan=True):
            return False, f"selected parameter mismatch in {column}"
    return True, "old selected window parameters match replay artifact"


def _classify_regression(row: Dict[str, Any]) -> str:
    if bool(row.get("baseline_equivalence_passed", False)):
        return "passed"
    if not bool(row.get("date_range_now_matches", False)) or not bool(row.get("row_count_now_matches", False)):
        return "wrong baseline comparison"
    if not bool(row.get("old_selected_params_replayed_exactly", False)):
        return "unresolved"
    if str(row.get("price_return_data_matches_old_baseline", "")).lower() == "no":
        return "wrong baseline comparison"
    likely = str(row.get("allocation_likely_source", "")).lower()
    if "pre-shift signal mismatch" in likely or "shifted weight mismatch" in likely:
        return "accidental regression"
    if "turnover/cost mismatch" in likely or "return composition mismatch" in likely or "equity compounding mismatch" in likely:
        return "accidental regression"
    return "unresolved"


def _root_cause(row: Dict[str, Any]) -> str:
    allocation_source = _text_value(row.get("allocation_likely_source", ""))
    price_match = str(row.get("price_return_data_matches_old_baseline", "")).lower()
    if price_match == "no" and "data mismatch" in allocation_source:
        return (
            "Current cached TQQQ/QQQ daily returns differ from the v10 stitched raw returns; "
            "exact equivalence cannot be claimed without the old price cache or formal acceptance."
        )
    if not bool(row.get("old_selected_params_replayed_exactly", False)):
        return "Replay did not prove exact reuse of old selected window parameters."
    if "shifted weight mismatch" in allocation_source:
        return "Allocation weights differ after signal shifting, suggesting wrong variant or allocation logic."
    if allocation_source:
        return allocation_source
    return "Unresolved baseline regression."


def create_baseline_regression_report(
    *,
    output_dir: Path = DEFAULT_BASELINE_REGRESSION_DIR,
    compare_summary_path: Path = Path("outputs") / "comparison" / "baseline_clipped" / "compare_run_summary.csv",
    replay_summary_path: Path = Path("outputs") / "replay" / "v10_regime_windows" / "replay_compare_summary.csv",
    replay_params_path: Path = Path("outputs") / "replay" / "v10_regime_windows" / "window_param_replay.csv",
    old_windows_path: Path = Path("ma_search_output_v10") / "regime_walk_forward_windows.csv",
    baseline_data_summary_path: Path = Path("outputs") / "comparison" / "baseline_data_audit" / "baseline_data_audit_summary.csv",
    allocation_summary_path: Path = Path("outputs") / "comparison" / "regime_allocation_audit" / "regime_allocation_audit_summary.csv",
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_summary_path = _preferred_compare_path(compare_summary_path)
    compare = _read_csv(compare_summary_path)
    replay = _read_csv(replay_summary_path)
    data_audit = _read_csv(baseline_data_summary_path)
    allocation = _read_csv(allocation_summary_path)
    replay_row = _first_row(replay)
    data_row = _first_row(data_audit)
    allocation_row = _first_row(allocation)

    params_exact, params_details = _params_replayed_exactly(old_windows_path, replay_params_path)
    date_match = _compare_check(compare, "date_index")
    row_match = _compare_check(compare, "row_count")
    if date_match is None:
        date_match = (
            _text_value(replay_row.get("old_start")) == _text_value(replay_row.get("replay_start"))
            and _text_value(replay_row.get("old_end")) == _text_value(replay_row.get("replay_end"))
        )
    if row_match is None:
        row_match = (
            _float_value(replay_row.get("old_rows")) == _float_value(replay_row.get("replay_rows"))
            == _float_value(replay_row.get("common_rows"))
        )

    price_match = _text_value(data_row.get("current_data_matches_old_baseline_returns", "missing"))
    replay_passed = _bool_value(replay_row.get("passed", False))
    compare_passed = _compare_check(compare, "overall")
    if compare_passed is None:
        compare_passed = False
    data_passed = price_match.lower() == "yes"
    baseline_passed = bool(compare_passed and replay_passed and data_passed and params_exact)

    old_final = _float_value(replay_row.get("old_final_equity"))
    replay_final = _float_value(replay_row.get("replay_final_equity"))
    new_final = _compare_final_value(compare, "new_value")
    if pd.isna(new_final):
        new_final = replay_final

    row: Dict[str, Any] = {
        "baseline_equivalence_passed": baseline_passed,
        "date_range_now_matches": bool(date_match),
        "row_count_now_matches": bool(row_match),
        "old_selected_params_replayed_exactly": bool(params_exact),
        "old_selected_params_replay_details": params_details,
        "price_return_data_matches_old_baseline": price_match,
        "first_signal_divergence_date": _text_value(allocation_row.get("first_raw_signal_divergence_date")),
        "first_weight_divergence_date": _text_value(allocation_row.get("first_weight_divergence_date")),
        "first_return_divergence_date": _text_value(allocation_row.get("first_return_divergence_date")),
        "first_equity_divergence_date": _text_value(allocation_row.get("first_equity_divergence_date")),
        "old_final_equity": old_final,
        "new_final_equity": new_final,
        "replay_final_equity": replay_final,
        "replay_final_equity_diff": _float_value(replay_row.get("final_equity_diff")),
        "max_abs_replay_ret_diff": _float_value(replay_row.get("max_abs_ret_diff")),
        "max_abs_allocation_tqqq_weight_diff": _float_value(allocation_row.get("max_abs_tqqq_weight_diff")),
        "max_abs_allocation_qqq_weight_diff": _float_value(allocation_row.get("max_abs_qqq_weight_diff")),
        "max_abs_allocation_position_diff": _float_value(allocation_row.get("max_abs_position_diff")),
        "max_abs_allocation_turnover_diff": _float_value(allocation_row.get("max_abs_turnover_diff")),
        "max_abs_price_tqqq_return_diff": _float_value(data_row.get("max_abs_tqqq_return_diff")),
        "max_abs_price_qqq_return_diff": _float_value(data_row.get("max_abs_qqq_return_diff")),
        "allocation_likely_source": _text_value(allocation_row.get("likely_source")),
        "compare_summary_path": str(compare_summary_path),
        "replay_summary_path": str(replay_summary_path),
        "baseline_data_summary_path": str(baseline_data_summary_path),
        "allocation_summary_path": str(allocation_summary_path),
        "tolerance": float(tolerance),
    }
    row["likely_root_cause"] = _root_cause(row)
    row["regression_classification"] = _classify_regression(row)
    row["gate_file"] = str(output_dir / "BASELINE_GATE_FAILED.txt")

    summary = pd.DataFrame([row])
    summary.to_csv(output_dir / "baseline_regression_summary.csv", index=False)
    _write_baseline_regression_report(output_dir, summary)
    gate = output_dir / "BASELINE_GATE_FAILED.txt"
    if baseline_passed:
        if gate.exists():
            gate.unlink()
    else:
        gate.write_text(
            "Baseline equivalence against ma_search_output_v10 failed.\n"
            f"Classification: {row['regression_classification']}\n"
            f"Likely root cause: {row['likely_root_cause']}\n"
            "Do not run full downstream research unless this regression is formally accepted.\n",
            encoding="utf-8",
        )

    print(summary.to_string(index=False))
    print(f"\nBaseline regression report written to: {output_dir / 'baseline_regression_report.md'}")
    return summary


def _write_baseline_regression_report(output_dir: Path, summary: pd.DataFrame) -> Path:
    row = summary.iloc[0].to_dict()
    gate_status = "passed" if bool(row.get("baseline_equivalence_passed")) else "failed"
    lines = [
        "# Baseline Regression Report",
        "",
        "This report summarizes baseline compatibility against `ma_search_output_v10`. It is a research gate, not a strategy result.",
        "",
        "## Verdict",
        f"- Baseline equivalence: `{gate_status}`",
        f"- Classification: `{row.get('regression_classification', '')}`",
        f"- Likely root cause: {row.get('likely_root_cause', '')}",
        "",
        "## Required Checks",
        f"- Date range now matches: `{bool(row.get('date_range_now_matches'))}`",
        f"- Row count now matches: `{bool(row.get('row_count_now_matches'))}`",
        f"- Old selected params replayed exactly: `{bool(row.get('old_selected_params_replayed_exactly'))}`",
        f"- Price/return data matches old baseline: `{row.get('price_return_data_matches_old_baseline', '')}`",
        "",
        "## Divergences",
        f"- First signal divergence: `{row.get('first_signal_divergence_date', '')}`",
        f"- First weight divergence: `{row.get('first_weight_divergence_date', '')}`",
        f"- First return divergence: `{row.get('first_return_divergence_date', '')}`",
        f"- First equity divergence: `{row.get('first_equity_divergence_date', '')}`",
        "",
        "## Final Equity",
        f"- Old final equity: `{row.get('old_final_equity', '')}`",
        f"- New final equity from compare-run: `{row.get('new_final_equity', '')}`",
        f"- Replay final equity: `{row.get('replay_final_equity', '')}`",
        "",
        "## Summary CSV View",
        _markdown_table(summary, max_rows=5),
        "",
        "## Source Artifacts",
        f"- Compare summary: `{row.get('compare_summary_path', '')}`",
        f"- Replay summary: `{row.get('replay_summary_path', '')}`",
        f"- Baseline data summary: `{row.get('baseline_data_summary_path', '')}`",
        f"- Regime allocation summary: `{row.get('allocation_summary_path', '')}`",
        "",
    ]
    path = output_dir / "baseline_regression_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
