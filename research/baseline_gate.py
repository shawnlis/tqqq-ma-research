from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .reports import _markdown_table


BASELINE_GATE_ENV = "RESEARCH_BASELINE_GATE_FILE"
DEFAULT_BASELINE_REGRESSION_DIR = Path("outputs") / "baseline_regression"
DEFAULT_BASELINE_GATE_FILE = DEFAULT_BASELINE_REGRESSION_DIR / "BASELINE_GATE_FAILED.txt"


def baseline_gate_file() -> Path:
    override = os.environ.get(BASELINE_GATE_ENV)
    return Path(override) if override else DEFAULT_BASELINE_GATE_FILE


def baseline_gate_failed() -> bool:
    return baseline_gate_file().exists()


def enforce_baseline_gate(*, accept_baseline_regression: bool = False, action: str = "run") -> None:
    gate = baseline_gate_file()
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
    rows = [
        {
            "blocker": "baseline_regression_gate",
            "status": "failed" if gate.exists() else "ok",
            "path": str(gate),
            "details": "Baseline equivalence has not been accepted." if gate.exists() else "",
        }
    ]
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    return summary


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
