from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


CHECK_COLUMNS = ("ret", "equity", "position", "tqqq_weight", "qqq_weight", "turnover")


def _read_run_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"CSV file is empty: {path}")

    date_column = None
    for candidate in ("Date", "date", "Datetime", "datetime"):
        if candidate in raw.columns:
            date_column = candidate
            break
    if date_column is None:
        date_column = raw.columns[0]

    out = raw.copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce")
    if out[date_column].isna().any():
        raise ValueError(f"Could not parse date index from column {date_column}: {path}")
    out = out.set_index(date_column).sort_index()
    out.index.name = "Date"
    return out


def _nan_to_text(value: Any) -> Any:
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return value


def _row(
    check: str,
    passed: bool,
    *,
    old_value: Any = "",
    new_value: Any = "",
    max_abs_diff: Any = "",
    tolerance: float,
    details: str = "",
    failure_type: str = "",
) -> Dict[str, Any]:
    return {
        "check": check,
        "passed": bool(passed),
        "failure_type": "" if passed else failure_type,
        "old_value": _nan_to_text(old_value),
        "new_value": _nan_to_text(new_value),
        "max_abs_diff": _nan_to_text(max_abs_diff),
        "tolerance": float(tolerance),
        "details": details,
    }


def _max_abs_diff(
    old: pd.DataFrame,
    new: pd.DataFrame,
    column: str,
    common_index: pd.Index,
) -> float:
    if column not in old.columns or column not in new.columns or len(common_index) == 0:
        return np.nan
    old_values = pd.to_numeric(old.loc[common_index, column], errors="coerce")
    new_values = pd.to_numeric(new.loc[common_index, column], errors="coerce")
    diff = (old_values - new_values).abs().replace([np.inf, -np.inf], np.nan)
    return float(diff.max()) if not diff.dropna().empty else np.nan


def _compare_numeric_column(
    rows: List[Dict[str, Any]],
    *,
    old: pd.DataFrame,
    new: pd.DataFrame,
    column: str,
    common_index: pd.Index,
    tolerance: float,
) -> None:
    old_has = column in old.columns
    new_has = column in new.columns
    if not old_has and not new_has:
        if column in {"position", "tqqq_weight", "qqq_weight"}:
            return
        rows.append(
            _row(
                column,
                False,
                tolerance=tolerance,
                details=f"{column} missing from both files",
                failure_type="schema_mismatch",
            )
        )
        return
    if old_has != new_has:
        rows.append(
            _row(
                column,
                False,
                tolerance=tolerance,
                details=f"{column} present in only one file",
                failure_type="schema_mismatch",
            )
        )
        return

    diff = _max_abs_diff(old, new, column, common_index)
    passed = bool(pd.notna(diff) and diff <= float(tolerance))
    rows.append(
        _row(
            column,
            passed,
            max_abs_diff=diff,
            tolerance=tolerance,
            details="" if passed else f"{column} differs beyond tolerance",
            failure_type="numeric_mismatch",
        )
    )


def _date_difference_details(old_index: pd.Index, new_index: pd.Index) -> str:
    old_only = old_index.difference(new_index)
    new_only = new_index.difference(old_index)
    parts = []
    if len(old_only) > 0:
        parts.append(f"old_only_dates={len(old_only)} first={old_only[0].date()}")
    if len(new_only) > 0:
        parts.append(f"new_only_dates={len(new_only)} first={new_only[0].date()}")
    if not parts and not old_index.equals(new_index):
        parts.append("same dates but different ordering")
    return "; ".join(parts)


def compare_run_files(
    old_csv: Path,
    new_csv: Path,
    *,
    tolerance: float = 1e-8,
    output_dir: Path = Path("outputs/comparison"),
) -> Tuple[pd.DataFrame, bool]:
    old = _read_run_csv(old_csv)
    new = _read_run_csv(new_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    common_index = old.index.intersection(new.index)
    common_columns = sorted(set(old.columns).intersection(set(new.columns)))

    rows.append(
        _row(
            "date_index",
            old.index.equals(new.index),
            old_value=f"{old.index.min().date()} to {old.index.max().date()}" if len(old.index) else "",
            new_value=f"{new.index.min().date()} to {new.index.max().date()}" if len(new.index) else "",
            tolerance=tolerance,
            details=_date_difference_details(old.index, new.index),
            failure_type="date_range_mismatch",
        )
    )
    rows.append(
        _row(
            "row_count",
            len(old) == len(new),
            old_value=len(old),
            new_value=len(new),
            tolerance=tolerance,
            details="" if len(old) == len(new) else "row counts differ",
            failure_type="date_range_mismatch",
        )
    )
    rows.append(
        _row(
            "common_columns",
            len(common_columns) > 0,
            old_value=len(old.columns),
            new_value=len(new.columns),
            tolerance=tolerance,
            details=";".join(common_columns),
            failure_type="schema_mismatch",
        )
    )

    for column in CHECK_COLUMNS:
        _compare_numeric_column(
            rows,
            old=old,
            new=new,
            column=column,
            common_index=common_index,
            tolerance=tolerance,
        )

    if "equity" in old.columns and "equity" in new.columns and len(common_index) > 0:
        old_final = float(pd.to_numeric(old.loc[common_index, "equity"], errors="coerce").iloc[-1])
        new_final = float(pd.to_numeric(new.loc[common_index, "equity"], errors="coerce").iloc[-1])
        final_diff = abs(old_final - new_final)
        rows.append(
            _row(
                "final_equity",
                final_diff <= float(tolerance),
                old_value=old_final,
                new_value=new_final,
                max_abs_diff=final_diff,
                tolerance=tolerance,
                details="" if final_diff <= float(tolerance) else "final equity differs beyond tolerance",
                failure_type="numeric_mismatch",
            )
        )
    else:
        rows.append(
            _row(
                "final_equity",
                False,
                tolerance=tolerance,
                details="equity missing or no overlapping dates",
                failure_type="schema_mismatch",
            )
        )

    daily_ret_diff = _max_abs_diff(old, new, "ret", common_index)
    rows.append(
        _row(
            "max_absolute_daily_return_difference",
            bool(pd.notna(daily_ret_diff) and daily_ret_diff <= float(tolerance)),
            max_abs_diff=daily_ret_diff,
            tolerance=tolerance,
            details="" if pd.notna(daily_ret_diff) else "ret missing or no overlapping dates",
            failure_type="numeric_mismatch" if pd.notna(daily_ret_diff) else "schema_mismatch",
        )
    )
    equity_diff = _max_abs_diff(old, new, "equity", common_index)
    rows.append(
        _row(
            "max_absolute_equity_difference",
            bool(pd.notna(equity_diff) and equity_diff <= float(tolerance)),
            max_abs_diff=equity_diff,
            tolerance=tolerance,
            details="" if pd.notna(equity_diff) else "equity missing or no overlapping dates",
            failure_type="numeric_mismatch" if pd.notna(equity_diff) else "schema_mismatch",
        )
    )

    overall_pass = all(row["passed"] for row in rows)
    rows.insert(
        0,
        _row(
            "overall",
            overall_pass,
            old_value=str(old_csv),
            new_value=str(new_csv),
            tolerance=tolerance,
            details="all checks passed" if overall_pass else "one or more checks failed",
            failure_type="overall_failure",
        ),
    )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "compare_run_summary.csv", index=False)
    _write_report(
        output_dir=output_dir,
        summary=summary,
        old_csv=Path(old_csv),
        new_csv=Path(new_csv),
        tolerance=tolerance,
        overall_pass=overall_pass,
    )
    print(summary.to_string(index=False))
    print(f"\nComparison report written to: {output_dir / 'compare_run_report.md'}")
    return summary, overall_pass


def _markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No data available._"
    table = df.head(max_rows).copy()
    headers = [str(column).replace("|", "\\|") for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for column in table.columns:
            value = row[column]
            if isinstance(value, float):
                text = f"{value:.12g}"
            else:
                text = "" if pd.isna(value) else str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    old_csv: Path,
    new_csv: Path,
    tolerance: float,
    overall_pass: bool,
) -> Path:
    report_path = output_dir / "compare_run_report.md"
    failed = summary[~summary["passed"].astype(bool)].copy()
    diagnosis = (
        failed["failure_type"].replace("", np.nan).dropna().value_counts().rename_axis("failure_type").reset_index(name="failed_checks")
        if "failure_type" in failed.columns and not failed.empty
        else pd.DataFrame(columns=["failure_type", "failed_checks"])
    )
    lines = [
        "# Baseline Run Comparison",
        "",
        f"- Old CSV: `{old_csv}`",
        f"- New CSV: `{new_csv}`",
        f"- Tolerance: `{tolerance:g}`",
        f"- Overall pass: `{'yes' if overall_pass else 'no'}`",
        "",
        "## Summary",
        _markdown_table(summary),
        "",
        "## Failed Checks",
        _markdown_table(failed),
        "",
        "## Diagnosis",
        _markdown_table(diagnosis),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
