from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .experiments import _is_vol_target_strategy, _load_price_data, load_yaml_file
from .strategies.ma_strategy import compute_ma
from .strategies.vol_target_strategy import build_vol_target_signal
from .validation import _vol_target_row_to_params


DEFAULT_AUDIT_OUTPUT_DIR = Path("outputs/audits")
SUMMARY_FILENAME = "voltarget_timing_audit_summary.csv"
REPORT_FILENAME = "voltarget_timing_audit_report.md"


def _read_dated_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        first_column = df.columns[0]
        parsed = pd.to_datetime(df[first_column], errors="coerce")
        if parsed.notna().all():
            df = df.drop(columns=[first_column])
            df.index = parsed
        else:
            raise ValueError(f"{path} must contain a Date column or date-like first column.")
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def _read_output_frames(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stitched = _read_dated_csv(output_dir / "stitched_equity.csv")
    windows_path = output_dir / "walk_forward_windows.csv"
    if not windows_path.exists():
        windows_path = output_dir / "parameter_selection_by_window.csv"
    if not windows_path.exists():
        raise FileNotFoundError(f"Required walk-forward windows CSV is missing in {output_dir}")
    windows = pd.read_csv(windows_path)
    return stitched, windows


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _max_abs_diff(left: pd.Series, right: pd.Series) -> float:
    diff = (pd.to_numeric(left, errors="coerce") - pd.to_numeric(right, errors="coerce")).abs()
    diff = diff.replace([np.inf, -np.inf], np.nan).dropna()
    return float(diff.max()) if not diff.empty else 0.0


def _first_diff_date(left: pd.Series, right: pd.Series, tolerance: float) -> str:
    diff = (pd.to_numeric(left, errors="coerce") - pd.to_numeric(right, errors="coerce")).abs()
    diff = diff.replace([np.inf, -np.inf], np.nan)
    failures = diff[diff > tolerance]
    if failures.empty:
        return ""
    return pd.Timestamp(failures.index[0]).date().isoformat()


def _first_true_date(mask: pd.Series) -> str:
    mask = mask.fillna(False).astype(bool)
    if not mask.any():
        return ""
    return pd.Timestamp(mask[mask].index[0]).date().isoformat()


def _trade_weight_column(stitched: pd.DataFrame) -> str:
    for column in ("trade_weight", "tqqq_weight"):
        if column in stitched.columns:
            return column
    return "position"


def _window_dates(row: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(row["test_start"]), pd.Timestamp(row["test_end"])


def _first_warmup_complete_date(
    data: pd.DataFrame,
    row: pd.Series,
    trade_asset: str,
    signal_asset: str,
    filter_asset: str,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    realized_window = int(row["realized_vol_window"])
    trend_window = int(row["trend_window"])
    momentum_window = int(row["momentum_window"])
    trade_returns = data[trade_asset].astype(float).pct_change().fillna(0.0)
    realized_vol = trade_returns.rolling(realized_window, min_periods=realized_window).std(ddof=0)
    trend_ma = compute_ma(data[filter_asset].astype(float), trend_window, "sma")
    momentum = data[signal_asset].astype(float).pct_change(momentum_window)
    complete = pd.DataFrame(
        {
            "realized_vol": realized_vol,
            "trend_ma": trend_ma,
            "momentum": momentum,
        },
        index=data.index,
    ).dropna()
    if complete.empty:
        return None, None
    complete_date = pd.Timestamp(complete.index[0])
    complete_pos = data.index.get_indexer([complete_date])[0]
    if complete_pos + 1 >= len(data.index):
        return complete_date, None
    return complete_date, pd.Timestamp(data.index[complete_pos + 1])


def audit_voltarget_timing_frames(
    *,
    config: Dict[str, Any],
    stitched: pd.DataFrame,
    windows: pd.DataFrame,
    price_data: pd.DataFrame,
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    if not _is_vol_target_strategy(config):
        raise ValueError("audit-voltarget-timing requires a VolTarget config.")
    if stitched.empty:
        raise ValueError("stitched_equity.csv is empty; timing audit cannot run.")
    if windows.empty:
        raise ValueError("walk_forward_windows.csv is empty; timing audit cannot run.")

    stitched = stitched.sort_index().copy()
    data = price_data.sort_index().copy()
    trade_column = _trade_weight_column(stitched)
    transaction_cost_bps = _safe_float(config.get("transaction_cost_bps"), 0.0)

    rows_checked = 0
    windows_checked = 0
    max_position_diff = 0.0
    max_target_diff = 0.0
    max_realized_vol_diff = 0.0
    max_trend_diff = 0.0
    max_momentum_diff = 0.0
    first_position_failure = ""
    first_current_day_leak = ""
    first_target_diff = ""
    first_realized_vol_diff = ""
    first_trend_diff = ""
    first_momentum_diff = ""
    first_stitched_date = stitched.index[0].date().isoformat()
    earliest_warmup_complete: Optional[pd.Timestamp] = None
    earliest_tradable_after_warmup: Optional[pd.Timestamp] = None
    warmup_failures: list[pd.Timestamp] = []

    for _, window in windows.iterrows():
        test_start, test_end = _window_dates(window)
        actual = stitched.loc[(stitched.index >= test_start) & (stitched.index <= test_end)].copy()
        if actual.empty:
            continue

        params = _vol_target_row_to_params(window)
        asset_config = params.asset_config.normalized()
        history = data.loc[:test_end].copy()
        signal = build_vol_target_signal(
            tqqq_close=history[asset_config.trade_asset],
            qqq_close=history[asset_config.primary_signal_asset],
            params=params,
            price_data=history,
            filter_close=history[asset_config.secondary_filter_asset],
        )
        expected_position = signal["target_exposure"].shift(1).fillna(0.0)
        actual_index = actual.index.intersection(signal.index)
        if actual_index.empty:
            continue

        actual = actual.loc[actual_index]
        expected_target = signal.loc[actual_index, "target_exposure"]
        expected_shifted_position = expected_position.loc[actual_index]
        expected_realized_vol = signal.loc[actual_index, "realized_daily_vol"]
        expected_trend_ma = signal.loc[actual_index, "qqq_trend_ma"]
        expected_momentum = signal.loc[actual_index, "qqq_momentum"]
        actual_weight = pd.to_numeric(actual[trade_column], errors="coerce")

        rows_checked += int(len(actual))
        windows_checked += 1
        max_position_diff = max(max_position_diff, _max_abs_diff(actual_weight, expected_shifted_position))
        max_target_diff = max(max_target_diff, _max_abs_diff(actual["target_exposure"], expected_target))
        max_realized_vol_diff = max(max_realized_vol_diff, _max_abs_diff(actual["realized_daily_vol"], expected_realized_vol))
        max_trend_diff = max(max_trend_diff, _max_abs_diff(actual["qqq_trend_ma"], expected_trend_ma))
        max_momentum_diff = max(max_momentum_diff, _max_abs_diff(actual["qqq_momentum"], expected_momentum))

        first_position_failure = first_position_failure or _first_diff_date(
            actual_weight,
            expected_shifted_position,
            tolerance,
        )
        first_target_diff = first_target_diff or _first_diff_date(actual["target_exposure"], expected_target, tolerance)
        first_realized_vol_diff = first_realized_vol_diff or _first_diff_date(
            actual["realized_daily_vol"],
            expected_realized_vol,
            tolerance,
        )
        first_trend_diff = first_trend_diff or _first_diff_date(actual["qqq_trend_ma"], expected_trend_ma, tolerance)
        first_momentum_diff = first_momentum_diff or _first_diff_date(actual["qqq_momentum"], expected_momentum, tolerance)

        current_day_target = expected_target
        current_day_leak_mask = (
            (actual_weight - current_day_target).abs() <= tolerance
        ) & (
            (current_day_target - expected_shifted_position).abs() > tolerance
        )
        first_current_day_leak = first_current_day_leak or _first_true_date(current_day_leak_mask)

        warmup_complete, first_tradable = _first_warmup_complete_date(
            history,
            window,
            asset_config.trade_asset,
            asset_config.primary_signal_asset,
            asset_config.secondary_filter_asset,
        )
        if warmup_complete is not None and (
            earliest_warmup_complete is None or warmup_complete < earliest_warmup_complete
        ):
            earliest_warmup_complete = warmup_complete
        if first_tradable is not None and (
            earliest_tradable_after_warmup is None or first_tradable < earliest_tradable_after_warmup
        ):
            earliest_tradable_after_warmup = first_tradable
        if first_tradable is not None:
            premature = actual.index[(actual.index < first_tradable) & (actual_weight.abs() > tolerance)]
            if len(premature) > 0:
                warmup_failures.append(pd.Timestamp(premature[0]))

    if rows_checked == 0:
        raise ValueError("No stitched rows overlapped selected VolTarget walk-forward windows.")

    requested_execution = ""
    effective_execution = ""
    execution_warning_present = False
    silent_fallback_date = ""
    if "requested_execution_model" in stitched.columns:
        requested_execution = ";".join(sorted(str(value) for value in stitched["requested_execution_model"].dropna().unique()))
    if "execution_model" in stitched.columns:
        effective_execution = ";".join(sorted(str(value) for value in stitched["execution_model"].dropna().unique()))
    if "execution_model_warning" in stitched.columns:
        warning_values = stitched["execution_model_warning"].fillna("").astype(str)
        execution_warning_present = bool(warning_values.str.len().gt(0).any())
    if requested_execution and effective_execution and requested_execution != effective_execution and not execution_warning_present:
        fallback_mask = stitched["requested_execution_model"].astype(str) != stitched["execution_model"].astype(str)
        silent_fallback_date = _first_true_date(fallback_mask)

    position_shift_pass = bool(max_position_diff <= tolerance and not first_position_failure)
    target_diagnostic_pass = bool(max_target_diff <= tolerance and not first_target_diff)
    realized_vol_diagnostic_pass = bool(max_realized_vol_diff <= tolerance and not first_realized_vol_diff)
    trend_diagnostic_pass = bool(max_trend_diff <= tolerance and not first_trend_diff)
    momentum_diagnostic_pass = bool(max_momentum_diff <= tolerance and not first_momentum_diff)
    no_current_day_return_pass = bool(position_shift_pass and not first_current_day_leak)
    realized_vol_prior_bar_pass = bool(position_shift_pass and realized_vol_diagnostic_pass and no_current_day_return_pass)
    trend_shift_pass = bool(position_shift_pass and trend_diagnostic_pass)
    momentum_shift_pass = bool(position_shift_pass and momentum_diagnostic_pass)
    warmup_pass = not warmup_failures
    ohlc_fallback_pass = not silent_fallback_date
    passed = bool(
        position_shift_pass
        and target_diagnostic_pass
        and realized_vol_prior_bar_pass
        and trend_shift_pass
        and momentum_shift_pass
        and no_current_day_return_pass
        and warmup_pass
        and ohlc_fallback_pass
    )

    summary = pd.DataFrame(
        [
            {
                "config_path": str(config_path) if config_path else "",
                "generated_output_dir": str(config.get("output_dir", "")),
                "audit_output_dir": str(output_dir) if output_dir else "",
                "passed": passed,
                "rows_checked": rows_checked,
                "windows_checked": windows_checked,
                "tolerance": float(tolerance),
                "transaction_cost_bps": transaction_cost_bps,
                "applied_trade_weight_column": trade_column,
                "target_exposure_diagnostic_pass": target_diagnostic_pass,
                "realized_vol_diagnostic_pass": realized_vol_diagnostic_pass,
                "realized_vol_prior_bar_pass": realized_vol_prior_bar_pass,
                "trend_shift_pass": trend_shift_pass,
                "momentum_shift_pass": momentum_shift_pass,
                "position_shift_pass": position_shift_pass,
                "no_current_day_return_pass": no_current_day_return_pass,
                "warmup_pass": warmup_pass,
                "ohlc_fallback_pass": ohlc_fallback_pass,
                "execution_warning_present": execution_warning_present,
                "requested_execution_model": requested_execution,
                "effective_execution_model": effective_execution,
                "first_stitched_date": first_stitched_date,
                "first_warmup_complete_date": earliest_warmup_complete.date().isoformat()
                if earliest_warmup_complete is not None
                else "",
                "first_tradable_after_warmup": earliest_tradable_after_warmup.date().isoformat()
                if earliest_tradable_after_warmup is not None
                else "",
                "first_position_shift_failure_date": first_position_failure,
                "first_current_day_leak_date": first_current_day_leak,
                "first_target_exposure_diff_date": first_target_diff,
                "first_realized_vol_diff_date": first_realized_vol_diff,
                "first_trend_ma_diff_date": first_trend_diff,
                "first_momentum_diff_date": first_momentum_diff,
                "first_warmup_failure_date": min(warmup_failures).date().isoformat()
                if warmup_failures
                else "",
                "first_silent_ohlc_fallback_date": silent_fallback_date,
                "max_abs_position_shift_diff": max_position_diff,
                "max_abs_target_exposure_diff": max_target_diff,
                "max_abs_realized_daily_vol_diff": max_realized_vol_diff,
                "max_abs_trend_ma_diff": max_trend_diff,
                "max_abs_momentum_diff": max_momentum_diff,
            }
        ]
    )
    return summary


def audit_voltarget_timing(
    config_path: Path,
    *,
    output_dir: Path = DEFAULT_AUDIT_OUTPUT_DIR,
    tolerance: float = 1e-8,
) -> Tuple[pd.DataFrame, bool]:
    config_path = Path(config_path)
    config = load_yaml_file(config_path)
    generated_output_dir = Path(str(config.get("output_dir", "")))
    if not generated_output_dir:
        raise ValueError("VolTarget timing audit requires config output_dir.")
    stitched, windows = _read_output_frames(generated_output_dir)
    data = _load_price_data(config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = audit_voltarget_timing_frames(
        config=config,
        stitched=stitched,
        windows=windows,
        price_data=data,
        config_path=config_path,
        output_dir=output_dir,
        tolerance=float(tolerance),
    )
    summary.to_csv(output_dir / SUMMARY_FILENAME, index=False)
    _write_report(output_dir / REPORT_FILENAME, summary)
    passed = bool(summary["passed"].iloc[0])
    print(_terminal_summary(summary))
    return summary, passed


def _terminal_summary(summary: pd.DataFrame) -> str:
    row = summary.iloc[0]
    status = "PASS" if bool(row["passed"]) else "FAIL"
    return "\n".join(
        [
            "=== VolTarget Timing Audit ===",
            f"STATUS: {status}",
            f"ROWS CHECKED: {int(row['rows_checked'])}",
            f"WINDOWS CHECKED: {int(row['windows_checked'])}",
            f"POSITION SHIFT PASS: {bool(row['position_shift_pass'])}",
            f"NO CURRENT-DAY RETURN PASS: {bool(row['no_current_day_return_pass'])}",
            f"FIRST TRADABLE AFTER WARMUP: {row['first_tradable_after_warmup']}",
        ]
    )


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    status = "PASS" if bool(row["passed"]) else "FAIL"
    lines = [
        "# VolTarget Timing Audit",
        "",
        f"Status: `{status}`",
        "",
        "This audit rebuilds the selected VolTarget walk-forward signals from price data and checks the generated daily output.",
        "The applied trade weight on day t must equal the target exposure computed at close t-1.",
        "",
        "## Checks",
        "",
        f"- Realized volatility prior-bar use: `{bool(row['realized_vol_prior_bar_pass'])}`",
        f"- Trend signal shifted consistently: `{bool(row['trend_shift_pass'])}`",
        f"- Momentum signal shifted consistently: `{bool(row['momentum_shift_pass'])}`",
        f"- No current-day TQQQ return exposure leak: `{bool(row['no_current_day_return_pass'])}`",
        f"- Position shifted by one bar: `{bool(row['position_shift_pass'])}`",
        f"- Warmup date reported: `{row['first_tradable_after_warmup']}`",
        f"- OHLC fallback visible when applicable: `{bool(row['ohlc_fallback_pass'])}`",
        "",
        "## Failure Dates",
        "",
        f"- First position shift failure: `{row['first_position_shift_failure_date']}`",
        f"- First current-day leak: `{row['first_current_day_leak_date']}`",
        f"- First realized-vol diagnostic diff: `{row['first_realized_vol_diff_date']}`",
        f"- First target-exposure diagnostic diff: `{row['first_target_exposure_diff_date']}`",
        f"- First silent OHLC fallback: `{row['first_silent_ohlc_fallback_date']}`",
        "",
        "## Max Differences",
        "",
        f"- Max position shift diff: `{float(row['max_abs_position_shift_diff']):.12g}`",
        f"- Max target exposure diff: `{float(row['max_abs_target_exposure_diff']):.12g}`",
        f"- Max realized daily vol diff: `{float(row['max_abs_realized_daily_vol_diff']):.12g}`",
        f"- Max trend MA diff: `{float(row['max_abs_trend_ma_diff']):.12g}`",
        f"- Max momentum diff: `{float(row['max_abs_momentum_diff']):.12g}`",
        "",
        "This is a signal-timing audit only. It does not evaluate strategy merit.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
