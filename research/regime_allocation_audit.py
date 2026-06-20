from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import make_regime_eval_piece
from .data import load_prices
from .execution import CLOSE_TO_CLOSE_SHIFTED
from .reports import _markdown_table
from .replay import (
    _as_float,
    _date_text,
    _load_old_run_config,
    _params_from_window_row,
    _read_date_indexed_csv,
)
from .strategies.ma_strategy import compute_ma
from .strategies.regime_strategy import build_regime_weights


DAILY_BASE_COLUMNS = [
    "date",
    "window_index",
    "QQQ_close",
    "TQQQ_close",
    "qqq_ma",
    "trend",
    "momentum",
    "vol",
    "risk_on",
    "super_risk_on",
    "transition",
    "raw_tqqq_weight",
    "shifted_tqqq_weight",
    "shifted_qqq_weight",
    "total_position",
    "turnover",
    "cost",
    "tqqq_return",
    "qqq_return",
    "strategy_return",
    "equity",
]

OLD_COMPARE_COLUMNS = {
    "tqqq_weight": "shifted_tqqq_weight",
    "qqq_weight": "shifted_qqq_weight",
    "position": "total_position",
    "turnover": "turnover",
    "ret": "strategy_return",
    "equity": "equity",
    "daily_ret_tqqq": "tqqq_return",
    "daily_ret_qqq": "qqq_return",
    "cost": "cost",
    "raw_tqqq_weight": "raw_tqqq_weight",
}

SUMMARY_COLUMNS = [
    "old_start",
    "old_end",
    "audit_start",
    "audit_end",
    "common_start",
    "common_end",
    "old_rows",
    "audit_rows",
    "common_rows",
    "max_abs_raw_tqqq_weight_diff",
    "max_abs_tqqq_weight_diff",
    "max_abs_qqq_weight_diff",
    "max_abs_position_diff",
    "max_abs_turnover_diff",
    "max_abs_cost_diff",
    "max_abs_tqqq_return_diff",
    "max_abs_qqq_return_diff",
    "max_abs_ret_diff",
    "max_abs_equity_diff",
    "first_raw_signal_divergence_date",
    "first_weight_divergence_date",
    "first_turnover_divergence_date",
    "first_return_divergence_date",
    "first_equity_divergence_date",
    "likely_source",
    "passed",
]

FIRST_DIVERGENCE_COLUMNS = [
    "date",
    "old_raw_tqqq_weight",
    "audit_raw_tqqq_weight",
    "raw_tqqq_weight_diff",
    "old_tqqq_weight",
    "audit_tqqq_weight",
    "tqqq_weight_diff",
    "old_qqq_weight",
    "audit_qqq_weight",
    "qqq_weight_diff",
    "old_position",
    "audit_position",
    "position_diff",
    "old_turnover",
    "audit_turnover",
    "turnover_diff",
    "old_cost",
    "audit_cost",
    "cost_diff",
    "old_daily_ret_tqqq",
    "audit_tqqq_return",
    "tqqq_return_diff",
    "old_daily_ret_qqq",
    "audit_qqq_return",
    "qqq_return_diff",
    "old_ret",
    "audit_strategy_return",
    "ret_diff",
    "old_equity",
    "audit_equity",
    "equity_diff",
]


def _is_material(value: Any, tolerance: float) -> bool:
    try:
        if pd.isna(value):
            return False
        return abs(float(value)) > float(tolerance)
    except (TypeError, ValueError):
        return False


def _max_abs_diff(old: pd.DataFrame, audit: pd.DataFrame, old_col: str, audit_col: str, index: pd.Index) -> float:
    if old_col not in old.columns or audit_col not in audit.columns or len(index) == 0:
        return np.nan
    old_values = pd.to_numeric(old.loc[index, old_col], errors="coerce")
    audit_values = pd.to_numeric(audit.loc[index, audit_col], errors="coerce")
    diff = (old_values - audit_values).abs().replace([np.inf, -np.inf], np.nan).dropna()
    return float(diff.max()) if not diff.empty else np.nan


def _first_diff_date(
    old: pd.DataFrame,
    audit: pd.DataFrame,
    pairs: Tuple[Tuple[str, str], ...],
    index: pd.Index,
    tolerance: float,
) -> str:
    for idx in index:
        for old_col, audit_col in pairs:
            if old_col not in old.columns or audit_col not in audit.columns:
                continue
            old_value = _as_float(old.at[idx, old_col], np.nan)
            audit_value = _as_float(audit.at[idx, audit_col], np.nan)
            if pd.notna(old_value) and pd.notna(audit_value) and abs(old_value - audit_value) > tolerance:
                return _date_text(pd.Timestamp(idx))
    return ""


def classify_regime_allocation_source(row: Dict[str, Any] | pd.Series, tolerance: float = 1e-8) -> str:
    raw_signal = _is_material(row.get("max_abs_raw_tqqq_weight_diff", np.nan), tolerance)
    shifted_weight = any(
        _is_material(row.get(column, np.nan), tolerance)
        for column in ("max_abs_tqqq_weight_diff", "max_abs_qqq_weight_diff", "max_abs_position_diff")
    )
    turnover_cost = any(
        _is_material(row.get(column, np.nan), tolerance)
        for column in ("max_abs_turnover_diff", "max_abs_cost_diff")
    )
    data_returns = any(
        _is_material(row.get(column, np.nan), tolerance)
        for column in ("max_abs_tqqq_return_diff", "max_abs_qqq_return_diff")
    )
    strategy_return = _is_material(row.get("max_abs_ret_diff", np.nan), tolerance)
    equity = _is_material(row.get("max_abs_equity_diff", np.nan), tolerance)

    sources: List[str] = []
    if raw_signal:
        sources.append("pre-shift signal mismatch")
    if shifted_weight:
        sources.append("shifted weight mismatch")
    if turnover_cost:
        sources.append("turnover/cost mismatch")
    if data_returns:
        sources.append("data mismatch")
    if strategy_return:
        sources.append("return composition mismatch")
    if equity and not strategy_return:
        sources.append("equity compounding mismatch")
    if shifted_weight and not raw_signal:
        sources.append("wrong old/new variant likely")

    if not sources and equity:
        sources.append("equity compounding mismatch")
    if not sources:
        sources.append("no material mismatch detected")
    return "; ".join(dict.fromkeys(sources))


def _load_audit_data(
    *,
    start_date: str,
    end_date: Optional[str],
    data_csv: Optional[Path],
    cache_dir: str,
) -> pd.DataFrame:
    if data_csv is not None:
        data = pd.read_csv(data_csv, parse_dates=["Date"]).set_index("Date").sort_index()
        missing = {"TQQQ", "QQQ"} - set(data.columns)
        if missing:
            raise ValueError(f"data_csv is missing required symbols: {sorted(missing)}")
        data = data.loc[data.index >= pd.Timestamp(start_date)].copy()
        if end_date:
            data = data.loc[data.index <= pd.Timestamp(end_date)].copy()
        return data

    return load_prices(
        start=start_date,
        end=end_date,
        symbols=("TQQQ", "QQQ"),
        use_csv_if_exists=True,
        cache_dir=cache_dir,
        include_ohlc=False,
    )


def _raw_regime_diagnostics(data: pd.DataFrame, params) -> pd.DataFrame:
    asset_config = params.asset_config.normalized()
    tqqq_close = data[asset_config.trade_asset].astype(float)
    qqq_close = data[asset_config.primary_signal_asset].astype(float)
    filter_close = data[asset_config.secondary_filter_asset].astype(float)

    qqq_ma = compute_ma(filter_close, params.trend_window, "sma")
    trend = filter_close / qqq_ma - 1.0
    momentum = qqq_close.pct_change(params.momentum_window)
    vol = filter_close.pct_change().rolling(params.vol_window, min_periods=params.vol_window).std()

    if params.market_internals.use_market_internals:
        risk_on = pd.Series(False, index=data.index)
        super_risk_on = pd.Series(False, index=data.index)
        transition = pd.Series(False, index=data.index)
    else:
        risk_on = (
            (trend > params.trend_on)
            & (momentum > params.mom_on)
            & (vol < params.vol_cap)
        ).fillna(False)
        super_risk_on = (
            risk_on
            & (trend > params.trend_on + 0.015).fillna(False)
            & (momentum > params.mom_on + 0.04).fillna(False)
            & (vol < params.vol_cap * 0.90).fillna(False)
        )
        transition = (
            (~risk_on)
            & (trend > params.trend_off)
            & (momentum > params.mom_off)
        ).fillna(False)

    weights = build_regime_weights(
        tqqq_close=tqqq_close,
        qqq_close=qqq_close,
        params=params,
        price_data=data,
        filter_close=filter_close,
    )
    return pd.DataFrame(
        {
            "QQQ_close": qqq_close,
            "TQQQ_close": tqqq_close,
            "qqq_ma": qqq_ma,
            "trend": trend,
            "momentum": momentum,
            "vol": vol,
            "risk_on": risk_on.astype(int),
            "super_risk_on": super_risk_on.astype(int),
            "transition": transition.astype(int),
            "raw_tqqq_weight": weights["tqqq_weight"],
        },
        index=data.index,
    )


def _window_audit_piece(
    *,
    data: pd.DataFrame,
    params,
    window_index: int,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    transaction_cost_bps: float,
    prev_tqqq_weight: Optional[float],
    prev_qqq_weight: Optional[float],
    execution_model: str,
) -> pd.DataFrame:
    history = data.loc[:eval_end].copy()
    raw = _raw_regime_diagnostics(history, params)
    piece = make_regime_eval_piece(
        full_data=data,
        params=params,
        eval_start=eval_start,
        eval_end=eval_end,
        transaction_cost_bps=transaction_cost_bps,
        prev_tqqq_weight=prev_tqqq_weight,
        prev_qqq_weight=prev_qqq_weight,
        execution_model=execution_model,
    )

    out = raw.loc[piece.index].copy()
    out["window_index"] = int(window_index)
    out["shifted_tqqq_weight"] = piece["tqqq_weight"]
    out["shifted_qqq_weight"] = piece["qqq_weight"]
    out["total_position"] = piece["position"]
    out["turnover"] = piece["turnover"]
    out["cost"] = piece["cost"]
    out["tqqq_return"] = piece["daily_ret_tqqq"]
    out["qqq_return"] = piece["daily_ret_qqq"]
    out["strategy_return"] = piece["ret"]
    return out


def _attach_old_comparison_columns(daily: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    common = out.index.intersection(old.index)
    for old_col, audit_col in OLD_COMPARE_COLUMNS.items():
        old_name = f"old_{old_col}"
        diff_name = f"{old_col}_diff"
        out[old_name] = np.nan
        out[diff_name] = np.nan
        if old_col in old.columns and audit_col in out.columns and len(common):
            out.loc[common, old_name] = pd.to_numeric(old.loc[common, old_col], errors="coerce")
            out.loc[common, diff_name] = out.loc[common, old_name] - pd.to_numeric(
                out.loc[common, audit_col],
                errors="coerce",
            )
    return out


def _summary(old: pd.DataFrame, daily: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    common = old.index.intersection(daily.index)
    row = {
        "old_start": _date_text(old.index.min()) if not old.empty else "",
        "old_end": _date_text(old.index.max()) if not old.empty else "",
        "audit_start": _date_text(daily.index.min()) if not daily.empty else "",
        "audit_end": _date_text(daily.index.max()) if not daily.empty else "",
        "common_start": _date_text(common.min()) if len(common) else "",
        "common_end": _date_text(common.max()) if len(common) else "",
        "old_rows": int(len(old)),
        "audit_rows": int(len(daily)),
        "common_rows": int(len(common)),
        "max_abs_raw_tqqq_weight_diff": _max_abs_diff(old, daily, "raw_tqqq_weight", "raw_tqqq_weight", common),
        "max_abs_tqqq_weight_diff": _max_abs_diff(old, daily, "tqqq_weight", "shifted_tqqq_weight", common),
        "max_abs_qqq_weight_diff": _max_abs_diff(old, daily, "qqq_weight", "shifted_qqq_weight", common),
        "max_abs_position_diff": _max_abs_diff(old, daily, "position", "total_position", common),
        "max_abs_turnover_diff": _max_abs_diff(old, daily, "turnover", "turnover", common),
        "max_abs_cost_diff": _max_abs_diff(old, daily, "cost", "cost", common),
        "max_abs_tqqq_return_diff": _max_abs_diff(old, daily, "daily_ret_tqqq", "tqqq_return", common),
        "max_abs_qqq_return_diff": _max_abs_diff(old, daily, "daily_ret_qqq", "qqq_return", common),
        "max_abs_ret_diff": _max_abs_diff(old, daily, "ret", "strategy_return", common),
        "max_abs_equity_diff": _max_abs_diff(old, daily, "equity", "equity", common),
        "first_raw_signal_divergence_date": _first_diff_date(
            old,
            daily,
            (("raw_tqqq_weight", "raw_tqqq_weight"),),
            common,
            tolerance,
        ),
        "first_weight_divergence_date": _first_diff_date(
            old,
            daily,
            (
                ("tqqq_weight", "shifted_tqqq_weight"),
                ("qqq_weight", "shifted_qqq_weight"),
                ("position", "total_position"),
            ),
            common,
            tolerance,
        ),
        "first_turnover_divergence_date": _first_diff_date(
            old,
            daily,
            (("turnover", "turnover"), ("cost", "cost")),
            common,
            tolerance,
        ),
        "first_return_divergence_date": _first_diff_date(
            old,
            daily,
            (
                ("daily_ret_tqqq", "tqqq_return"),
                ("daily_ret_qqq", "qqq_return"),
                ("ret", "strategy_return"),
            ),
            common,
            tolerance,
        ),
        "first_equity_divergence_date": _first_diff_date(
            old,
            daily,
            (("equity", "equity"),),
            common,
            tolerance,
        ),
    }
    diff_columns = [
        "max_abs_raw_tqqq_weight_diff",
        "max_abs_tqqq_weight_diff",
        "max_abs_qqq_weight_diff",
        "max_abs_position_diff",
        "max_abs_turnover_diff",
        "max_abs_cost_diff",
        "max_abs_tqqq_return_diff",
        "max_abs_qqq_return_diff",
        "max_abs_ret_diff",
        "max_abs_equity_diff",
    ]
    available_diffs = [
        float(row[column])
        for column in diff_columns
        if pd.notna(row[column])
    ]
    row["likely_source"] = classify_regime_allocation_source(row, tolerance=tolerance)
    row["passed"] = bool(
        len(common) == len(old) == len(daily)
        and available_diffs
        and all(abs(value) <= float(tolerance) for value in available_diffs)
    )
    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)


def _first_divergence_rows(old: pd.DataFrame, daily: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    common = old.index.intersection(daily.index)
    rows: List[Dict[str, Any]] = []
    for idx in common:
        values = {
            "date": _date_text(pd.Timestamp(idx)),
            "old_raw_tqqq_weight": _as_float(old.at[idx, "raw_tqqq_weight"], np.nan)
            if "raw_tqqq_weight" in old.columns
            else np.nan,
            "audit_raw_tqqq_weight": _as_float(daily.at[idx, "raw_tqqq_weight"], np.nan),
            "old_tqqq_weight": _as_float(old.at[idx, "tqqq_weight"], np.nan) if "tqqq_weight" in old.columns else np.nan,
            "audit_tqqq_weight": _as_float(daily.at[idx, "shifted_tqqq_weight"], np.nan),
            "old_qqq_weight": _as_float(old.at[idx, "qqq_weight"], np.nan) if "qqq_weight" in old.columns else np.nan,
            "audit_qqq_weight": _as_float(daily.at[idx, "shifted_qqq_weight"], np.nan),
            "old_position": _as_float(old.at[idx, "position"], np.nan) if "position" in old.columns else np.nan,
            "audit_position": _as_float(daily.at[idx, "total_position"], np.nan),
            "old_turnover": _as_float(old.at[idx, "turnover"], np.nan) if "turnover" in old.columns else np.nan,
            "audit_turnover": _as_float(daily.at[idx, "turnover"], np.nan),
            "old_cost": _as_float(old.at[idx, "cost"], np.nan) if "cost" in old.columns else np.nan,
            "audit_cost": _as_float(daily.at[idx, "cost"], np.nan),
            "old_daily_ret_tqqq": _as_float(old.at[idx, "daily_ret_tqqq"], np.nan)
            if "daily_ret_tqqq" in old.columns
            else np.nan,
            "audit_tqqq_return": _as_float(daily.at[idx, "tqqq_return"], np.nan),
            "old_daily_ret_qqq": _as_float(old.at[idx, "daily_ret_qqq"], np.nan)
            if "daily_ret_qqq" in old.columns
            else np.nan,
            "audit_qqq_return": _as_float(daily.at[idx, "qqq_return"], np.nan),
            "old_ret": _as_float(old.at[idx, "ret"], np.nan) if "ret" in old.columns else np.nan,
            "audit_strategy_return": _as_float(daily.at[idx, "strategy_return"], np.nan),
            "old_equity": _as_float(old.at[idx, "equity"], np.nan) if "equity" in old.columns else np.nan,
            "audit_equity": _as_float(daily.at[idx, "equity"], np.nan),
        }
        for old_key, audit_key, diff_key in (
            ("old_raw_tqqq_weight", "audit_raw_tqqq_weight", "raw_tqqq_weight_diff"),
            ("old_tqqq_weight", "audit_tqqq_weight", "tqqq_weight_diff"),
            ("old_qqq_weight", "audit_qqq_weight", "qqq_weight_diff"),
            ("old_position", "audit_position", "position_diff"),
            ("old_turnover", "audit_turnover", "turnover_diff"),
            ("old_cost", "audit_cost", "cost_diff"),
            ("old_daily_ret_tqqq", "audit_tqqq_return", "tqqq_return_diff"),
            ("old_daily_ret_qqq", "audit_qqq_return", "qqq_return_diff"),
            ("old_ret", "audit_strategy_return", "ret_diff"),
            ("old_equity", "audit_equity", "equity_diff"),
        ):
            values[diff_key] = values[old_key] - values[audit_key]
        if any(_is_material(values[column], tolerance) for column in values if column.endswith("_diff")):
            rows.append(values)
            if len(rows) >= 20:
                break
    return pd.DataFrame(rows, columns=FIRST_DIVERGENCE_COLUMNS)


def _write_report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    first_divergence: pd.DataFrame,
    windows_path: Path,
    old_stitched_path: Path,
) -> Path:
    report_path = output_dir / "regime_allocation_audit_report.md"
    likely_source = str(summary["likely_source"].iloc[0]) if not summary.empty else "unknown"
    lines = [
        "# Regime Allocation Baseline Audit",
        "",
        f"- Windows: `{windows_path}`",
        f"- Old stitched: `{old_stitched_path}`",
        f"- Likely source: `{likely_source}`",
        "",
        "## Summary",
        _markdown_table(summary, max_rows=5),
        "",
        "## First Divergences",
        _markdown_table(first_divergence, max_rows=20),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def audit_regime_allocation(
    *,
    windows_path: Path,
    old_stitched_path: Path,
    output_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tolerance: float = 1e-8,
    transaction_cost_bps: Optional[float] = None,
    execution_model: Optional[str] = None,
    data_csv: Optional[Path] = None,
    cache_dir: str = "./price_cache",
) -> pd.DataFrame:
    windows_path = Path(windows_path)
    old_stitched_path = Path(old_stitched_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(windows_path)
    if windows.empty:
        raise ValueError(f"Window file is empty: {windows_path}")
    old = _read_date_indexed_csv(old_stitched_path)
    if start_date:
        old = old.loc[old.index >= pd.Timestamp(start_date)].copy()
    if end_date:
        old = old.loc[old.index <= pd.Timestamp(end_date)].copy()
    if old.empty:
        raise ValueError("Old stitched file has no rows after applying start/end filters.")

    run_config = _load_old_run_config(windows_path)
    cost_bps = (
        float(transaction_cost_bps)
        if transaction_cost_bps is not None
        else _as_float(run_config.get("transaction_cost_bps", 10.0), 10.0)
    )
    effective_execution_model = str(
        execution_model
        or run_config.get("execution_model")
        or run_config.get("requested_execution_model")
        or CLOSE_TO_CLOSE_SHIFTED
    )
    data_start = start_date or str(pd.to_datetime(windows["train_start"]).min().date())
    data_end = end_date or _date_text(old.index.max())
    data = _load_audit_data(
        start_date=data_start,
        end_date=data_end,
        data_csv=data_csv,
        cache_dir=cache_dir,
    )

    pieces = []
    prev_tqqq_weight = 0.0
    prev_qqq_weight = 0.0
    for window_index, row in windows.reset_index(drop=True).iterrows():
        params = _params_from_window_row(row)
        test_start = pd.Timestamp(row["test_start"])
        test_end = pd.Timestamp(row["test_end"])
        if start_date:
            test_start = max(test_start, pd.Timestamp(start_date))
        if end_date:
            test_end = min(test_end, pd.Timestamp(end_date))
        if test_end < test_start:
            continue
        piece = _window_audit_piece(
            data=data,
            params=params,
            window_index=int(window_index) + 1,
            eval_start=test_start,
            eval_end=test_end,
            transaction_cost_bps=cost_bps,
            prev_tqqq_weight=prev_tqqq_weight,
            prev_qqq_weight=prev_qqq_weight,
            execution_model=effective_execution_model,
        )
        pieces.append(piece)
        prev_tqqq_weight = float(piece["shifted_tqqq_weight"].iloc[-1])
        prev_qqq_weight = float(piece["shifted_qqq_weight"].iloc[-1])

    if not pieces:
        raise ValueError("No allocation audit windows were produced.")

    daily = pd.concat(pieces).sort_index()
    daily = daily[~daily.index.duplicated(keep="first")].copy()
    daily["equity"] = (1.0 + daily["strategy_return"]).cumprod()
    daily = _attach_old_comparison_columns(daily, old)

    daily_out = daily.reset_index(names="date")
    ordered = [column for column in DAILY_BASE_COLUMNS if column in daily_out.columns]
    remaining = [column for column in daily_out.columns if column not in ordered]
    daily_out = daily_out[ordered + remaining]
    daily_out.to_csv(output_dir / "regime_allocation_audit_daily.csv", index=False)

    summary = _summary(old, daily, tolerance=float(tolerance))
    summary.to_csv(output_dir / "regime_allocation_audit_summary.csv", index=False)

    first_divergence = _first_divergence_rows(old, daily, tolerance=float(tolerance))
    first_divergence.to_csv(output_dir / "regime_allocation_first_divergence.csv", index=False)
    _write_report(
        output_dir=output_dir,
        summary=summary,
        first_divergence=first_divergence,
        windows_path=windows_path,
        old_stitched_path=old_stitched_path,
    )

    print(summary.to_string(index=False))
    print(f"\nRegime allocation audit written to: {output_dir / 'regime_allocation_audit_report.md'}")
    return summary
