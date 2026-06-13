from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import make_regime_eval_piece
from .data import load_prices
from .metrics import summarize_performance
from .reports import _markdown_table
from .strategies.regime_strategy import RegimeParams


PARAM_COLUMNS = [
    "train_start",
    "train_end",
    "test_start",
    "test_end",
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
    "old_test_final_equity",
    "replay_test_final_equity",
    "old_test_avg_exposure",
    "replay_test_avg_exposure",
    "old_test_trades",
    "replay_test_trades",
]

DIVERGENCE_COLUMNS = [
    "date",
    "old_ret",
    "replay_ret",
    "ret_diff",
    "old_position",
    "replay_position",
    "position_diff",
    "old_tqqq_weight",
    "replay_tqqq_weight",
    "old_qqq_weight",
    "replay_qqq_weight",
    "old_turnover",
    "replay_turnover",
    "old_equity",
    "replay_equity",
]

SUMMARY_COLUMNS = [
    "old_start",
    "old_end",
    "replay_start",
    "replay_end",
    "common_start",
    "common_end",
    "old_rows",
    "replay_rows",
    "common_rows",
    "old_final_equity",
    "replay_final_equity",
    "final_equity_diff",
    "max_abs_ret_diff",
    "max_abs_equity_diff",
    "max_abs_position_diff",
    "max_abs_tqqq_weight_diff",
    "max_abs_qqq_weight_diff",
    "max_abs_turnover_diff",
    "first_ret_divergence_date",
    "first_position_divergence_date",
    "first_weight_divergence_date",
    "passed",
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_date_indexed_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"CSV is empty: {path}")
    date_column = "Date" if "Date" in raw.columns else "date" if "date" in raw.columns else raw.columns[0]
    raw[date_column] = pd.to_datetime(raw[date_column], errors="coerce")
    if raw[date_column].isna().any():
        raise ValueError(f"Could not parse date column {date_column}: {path}")
    out = raw.set_index(date_column).sort_index()
    out.index.name = "Date"
    return out


def _date_text(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def _row_value(row: pd.Series, name: str, default: Any = np.nan) -> Any:
    return row[name] if name in row.index else default


def _params_from_window_row(row: pd.Series) -> RegimeParams:
    risk_off_weight = _optional_float(_row_value(row, "risk_off_weight", np.nan))
    return RegimeParams(
        trend_window=_as_int(row["trend_window"]),
        momentum_window=_as_int(row["momentum_window"]),
        vol_window=_as_int(row["vol_window"]),
        trend_on=_as_float(row["trend_on"]),
        trend_off=_as_float(row["trend_off"]),
        mom_on=_as_float(row["mom_on"]),
        mom_off=_as_float(row["mom_off"]),
        vol_cap=_as_float(row["vol_cap"]),
        risk_on_leverage=_as_float(row["risk_on_leverage"], 1.0),
        risk_off_qqq_position=_as_float(row["risk_off_qqq_position"], 0.0),
        transition_position=_as_float(row["transition_position"], 0.35),
        min_hold_days=_as_int(_row_value(row, "min_hold_days", 0), 0),
        cooldown_days=_as_int(_row_value(row, "cooldown_days", 2), 2),
        risk_off_symbol=str(_row_value(row, "risk_off_symbol", "QQQ") or "QQQ").upper(),
        risk_off_weight=risk_off_weight,
    )


def _load_replay_data(
    *,
    start_date: str,
    end_date: Optional[str],
    data_csv: Optional[Path],
) -> pd.DataFrame:
    if data_csv is None:
        return load_prices(
            start=start_date,
            end=end_date,
            symbols=("TQQQ", "QQQ"),
            use_csv_if_exists=True,
            include_ohlc=False,
        )

    data = pd.read_csv(data_csv, parse_dates=["Date"]).set_index("Date").sort_index()
    required = {"TQQQ", "QQQ"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"data_csv is missing required symbols: {sorted(missing)}")
    data = data.loc[data.index >= pd.Timestamp(start_date)].copy()
    if end_date:
        data = data.loc[data.index <= pd.Timestamp(end_date)].copy()
    return data


def _load_old_run_config(windows_path: Path) -> Dict[str, Any]:
    config_path = windows_path.parent / "run_config.json"
    if not config_path.exists():
        return {}
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _max_abs_diff(old: pd.DataFrame, replay: pd.DataFrame, column: str, common_index: pd.Index) -> float:
    if column not in old.columns or column not in replay.columns or len(common_index) == 0:
        return np.nan
    diff = (
        pd.to_numeric(old.loc[common_index, column], errors="coerce")
        - pd.to_numeric(replay.loc[common_index, column], errors="coerce")
    ).abs()
    diff = diff.replace([np.inf, -np.inf], np.nan).dropna()
    return float(diff.max()) if not diff.empty else np.nan


def _first_divergence_date(
    old: pd.DataFrame,
    replay: pd.DataFrame,
    columns: Tuple[str, ...],
    common_index: pd.Index,
    tolerance: float,
) -> str:
    if len(common_index) == 0:
        return ""
    for idx in common_index:
        for column in columns:
            if column not in old.columns or column not in replay.columns:
                continue
            old_value = _as_float(old.at[idx, column], np.nan)
            replay_value = _as_float(replay.at[idx, column], np.nan)
            if pd.notna(old_value) and pd.notna(replay_value) and abs(old_value - replay_value) > tolerance:
                return _date_text(pd.Timestamp(idx))
    return ""


def _first_divergence_rows(
    old: pd.DataFrame,
    replay: pd.DataFrame,
    common_index: pd.Index,
    tolerance: float,
) -> pd.DataFrame:
    rows = []
    for idx in common_index:
        values = {
            "date": _date_text(pd.Timestamp(idx)),
            "old_ret": _as_float(old.at[idx, "ret"], np.nan) if "ret" in old.columns else np.nan,
            "replay_ret": _as_float(replay.at[idx, "ret"], np.nan) if "ret" in replay.columns else np.nan,
            "old_position": _as_float(old.at[idx, "position"], np.nan) if "position" in old.columns else np.nan,
            "replay_position": _as_float(replay.at[idx, "position"], np.nan) if "position" in replay.columns else np.nan,
            "old_tqqq_weight": _as_float(old.at[idx, "tqqq_weight"], np.nan) if "tqqq_weight" in old.columns else np.nan,
            "replay_tqqq_weight": _as_float(replay.at[idx, "tqqq_weight"], np.nan) if "tqqq_weight" in replay.columns else np.nan,
            "old_qqq_weight": _as_float(old.at[idx, "qqq_weight"], np.nan) if "qqq_weight" in old.columns else np.nan,
            "replay_qqq_weight": _as_float(replay.at[idx, "qqq_weight"], np.nan) if "qqq_weight" in replay.columns else np.nan,
            "old_turnover": _as_float(old.at[idx, "turnover"], np.nan) if "turnover" in old.columns else np.nan,
            "replay_turnover": _as_float(replay.at[idx, "turnover"], np.nan) if "turnover" in replay.columns else np.nan,
            "old_equity": _as_float(old.at[idx, "equity"], np.nan) if "equity" in old.columns else np.nan,
            "replay_equity": _as_float(replay.at[idx, "equity"], np.nan) if "equity" in replay.columns else np.nan,
        }
        values["ret_diff"] = values["old_ret"] - values["replay_ret"]
        values["position_diff"] = values["old_position"] - values["replay_position"]
        material = any(
            pd.notna(values[key]) and abs(values[key]) > tolerance
            for key in ("ret_diff", "position_diff")
        )
        for old_key, replay_key in (
            ("old_tqqq_weight", "replay_tqqq_weight"),
            ("old_qqq_weight", "replay_qqq_weight"),
            ("old_turnover", "replay_turnover"),
            ("old_equity", "replay_equity"),
        ):
            if pd.notna(values[old_key]) and pd.notna(values[replay_key]):
                material = material or abs(values[old_key] - values[replay_key]) > tolerance
        if material:
            rows.append(values)
            if len(rows) >= 20:
                break
    return pd.DataFrame(rows, columns=DIVERGENCE_COLUMNS)


def _comparison_summary(
    old: pd.DataFrame,
    replay: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    common_index = old.index.intersection(replay.index)
    old_final = _as_float(old["equity"].iloc[-1], np.nan) if "equity" in old.columns and not old.empty else np.nan
    replay_final = (
        _as_float(replay["equity"].iloc[-1], np.nan)
        if "equity" in replay.columns and not replay.empty
        else np.nan
    )
    row = {
        "old_start": _date_text(old.index.min()) if not old.empty else "",
        "old_end": _date_text(old.index.max()) if not old.empty else "",
        "replay_start": _date_text(replay.index.min()) if not replay.empty else "",
        "replay_end": _date_text(replay.index.max()) if not replay.empty else "",
        "common_start": _date_text(common_index.min()) if len(common_index) else "",
        "common_end": _date_text(common_index.max()) if len(common_index) else "",
        "old_rows": int(len(old)),
        "replay_rows": int(len(replay)),
        "common_rows": int(len(common_index)),
        "old_final_equity": old_final,
        "replay_final_equity": replay_final,
        "final_equity_diff": replay_final - old_final if pd.notna(old_final) and pd.notna(replay_final) else np.nan,
        "max_abs_ret_diff": _max_abs_diff(old, replay, "ret", common_index),
        "max_abs_equity_diff": _max_abs_diff(old, replay, "equity", common_index),
        "max_abs_position_diff": _max_abs_diff(old, replay, "position", common_index),
        "max_abs_tqqq_weight_diff": _max_abs_diff(old, replay, "tqqq_weight", common_index),
        "max_abs_qqq_weight_diff": _max_abs_diff(old, replay, "qqq_weight", common_index),
        "max_abs_turnover_diff": _max_abs_diff(old, replay, "turnover", common_index),
        "first_ret_divergence_date": _first_divergence_date(old, replay, ("ret",), common_index, tolerance),
        "first_position_divergence_date": _first_divergence_date(old, replay, ("position",), common_index, tolerance),
        "first_weight_divergence_date": _first_divergence_date(
            old,
            replay,
            ("tqqq_weight", "qqq_weight"),
            common_index,
            tolerance,
        ),
    }
    diff_columns = [
        "final_equity_diff",
        "max_abs_ret_diff",
        "max_abs_equity_diff",
        "max_abs_position_diff",
        "max_abs_tqqq_weight_diff",
        "max_abs_qqq_weight_diff",
        "max_abs_turnover_diff",
    ]
    row["passed"] = bool(
        len(common_index) == len(old) == len(replay)
        and all(pd.notna(row[column]) and abs(float(row[column])) <= tolerance for column in diff_columns)
    )
    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)


def _write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    window_replay: pd.DataFrame,
    first_divergence: pd.DataFrame,
) -> Path:
    report_path = output_dir / "replay_compare_report.md"
    lines = [
        "# Legacy Regime Window Replay",
        "",
        "This diagnostic replays selected legacy walk-forward parameters in the current backtest engine without running grid search.",
        "",
        "## Comparison Summary",
        _markdown_table(summary, max_rows=5),
        "",
        "## Window Replay",
        _markdown_table(window_replay, max_rows=50),
        "",
        "## First Divergences",
        _markdown_table(first_divergence, max_rows=20),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def replay_regime_windows(
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
) -> pd.DataFrame:
    windows_path = Path(windows_path)
    old_stitched_path = Path(old_stitched_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(windows_path)
    if windows.empty:
        raise ValueError(f"Window file is empty: {windows_path}")
    old = _read_date_indexed_csv(old_stitched_path)

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
        or "close_to_close_shifted"
    )
    data_start = start_date or str(pd.to_datetime(windows["train_start"]).min().date())
    data_end = end_date or _date_text(old.index.max())
    data = _load_replay_data(start_date=data_start, end_date=data_end, data_csv=data_csv)

    pieces = []
    replay_rows = []
    prev_tqqq_weight = 0.0
    prev_qqq_weight = 0.0
    for _, row in windows.iterrows():
        params = _params_from_window_row(row)
        test_start = pd.Timestamp(row["test_start"])
        test_end = pd.Timestamp(row["test_end"])
        if end_date is not None:
            test_end = min(test_end, pd.Timestamp(end_date))
        piece = make_regime_eval_piece(
            full_data=data,
            params=params,
            eval_start=test_start,
            eval_end=test_end,
            transaction_cost_bps=cost_bps,
            prev_tqqq_weight=prev_tqqq_weight,
            prev_qqq_weight=prev_qqq_weight,
            execution_model=effective_execution_model,
        )
        benchmark_ret = data[params.asset_config.benchmark_symbol].pct_change().fillna(0.0).loc[piece.index]
        replay_perf = summarize_performance(piece, benchmark_ret)
        replay_rows.append(
            {
                "train_start": str(row["train_start"]),
                "train_end": str(row["train_end"]),
                "test_start": str(row["test_start"]),
                "test_end": str(row["test_end"]),
                "trend_window": params.trend_window,
                "momentum_window": params.momentum_window,
                "vol_window": params.vol_window,
                "trend_on": params.trend_on,
                "trend_off": params.trend_off,
                "mom_on": params.mom_on,
                "mom_off": params.mom_off,
                "vol_cap": params.vol_cap,
                "risk_on_leverage": params.risk_on_leverage,
                "risk_off_qqq_position": params.risk_off_qqq_position,
                "transition_position": params.transition_position,
                "min_hold_days": params.min_hold_days,
                "cooldown_days": params.cooldown_days,
                "old_test_final_equity": _row_value(row, "test_final_equity", np.nan),
                "replay_test_final_equity": replay_perf["final_equity"],
                "old_test_avg_exposure": _row_value(row, "test_avg_exposure", np.nan),
                "replay_test_avg_exposure": replay_perf["avg_exposure"],
                "old_test_trades": _row_value(row, "test_trades", np.nan),
                "replay_test_trades": replay_perf["trades"],
            }
        )
        pieces.append(piece)
        prev_tqqq_weight = float(piece["tqqq_weight"].iloc[-1])
        prev_qqq_weight = float(piece["qqq_weight"].iloc[-1])

    replayed = pd.concat(pieces).sort_index()
    replayed = replayed[~replayed.index.duplicated(keep="first")].copy()
    replayed["equity"] = (1.0 + replayed["ret"]).cumprod()
    replayed.to_csv(output_dir / "replayed_stitched_equity.csv", index_label="Date")

    window_replay = pd.DataFrame(replay_rows, columns=PARAM_COLUMNS)
    window_replay.to_csv(output_dir / "window_param_replay.csv", index=False)

    summary = _comparison_summary(old, replayed, tolerance=tolerance)
    summary.to_csv(output_dir / "replay_compare_summary.csv", index=False)

    common_index = old.index.intersection(replayed.index)
    first_divergence = _first_divergence_rows(old, replayed, common_index, tolerance=tolerance)
    first_divergence.to_csv(output_dir / "first_divergence.csv", index=False)
    _write_report(output_dir, summary, window_replay, first_divergence)

    print(summary.to_string(index=False))
    print(f"\nReplay outputs written to: {output_dir}")
    return summary
