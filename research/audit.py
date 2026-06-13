from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import CASH_SYMBOL, load_prices
from .execution import CLOSE_TO_CLOSE_SHIFTED, has_ohlc, normalize_execution_model
from .experiments import (
    _asset_config,
    _asset_config_symbols_from_config,
    _load_price_data,
    _market_internals_symbols_from_config,
    _risk_off_symbols_from_config,
    _rotation_required_symbols_from_config,
    _rotation_symbols_from_config,
    load_yaml_file,
)
from .reports import _markdown_table


RAW_FINAL_EQUITY_OBJECTIVES = {"objective_final_ratio", "final_equity_ratio"}
RISK_FIRST_OBJECTIVES = {"sharpe", "calmar", "objective_sharpe", "objective_calmar"}
HIGH_GRID_WARNING_THRESHOLD = 10000
BASELINE_RETURN_COLUMNS = [
    "date",
    "old_daily_ret_tqqq",
    "current_daily_ret_tqqq",
    "tqqq_return_diff",
    "old_daily_ret_qqq",
    "current_daily_ret_qqq",
    "qqq_return_diff",
    "old_daily_ret",
    "current_daily_ret",
    "daily_ret_diff",
]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, range):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _unique(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).upper()
        if text and text not in out:
            out.append(text)
    return out


def _resolve_config_path(parent_config_path: Path, value: Any) -> Path:
    candidate = Path(str(value))
    if candidate.exists():
        return candidate
    return parent_config_path.parent / candidate


def _output_dir_for_parent(config_path: Path, parent_config: Dict[str, Any]) -> Path:
    return Path(parent_config.get("output_dir", Path("outputs") / config_path.stem))


def _strategy_entries(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = config.get("strategies")
    if entries is None:
        entries = config.get("configs")
    if isinstance(entries, list) and entries:
        normalized = []
        for item in entries:
            if isinstance(item, str):
                normalized.append({"config": item})
            elif isinstance(item, dict):
                normalized.append(copy.deepcopy(item))
            else:
                normalized.append({"invalid_entry": repr(item)})
        return normalized
    return [{"experiment": copy.deepcopy(config)}]


def _load_entry_config(parent_config_path: Path, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    if "config" in entry:
        config_path = _resolve_config_path(parent_config_path, entry["config"])
        config = load_yaml_file(config_path)
        label = str(config_path)
    else:
        config = copy.deepcopy(entry.get("experiment", entry))
        label = str(parent_config_path)
    overrides = entry.get("overrides")
    if isinstance(overrides, dict):
        config.update(copy.deepcopy(overrides))
    return config, label


def _iter_experiment_configs(config_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    parent = load_yaml_file(config_path)
    entries = _strategy_entries(parent)
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries, start=1):
        row = {
            "entry_index": idx,
            "family": str(entry.get("family", entry.get("name", ""))),
            "category": str(entry.get("category", "")),
            "entry_error": "",
        }
        try:
            experiment_config, experiment_path = _load_entry_config(config_path, entry)
            row["config"] = experiment_config
            row["config_path"] = experiment_path
        except Exception as exc:
            row["config"] = {}
            row["config_path"] = str(entry.get("config", ""))
            row["entry_error"] = str(exc)
        rows.append(row)
    return parent, rows


def _grid_option_count(key: str, value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return 1
    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        if key in {"risk_assets", "risk_asset_sets", "market_internals_components"}:
            if all(not isinstance(item, (list, tuple)) for item in value):
                return 1
        return len(value)
    return 1


def _estimated_parameter_combinations(config: Dict[str, Any]) -> int:
    grid = config.get("parameter_grid")
    if not isinstance(grid, dict):
        return 1
    total = 1
    for key, value in grid.items():
        count = _grid_option_count(str(key), value)
        total *= max(int(count), 1)
    return int(total)


def _objective_prioritizes_raw_final_equity(config: Dict[str, Any]) -> bool:
    objective = str(config.get("objective", "")).strip()
    return objective in RAW_FINAL_EQUITY_OBJECTIVES


def _is_main_tqqq_experiment(config: Dict[str, Any], category: str) -> bool:
    benchmark_symbol = str(config.get("benchmark_symbol", "")).upper()
    try:
        trade_asset = _asset_config(config).trade_asset
    except Exception:
        trade_asset = str(config.get("trade_asset", "TQQQ")).upper()
    return str(category).lower() == "tqqq" or benchmark_symbol == "TQQQ" or trade_asset == "TQQQ"


def _output_overwrite_risk(output_dir: str) -> bool:
    if not output_dir:
        return False
    path = Path(output_dir)
    try:
        return path.exists() and any(path.iterdir())
    except OSError:
        return False


def _config_warnings(
    *,
    config: Dict[str, Any],
    category: str,
    estimated_combinations: int,
    output_dir: str,
) -> List[str]:
    warnings: List[str] = []
    if "benchmark_symbol" not in config or not str(config.get("benchmark_symbol", "")).strip():
        warnings.append("missing benchmark_symbol")
    if "strategy_name" not in config:
        warnings.append("missing strategy_name")
    if "objective" not in config:
        warnings.append("missing objective")
    objective = str(config.get("objective", "")).strip()
    if _is_main_tqqq_experiment(config, category):
        if objective.lower() in RISK_FIRST_OBJECTIVES or objective.lower() in {"sharpe", "calmar"}:
            warnings.append("main TQQQ experiment uses Sharpe/Calmar objective")
        elif objective and objective not in RAW_FINAL_EQUITY_OBJECTIVES:
            warnings.append("main TQQQ experiment does not prioritize raw final_equity_ratio")
    if estimated_combinations > HIGH_GRID_WARNING_THRESHOLD:
        warnings.append(f"large parameter grid: {estimated_combinations} combinations")
    if not output_dir:
        warnings.append("missing output_dir")
    elif _output_overwrite_risk(output_dir):
        warnings.append("output_dir already exists and is non-empty")
    train_years = config.get("train_years")
    test_years = config.get("test_years")
    if train_years is None or test_years is None:
        warnings.append("missing train/test setting")
    else:
        try:
            if int(train_years) <= 0 or int(test_years) <= 0:
                warnings.append("non-positive train/test setting")
        except (TypeError, ValueError):
            warnings.append("invalid train/test setting")
    try:
        if float(config.get("transaction_cost_bps", 0.0)) == 0.0:
            warnings.append("zero transaction costs")
    except (TypeError, ValueError):
        warnings.append("invalid transaction_cost_bps")
    return warnings


def audit_config(config_path: Path) -> pd.DataFrame:
    config_path = Path(config_path)
    parent, entries = _iter_experiment_configs(config_path)
    output_dir = _output_dir_for_parent(config_path, parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    number_of_experiments = len(entries)
    tournament_costs = ";".join(str(value) for value in _as_list(parent.get("cost_scenarios")))

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        config = entry["config"]
        family = entry.get("family", "")
        category = entry.get("category", "")
        estimated = _estimated_parameter_combinations(config)
        try:
            asset_config = _asset_config(config)
        except Exception:
            asset_config = None
        trade_asset = asset_config.trade_asset if asset_config is not None else str(config.get("trade_asset", ""))
        output = str(config.get("output_dir", ""))
        if not output and parent.get("output_dir"):
            output = str(Path(parent["output_dir"]) / "experiments" / str(config.get("experiment_name", family or entry["entry_index"])))
        objectives = ";".join(str(value) for value in _as_list(config.get("objectives")))
        warnings = _config_warnings(
            config=config,
            category=str(category),
            estimated_combinations=estimated,
            output_dir=output,
        )
        if entry.get("entry_error"):
            warnings.append(f"entry load error: {entry['entry_error']}")
        rows.append(
            {
                "config_path": entry.get("config_path", ""),
                "entry_index": entry["entry_index"],
                "family": family,
                "category": category,
                "number_of_experiments": number_of_experiments,
                "experiment_name": config.get("experiment_name", family),
                "strategy_name": config.get("strategy_name", ""),
                "benchmark_symbol": config.get("benchmark_symbol", ""),
                "trade_asset": trade_asset,
                "objective": config.get("objective", ""),
                "objectives": objectives,
                "train_years": config.get("train_years", ""),
                "test_years": config.get("test_years", ""),
                "walk_forward_mode": config.get("walk_forward_mode", "rolling"),
                "transaction_cost_bps": config.get("transaction_cost_bps", ""),
                "tournament_cost_scenarios": tournament_costs,
                "estimated_parameter_combinations": estimated,
                "output_dir": output,
                "overwrite_risk": _output_overwrite_risk(output),
                "objective_prioritizes_raw_final_equity_ratio": _objective_prioritizes_raw_final_equity(config),
                "warnings": "; ".join(warnings),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "audit_config_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nConfig audit written to: {output_dir / 'audit_config_summary.csv'}")
    return summary


def _required_symbols(config: Dict[str, Any]) -> List[str]:
    symbols = _as_list(config.get("symbols"))
    benchmark = config.get("benchmark_symbol")
    if benchmark:
        symbols.append(benchmark)
    for getter in (
        _asset_config_symbols_from_config,
        _risk_off_symbols_from_config,
        _market_internals_symbols_from_config,
        _rotation_symbols_from_config,
        _rotation_required_symbols_from_config,
    ):
        try:
            symbols.extend(getter(config))
        except Exception:
            continue
    if bool(config.get("use_synthetic_leverage", False)):
        symbols.extend(
            [
                config.get("synthetic_base_symbol", "QQQ"),
                config.get("synthetic_target_symbol", "TQQQ"),
                config.get("real_leveraged_symbol", config.get("synthetic_target_symbol", "TQQQ")),
            ]
        )
    return _unique(symbols)


def _symbol_summary(symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
    if symbol in data.columns:
        series = data[symbol].dropna()
        return {
            "symbol_loaded_successfully": True,
            "symbol_start_date": series.index.min().date().isoformat() if not series.empty else "",
            "symbol_end_date": series.index.max().date().isoformat() if not series.empty else "",
            "symbol_row_count": int(len(series)),
            "missing_date_count": int(data[symbol].isna().sum()),
        }
    return {
        "symbol_loaded_successfully": False,
        "symbol_start_date": "",
        "symbol_end_date": "",
        "symbol_row_count": 0,
        "missing_date_count": "",
    }


def _cash_aligned(required: Sequence[str], data: pd.DataFrame) -> Any:
    if CASH_SYMBOL not in required:
        return ""
    return bool(CASH_SYMBOL in data.columns and len(data[CASH_SYMBOL].dropna()) == len(data.index))


def _synthetic_overlap(config: Dict[str, Any], data: pd.DataFrame) -> int:
    if not bool(config.get("use_synthetic_leverage", False)):
        return 0
    target = str(config.get("synthetic_target_symbol", "TQQQ")).upper()
    real_symbol = str(config.get("real_leveraged_symbol", target)).upper()
    real_column = f"REAL_{real_symbol}"
    if target not in data.columns or real_column not in data.columns:
        return 0
    overlap = data[[target, real_column]].dropna()
    return int(len(overlap))


def _audit_one_data_entry(
    *,
    parent: Dict[str, Any],
    entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    config = entry["config"]
    required = _required_symbols(config)
    family = entry.get("family", "")
    category = entry.get("category", "")
    base = {
        "config_path": entry.get("config_path", ""),
        "entry_index": entry["entry_index"],
        "family": family,
        "category": category,
        "experiment_name": config.get("experiment_name", family),
        "strategy_name": config.get("strategy_name", ""),
        "required_symbols": ";".join(required),
        "benchmark_symbol": config.get("benchmark_symbol", ""),
        "execution_model": str(config.get("execution_model", CLOSE_TO_CLOSE_SHIFTED)),
        "synthetic_leverage_used": bool(config.get("use_synthetic_leverage", False)),
    }
    if entry.get("entry_error"):
        return [
            {
                **base,
                "symbol": "",
                "symbols_loaded_successfully": "",
                "symbol_loaded_successfully": False,
                "benchmark_available": False,
                "ohlc_required": False,
                "ohlc_available": "",
                "synthetic_real_tqqq_overlap_rows": 0,
                "cash_synthetic_series_aligned": "",
                "status": "error",
                "warnings": entry["entry_error"],
            }
        ]

    try:
        data = _load_price_data(config)
    except Exception as exc:
        return [
            {
                **base,
                "symbol": "",
                "symbols_loaded_successfully": "",
                "symbol_loaded_successfully": False,
                "benchmark_available": False,
                "ohlc_required": False,
                "ohlc_available": "",
                "synthetic_real_tqqq_overlap_rows": 0,
                "cash_synthetic_series_aligned": "",
                "status": "error",
                "warnings": str(exc),
            }
        ]

    loaded_symbols = [symbol for symbol in required if symbol in data.columns]
    benchmark_symbol = str(config.get("benchmark_symbol", "")).upper()
    benchmark_available = bool(benchmark_symbol and benchmark_symbol in data.columns and not data[benchmark_symbol].dropna().empty)
    try:
        execution_model = normalize_execution_model(config.get("execution_model", CLOSE_TO_CLOSE_SHIFTED))
    except Exception:
        execution_model = str(config.get("execution_model", CLOSE_TO_CLOSE_SHIFTED))
    ohlc_required = execution_model != CLOSE_TO_CLOSE_SHIFTED
    overlap_rows = _synthetic_overlap(config, data)
    cash_aligned = _cash_aligned(required, data)

    rows: List[Dict[str, Any]] = []
    for symbol in required:
        warnings: List[str] = []
        symbol_info = _symbol_summary(symbol, data)
        if not symbol_info["symbol_loaded_successfully"]:
            warnings.append("symbol not loaded")
        if symbol == benchmark_symbol and not benchmark_available:
            warnings.append("benchmark unavailable")
        ohlc_available: Any = ""
        if ohlc_required:
            ohlc_available = has_ohlc(data, symbol)
            if not ohlc_available:
                warnings.append("OHLC unavailable for requested execution model")
        rows.append(
            {
                **base,
                "symbol": symbol,
                "symbols_loaded_successfully": ";".join(loaded_symbols),
                **symbol_info,
                "benchmark_available": benchmark_available,
                "ohlc_required": ohlc_required,
                "ohlc_available": ohlc_available,
                "synthetic_real_tqqq_overlap_rows": overlap_rows,
                "cash_synthetic_series_aligned": cash_aligned,
                "status": "ok",
                "warnings": "; ".join(warnings),
            }
        )
    return rows


def audit_data(config_path: Path) -> pd.DataFrame:
    config_path = Path(config_path)
    parent, entries = _iter_experiment_configs(config_path)
    output_dir = _output_dir_for_parent(config_path, parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        rows.extend(_audit_one_data_entry(parent=parent, entry=entry))

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "audit_data_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nData audit written to: {output_dir / 'audit_data_summary.csv'}")
    return summary


def _read_date_indexed_csv(path: Path) -> pd.DataFrame:
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
    raw[date_column] = pd.to_datetime(raw[date_column], errors="coerce")
    if raw[date_column].isna().any():
        raise ValueError(f"Could not parse date column {date_column}: {path}")
    out = raw.set_index(date_column).sort_index()
    out.index.name = "Date"
    return out


def _date_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return pd.Timestamp(value).date().isoformat()


def _baseline_cache_path(cache_dir: str) -> Path:
    return Path(cache_dir) / "tqqq_qqq_prices.csv"


def _load_baseline_prices(
    *,
    old: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    cache_dir: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cache_path = _baseline_cache_path(cache_dir)
    cache_exists_before = cache_path.exists()
    cache_start = ""
    cache_end = ""
    cache_rows = 0
    if cache_exists_before:
        try:
            cached = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date").sort_index()
            cache_start = _date_text(cached.index.min()) if not cached.empty else ""
            cache_end = _date_text(cached.index.max()) if not cached.empty else ""
            cache_rows = int(len(cached))
        except Exception:
            cache_start = ""
            cache_end = ""
            cache_rows = 0

    old_start = pd.Timestamp(old.index.min())
    old_end = pd.Timestamp(old.index.max())
    load_start = str(pd.Timestamp(start_date).date()) if start_date else str((old_start - pd.Timedelta(days=10)).date())
    load_end = str(pd.Timestamp(end_date).date()) if end_date else str(old_end.date())

    prices = load_prices(
        start=load_start,
        end=load_end,
        symbols=("TQQQ", "QQQ"),
        use_csv_if_exists=True,
        cache_dir=cache_dir,
        include_ohlc=False,
    )
    cache_exists_after = cache_path.exists()
    yfinance_download_occurred = bool(not cache_exists_before and cache_exists_after)
    cached_data_sliced = False
    if cache_exists_before and not prices.empty:
        price_start = _date_text(prices.index.min())
        price_end = _date_text(prices.index.max())
        cached_data_sliced = bool(
            cache_rows > len(prices)
            or (cache_start and price_start and price_start > cache_start)
            or (cache_end and price_end and price_end < cache_end)
        )

    metadata = {
        "cache_path_used": str(cache_path),
        "yfinance_download_occurred": yfinance_download_occurred,
        "cached_data_sliced_by_start_end": cached_data_sliced,
        "load_start": load_start,
        "load_end": load_end,
    }
    return prices, metadata


def _numeric_series(df: pd.DataFrame, column: str, index: pd.Index) -> pd.Series:
    if column not in df.columns:
        return pd.Series(index=index, dtype=float)
    return pd.to_numeric(df.loc[index, column], errors="coerce").astype(float)


def _max_abs_return_diff(old: pd.Series, current: pd.Series) -> float:
    diff = (old - current).abs().replace([np.inf, -np.inf], np.nan).dropna()
    return float(diff.max()) if not diff.empty else np.nan


def _first_return_diff_date(old: pd.Series, current: pd.Series, tolerance: float) -> str:
    diff = (old - current).abs().replace([np.inf, -np.inf], np.nan)
    diff = diff.dropna()
    if diff.empty:
        return ""
    differing = diff[diff > float(tolerance)]
    if differing.empty:
        return ""
    return _date_text(differing.index[0])


def _build_baseline_return_comparison(
    old: pd.DataFrame,
    prices: pd.DataFrame,
    tolerance: float,
) -> Tuple[Dict[str, Any], pd.DataFrame, List[str]]:
    common_index = old.index.intersection(prices.index)
    current_tqqq = prices["TQQQ"].pct_change().reindex(common_index)
    current_qqq = prices["QQQ"].pct_change().reindex(common_index)

    old_tqqq = _numeric_series(old, "daily_ret_tqqq", common_index)
    old_qqq = _numeric_series(old, "daily_ret_qqq", common_index)
    old_daily = _numeric_series(old, "daily_ret", common_index)
    old_tqqq_weight = _numeric_series(old, "tqqq_weight", common_index)
    old_qqq_weight = _numeric_series(old, "qqq_weight", common_index)
    current_daily = old_tqqq_weight * current_tqqq + old_qqq_weight * current_qqq

    warnings: List[str] = []
    compared: List[Tuple[str, pd.Series, pd.Series]] = []
    if "daily_ret_tqqq" in old.columns:
        compared.append(("tqqq", old_tqqq, current_tqqq))
    else:
        warnings.append("old stitched file lacks daily_ret_tqqq")
    if "daily_ret_qqq" in old.columns:
        compared.append(("qqq", old_qqq, current_qqq))
    else:
        warnings.append("old stitched file lacks daily_ret_qqq")
    if "daily_ret" in old.columns and {"tqqq_weight", "qqq_weight"}.issubset(old.columns):
        compared.append(("daily_ret", old_daily, current_daily))
    elif "daily_ret" in old.columns:
        warnings.append("old stitched file has daily_ret but lacks weights needed for current weighted return")
    else:
        warnings.append("old stitched file lacks daily_ret")

    limited = not {"daily_ret_tqqq", "daily_ret_qqq"}.issubset(old.columns)
    if limited:
        warnings.append("return-level data audit is limited")

    max_tqqq_diff = _max_abs_return_diff(old_tqqq, current_tqqq) if "daily_ret_tqqq" in old.columns else np.nan
    max_qqq_diff = _max_abs_return_diff(old_qqq, current_qqq) if "daily_ret_qqq" in old.columns else np.nan
    max_daily_diff = (
        _max_abs_return_diff(old_daily, current_daily)
        if "daily_ret" in old.columns and {"tqqq_weight", "qqq_weight"}.issubset(old.columns)
        else np.nan
    )
    first_tqqq = _first_return_diff_date(old_tqqq, current_tqqq, tolerance) if "daily_ret_tqqq" in old.columns else ""
    first_qqq = _first_return_diff_date(old_qqq, current_qqq, tolerance) if "daily_ret_qqq" in old.columns else ""
    first_daily = (
        _first_return_diff_date(old_daily, current_daily, tolerance)
        if "daily_ret" in old.columns and {"tqqq_weight", "qqq_weight"}.issubset(old.columns)
        else ""
    )

    if not compared:
        match_status = "limited"
    else:
        diffs = [_max_abs_return_diff(old_series, current_series) for _, old_series, current_series in compared]
        all_match = all(pd.notna(diff) and diff <= float(tolerance) for diff in diffs)
        if not all_match:
            match_status = "no"
        elif limited:
            match_status = "limited"
        else:
            match_status = "yes"

    difference_rows = []
    for idx in common_index:
        values = {
            "date": _date_text(idx),
            "old_daily_ret_tqqq": old_tqqq.get(idx, np.nan),
            "current_daily_ret_tqqq": current_tqqq.get(idx, np.nan),
            "old_daily_ret_qqq": old_qqq.get(idx, np.nan),
            "current_daily_ret_qqq": current_qqq.get(idx, np.nan),
            "old_daily_ret": old_daily.get(idx, np.nan),
            "current_daily_ret": current_daily.get(idx, np.nan),
        }
        values["tqqq_return_diff"] = values["old_daily_ret_tqqq"] - values["current_daily_ret_tqqq"]
        values["qqq_return_diff"] = values["old_daily_ret_qqq"] - values["current_daily_ret_qqq"]
        values["daily_ret_diff"] = values["old_daily_ret"] - values["current_daily_ret"]
        material = any(
            pd.notna(values[column]) and abs(float(values[column])) > float(tolerance)
            for column in ("tqqq_return_diff", "qqq_return_diff", "daily_ret_diff")
        )
        if material:
            difference_rows.append(values)
            if len(difference_rows) >= 20:
                break

    first_differences = pd.DataFrame(difference_rows, columns=BASELINE_RETURN_COLUMNS)
    fields = {
        "common_rows": int(len(common_index)),
        "max_abs_tqqq_return_diff": max_tqqq_diff,
        "max_abs_qqq_return_diff": max_qqq_diff,
        "max_abs_daily_ret_diff": max_daily_diff,
        "first_tqqq_return_diff_date": first_tqqq,
        "first_qqq_return_diff_date": first_qqq,
        "first_daily_ret_diff_date": first_daily,
        "current_data_matches_old_baseline_returns": match_status,
        "return_level_audit_limited": limited,
        "warnings": "; ".join(dict.fromkeys(warnings)),
    }
    return fields, first_differences, warnings


def _write_baseline_data_audit_report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    first_differences: pd.DataFrame,
    old_stitched_path: Path,
) -> Path:
    report_path = output_dir / "baseline_data_audit_report.md"
    status = str(summary["current_data_matches_old_baseline_returns"].iloc[0]) if not summary.empty else "unknown"
    lines = [
        "# Baseline Price And Return Audit",
        "",
        f"- Old stitched file: `{old_stitched_path}`",
        f"- Current data matches old baseline returns: `{status}`",
        "",
        "## Summary",
        _markdown_table(summary, max_rows=5),
        "",
        "## First Return Differences",
        _markdown_table(first_differences, max_rows=20),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def audit_baseline_data(
    *,
    old_stitched_path: Path,
    output_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    cache_dir: str = "./price_cache",
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    old_stitched_path = Path(old_stitched_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old = _read_date_indexed_csv(old_stitched_path)
    if start_date:
        old = old.loc[old.index >= pd.Timestamp(start_date)].copy()
    if end_date:
        old = old.loc[old.index <= pd.Timestamp(end_date)].copy()
    if old.empty:
        raise ValueError("Old stitched file has no rows after applying start/end filters.")

    prices, metadata = _load_baseline_prices(
        old=old,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
    )
    fields, first_differences, _ = _build_baseline_return_comparison(
        old=old,
        prices=prices,
        tolerance=float(tolerance),
    )
    summary = pd.DataFrame(
        [
            {
                "old_start": _date_text(old.index.min()),
                "old_end": _date_text(old.index.max()),
                "current_price_start": _date_text(prices.index.min()) if not prices.empty else "",
                "current_price_end": _date_text(prices.index.max()) if not prices.empty else "",
                **fields,
                "cache_path_used": metadata["cache_path_used"],
                "yfinance_download_occurred": metadata["yfinance_download_occurred"],
                "cached_data_sliced_by_start_end": metadata["cached_data_sliced_by_start_end"],
                "data_load_start": metadata["load_start"],
                "data_load_end": metadata["load_end"],
                "tolerance": float(tolerance),
            }
        ]
    )

    summary.to_csv(output_dir / "baseline_data_audit_summary.csv", index=False)
    first_differences.to_csv(output_dir / "first_return_differences.csv", index=False)
    _write_baseline_data_audit_report(
        output_dir=output_dir,
        summary=summary,
        first_differences=first_differences,
        old_stitched_path=old_stitched_path,
    )

    print(summary.to_string(index=False))
    print(f"\nBaseline data audit written to: {output_dir / 'baseline_data_audit_report.md'}")
    return summary
