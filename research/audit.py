from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import CASH_SYMBOL
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


RAW_FINAL_EQUITY_OBJECTIVES = {"objective_final_ratio", "final_equity_ratio"}
RISK_FIRST_OBJECTIVES = {"sharpe", "calmar", "objective_sharpe", "objective_calmar"}
HIGH_GRID_WARNING_THRESHOLD = 10000


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

