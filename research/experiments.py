from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from .anti_overfit import run_anti_overfit_validation
from .assets import AssetConfig, asset_config_from_mapping
from .data import CASH_SYMBOL, add_cash_series, load_prices, make_synthetic_leveraged_price
from .execution import (
    CLOSE_TO_CLOSE_SHIFTED,
    EXECUTION_MODELS,
    asset_returns,
    normalize_execution_model,
    ohlc_columns,
    resolve_execution_model,
)
from .metrics import summarize_performance
from .reports import (
    compare_real_to_synthetic,
    compare_to_benchmark,
    generate_experiment_report,
    print_benchmark_summary,
    save_run_config,
    save_synthetic_tracking,
)
from .sensitivity import COST_SCENARIOS, run_cost_and_execution_sensitivity
from .strategies.core_overlay_strategy import CoreOverlayParams, make_core_overlay_param_grid
from .strategies.drawdown_governor import DrawdownGovernorParams
from .strategies.ma_strategy import Params, make_param_grid
from .strategies.market_internals import (
    DEFAULT_COMPONENTS,
    MarketInternalsParams,
    make_market_internals_param_grid,
    market_internals_availability,
    requested_market_internals_symbols,
)
from .strategies.regime_strategy import RegimeParams, make_regime_param_grid
from .strategies.rebound import ReboundParams
from .strategies.rotation_strategy import (
    DEFAULT_RISK_ASSETS,
    RotationParams,
    make_rotation_param_grid,
)
from .strategies.vol_target_strategy import VolTargetParams, make_vol_target_param_grid
from .validation import (
    walk_forward_core_overlay_search,
    walk_forward_regime_search,
    walk_forward_rotation_search,
    walk_forward_search,
    walk_forward_vol_target_search,
    walk_forward_windows,
)

REQUIRED_EXPERIMENT_FIELDS = {
    "experiment_name",
    "symbols",
    "start_date",
    "end_date",
    "strategy_name",
    "benchmark_symbol",
    "transaction_cost_bps",
    "train_years",
    "test_years",
    "objective",
    "output_dir",
}
DEFAULT_SECONDS_PER_ESTIMATED_EVALUATION = 0.002


def load_yaml_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _require_experiment_fields(config: Dict[str, Any]) -> None:
    missing = sorted(REQUIRED_EXPERIMENT_FIELDS - set(config))
    if missing:
        raise ValueError(f"Experiment config is missing required fields: {missing}")
    if not any(key in config for key in ("strategy_params", "strategy_parameters", "parameters", "parameter_grid")):
        raise ValueError("Experiment config must include strategy_params, strategy_parameters, parameters, or parameter_grid.")


def _strategy_grid_size(config: Dict[str, Any]) -> int:
    strategy_name = str(config.get("strategy_name", "")).lower()
    if strategy_name in {"buy_and_hold", "buy-hold", "buyhold"}:
        return 1
    if strategy_name in {"ma", "moving_average", "moving-average"}:
        return len(_ma_grid(config))
    if strategy_name == "regime":
        return len(_regime_grid(config))
    if strategy_name in {"core_overlay", "core-overlay", "core_overlay_tqqq", "coreoverlaytqqqstrategy"}:
        return len(_core_overlay_grid(config))
    if strategy_name in {"vol_target", "vol-target", "voltargettqqqstrategy", "vol_target_tqqq"}:
        return len(_vol_target_grid(config))
    if strategy_name in {"rotation", "cross_asset_momentum", "cross_asset_leveraged_momentum", "crossassetleveragedmomentumstrategy"}:
        return len(_rotation_grid(config))
    raise ValueError(f"Unsupported strategy_name for workload estimate: {config.get('strategy_name', '')}")


def _estimated_window_count(config: Dict[str, Any]) -> int:
    start = pd.Timestamp(config.get("start_date"))
    end_value = config.get("end_date")
    end = pd.Timestamp(end_value) if end_value else pd.Timestamp.today().normalize()
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    dates = pd.bdate_range(start, end)
    if len(dates) == 0:
        return 0
    windows = walk_forward_windows(
        dates,
        train_years=int(config.get("train_years", 5)),
        test_years=int(config.get("test_years", 1)),
        mode=str(config.get("walk_forward_mode", "rolling")),
    )
    return int(len(windows))


def _full_experiment_run_multiplier(config: Dict[str, Any]) -> int:
    sensitivity_runs = max(len(COST_SCENARIOS) - 1, 0) + max(len(EXECUTION_MODELS) - 1, 0)
    validation_config = config.get("anti_overfit_validation", {})
    anti_overfit_enabled = not (
        validation_config is False
        or (isinstance(validation_config, dict) and validation_config.get("enabled") is False)
    )
    anti_overfit_extra_runs = 5 if anti_overfit_enabled else 0
    return int(1 + sensitivity_runs + anti_overfit_extra_runs)


def estimate_experiment_workload(
    config: Dict[str, Any],
    *,
    include_full_experiment_overhead: bool = True,
) -> Dict[str, Any]:
    estimate_error = ""
    try:
        parameter_combinations = int(_strategy_grid_size(config))
    except Exception as exc:
        parameter_combinations = 0
        estimate_error = str(exc)
    try:
        window_count = int(_estimated_window_count(config))
    except Exception as exc:
        window_count = 0
        estimate_error = "; ".join(part for part in (estimate_error, str(exc)) if part)

    run_multiplier = _full_experiment_run_multiplier(config) if include_full_experiment_overhead else 1
    estimated_evaluations = int(parameter_combinations * window_count * run_multiplier)
    seconds_per_eval = float(
        config.get("estimated_seconds_per_evaluation", DEFAULT_SECONDS_PER_ESTIMATED_EVALUATION)
    )
    estimated_runtime_seconds = float(estimated_evaluations * seconds_per_eval)
    return {
        "estimated_parameter_combinations": parameter_combinations,
        "estimated_walk_forward_windows": window_count,
        "estimated_run_multiplier": run_multiplier,
        "estimated_evaluations": estimated_evaluations,
        "estimated_runtime_seconds": estimated_runtime_seconds,
        "runtime_estimate_method": "grid_size * estimated_walk_forward_windows * run_multiplier * seconds_per_evaluation",
        "runtime_estimate_error": estimate_error,
    }


def _runtime_profile_fields(
    estimate: Dict[str, Any],
    *,
    actual_runtime_seconds: float,
) -> Dict[str, Any]:
    estimated_runtime = float(estimate.get("estimated_runtime_seconds", np.nan))
    estimated_evaluations = float(estimate.get("estimated_evaluations", np.nan))
    fields = dict(estimate)
    fields.update(
        {
            "actual_runtime_seconds": float(actual_runtime_seconds),
            "runtime_estimate_error_seconds": (
                float(actual_runtime_seconds - estimated_runtime)
                if pd.notna(estimated_runtime)
                else np.nan
            ),
            "actual_seconds_per_estimated_evaluation": (
                float(actual_runtime_seconds / estimated_evaluations)
                if pd.notna(estimated_evaluations) and estimated_evaluations > 0
                else np.nan
            ),
        }
    )
    return fields


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, range):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _grid_values(config: Dict[str, Any], key: str, default: Sequence[Any]) -> List[Any]:
    return _as_list(config.get(key, default))


def _strategy_params(config: Dict[str, Any]) -> Dict[str, Any]:
    params = (
        config.get("strategy_params")
        or config.get("strategy_parameters")
        or config.get("parameters")
        or {}
    )
    if not isinstance(params, dict):
        raise ValueError("Strategy parameters must be a mapping.")
    return params


def _asset_config(config: Dict[str, Any]) -> AssetConfig:
    params = _strategy_params(config)
    risk_off_symbol = (
        params.get("risk_off_symbol")
        or config.get("risk_off_symbol")
        or "CASH"
    )
    values: Dict[str, Any] = {}
    if isinstance(config.get("asset_config"), dict):
        values.update(config["asset_config"])
    for key in (
        "trade_asset",
        "primary_signal_asset",
        "secondary_filter_asset",
        "filter_asset",
    ):
        if key in config:
            values[key] = config[key]
        if key in params:
            values[key] = params[key]
    return asset_config_from_mapping(
        values,
        benchmark_symbol=str(config.get("benchmark_symbol", "TQQQ")),
        risk_off_symbol=str(risk_off_symbol),
    ).normalized()


def _asset_config_symbols_from_config(config: Dict[str, Any]) -> List[str]:
    asset_config = _asset_config(config)
    symbols = [
        asset_config.trade_asset,
        asset_config.primary_signal_asset,
        asset_config.secondary_filter_asset,
        asset_config.benchmark_symbol,
        asset_config.risk_off_symbol,
    ]
    extra = config.get("benchmark_comparison_symbols", config.get("additional_benchmark_symbols", []))
    symbols.extend(str(symbol).upper() for symbol in _as_list(extra))
    return list(dict.fromkeys(symbol for symbol in symbols if symbol and symbol != CASH_SYMBOL))


def _risk_off_symbols_from_config(config: Dict[str, Any]) -> List[str]:
    symbols: List[str] = []
    params = _strategy_params(config)
    if "risk_off_symbol" in params:
        symbols.append(str(params["risk_off_symbol"]).upper())

    grid = config.get("parameter_grid")
    if isinstance(grid, dict):
        for key in ("risk_off_symbol", "risk_off_symbols"):
            if key in grid:
                symbols.extend(str(value).upper() for value in _as_list(grid[key]))

    asset_risk_off = _asset_config(config).risk_off_symbol
    if asset_risk_off != CASH_SYMBOL:
        symbols.append(asset_risk_off)

    return list(dict.fromkeys(symbols))


def _market_internals_enabled_from_config(config: Dict[str, Any]) -> bool:
    params = _strategy_params(config)
    if bool(params.get("use_market_internals", False)):
        return True
    grid = config.get("parameter_grid")
    if isinstance(grid, dict):
        values = (
            grid.get("use_market_internals")
            or grid.get("use_market_internals_options")
            or []
        )
        return any(bool(value) for value in _as_list(values))
    return bool(config.get("use_market_internals", False))


def _market_internals_symbols_from_config(config: Dict[str, Any]) -> List[str]:
    params = _strategy_params(config)
    grid = config.get("parameter_grid")
    explicit = (
        config.get("market_internals_symbols")
        or params.get("market_internals_symbols")
        or (
            grid.get("market_internals_symbols")
            if isinstance(grid, dict)
            else None
        )
    )
    if explicit is None and not _market_internals_enabled_from_config(config):
        return []
    return requested_market_internals_symbols(_as_list(explicit) if explicit is not None else None)


def _component_sets_from_config(value: Any) -> List[Sequence[str]]:
    if value is None:
        return [DEFAULT_COMPONENTS]
    if isinstance(value, str):
        return [tuple(part.strip() for part in value.split(",") if part.strip())]
    if isinstance(value, (list, tuple)):
        if not value:
            return [DEFAULT_COMPONENTS]
        if all(not isinstance(item, (list, tuple)) for item in value):
            return [tuple(str(item) for item in value)]
        return [tuple(str(part) for part in item) for item in value]
    raise ValueError("market_internals_components must be a string, list, or list of lists.")


def _weights_options_from_config(value: Any) -> List[Dict[str, float]]:
    if value is None:
        return [{}]
    if isinstance(value, dict):
        return [{str(key): float(weight) for key, weight in value.items()}]
    if isinstance(value, list):
        parsed = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("market_internals_weights list entries must be mappings.")
            parsed.append({str(key): float(weight) for key, weight in item.items()})
        return parsed or [{}]
    raise ValueError("market_internals_weights must be a mapping or list of mappings.")


def _market_internals_grid(grid_config: Dict[str, Any]) -> List[MarketInternalsParams]:
    use_options = [
        bool(value)
        for value in _grid_values(
            grid_config,
            "use_market_internals",
            grid_config.get("use_market_internals_options", (False,)),
        )
    ]
    signal_windows = [
        int(value)
        for value in _grid_values(
            grid_config,
            "market_internals_signal_window",
            grid_config.get(
                "market_internals_signal_windows",
                grid_config.get("signal_windows", (100,)),
            ),
        )
    ]
    risk_on_thresholds = [
        float(value)
        for value in _grid_values(
            grid_config,
            "market_internals_risk_on_threshold",
            grid_config.get(
                "market_internals_risk_on_thresholds",
                grid_config.get("risk_on_threshold", grid_config.get("risk_on_thresholds", (0.60,))),
            ),
        )
    ]
    risk_off_thresholds = [
        float(value)
        for value in _grid_values(
            grid_config,
            "market_internals_risk_off_threshold",
            grid_config.get(
                "market_internals_risk_off_thresholds",
                grid_config.get("risk_off_threshold", grid_config.get("risk_off_thresholds", (0.40,))),
            ),
        )
    ]
    return make_market_internals_param_grid(
        use_market_internals_options=use_options,
        signal_windows=signal_windows,
        risk_on_thresholds=risk_on_thresholds,
        risk_off_thresholds=risk_off_thresholds,
        component_sets=_component_sets_from_config(grid_config.get("market_internals_components")),
        weights_options=_weights_options_from_config(grid_config.get("market_internals_weights")),
        score_methods=[
            str(value)
            for value in _grid_values(
                grid_config,
                "market_internals_score_method",
                grid_config.get("market_internals_score_methods", ("binary",)),
            )
        ],
        min_components_options=[
            int(value)
            for value in _grid_values(
                grid_config,
                "market_internals_min_components",
                grid_config.get("market_internals_min_components_options", (1,)),
            )
        ],
        vix_risk_on_thresholds=[
            float(value)
            for value in _grid_values(
                grid_config,
                "vix_risk_on_threshold",
                grid_config.get("vix_risk_on_thresholds", (25.0,)),
            )
        ],
        vix_risk_off_thresholds=[
            float(value)
            for value in _grid_values(
                grid_config,
                "vix_risk_off_threshold",
                grid_config.get("vix_risk_off_thresholds", (35.0,)),
            )
        ],
    )


def _market_internals_params_from_mapping(params: Dict[str, Any]) -> MarketInternalsParams:
    components = _component_sets_from_config(
        params.get("market_internals_components", params.get("components"))
    )[0]
    weights = _weights_options_from_config(
        params.get("market_internals_weights", params.get("weights"))
    )[0]
    return MarketInternalsParams(
        use_market_internals=bool(params.get("use_market_internals", False)),
        signal_window=int(params.get("market_internals_signal_window", params.get("signal_window", 100))),
        risk_on_threshold=float(
            params.get("market_internals_risk_on_threshold", params.get("risk_on_threshold", 0.60))
        ),
        risk_off_threshold=float(
            params.get("market_internals_risk_off_threshold", params.get("risk_off_threshold", 0.40))
        ),
        score_method=str(params.get("market_internals_score_method", params.get("score_method", "binary"))),
        components=tuple(components),
        weights=weights,
        min_components=int(params.get("market_internals_min_components", params.get("min_components", 1))),
        vix_risk_on_threshold=float(params.get("vix_risk_on_threshold", 25.0)),
        vix_risk_off_threshold=float(params.get("vix_risk_off_threshold", 35.0)),
    )


def _market_internals_max_required_window(config: Dict[str, Any]) -> int:
    grid = config.get("parameter_grid")
    windows = [200]
    if isinstance(grid, dict):
        windows.extend(
            int(value)
            for value in _grid_values(
                grid,
                "market_internals_signal_window",
                grid.get("market_internals_signal_windows", grid.get("signal_windows", (100,))),
            )
        )
    else:
        params = _strategy_params(config)
        windows.append(int(params.get("market_internals_signal_window", params.get("signal_window", 100))))
    return max(windows)


def _is_rotation_strategy_name(strategy_name: Any) -> bool:
    return str(strategy_name).lower() in {
        "rotation",
        "cross_asset_rotation",
        "cross-asset-rotation",
        "cross_asset_leveraged_momentum",
        "crossassetleveragedmomentumstrategy",
    }


def _asset_sets_from_config(value: Any, default: Sequence[str]) -> List[Sequence[str]]:
    if value is None:
        return [tuple(default)]
    if isinstance(value, str):
        return [tuple(part.strip().upper() for part in value.split(",") if part.strip())]
    if isinstance(value, (list, tuple)):
        if not value:
            return [tuple(default)]
        if all(not isinstance(item, (list, tuple)) for item in value):
            return [tuple(str(item).upper() for item in value)]
        return [tuple(str(part).upper() for part in item) for item in value]
    raise ValueError("risk_assets must be a string, list, or list of lists.")


def _rotation_symbols_from_config(config: Dict[str, Any]) -> List[str]:
    if not _is_rotation_strategy_name(config.get("strategy_name")):
        return []
    params = _strategy_params(config)
    grid = config.get("parameter_grid")
    risk_value = None
    defensive_value = None
    trend_value = None
    if isinstance(grid, dict):
        risk_value = grid.get("risk_assets", grid.get("risk_asset_sets"))
        defensive_value = grid.get("defensive_asset", grid.get("defensive_assets"))
        trend_value = grid.get("trend_filter_symbol", grid.get("trend_filter_symbols"))
    risk_value = params.get("risk_assets", risk_value if risk_value is not None else DEFAULT_RISK_ASSETS)
    defensive_value = params.get("defensive_asset", defensive_value if defensive_value is not None else "CASH")
    trend_value = params.get("trend_filter_symbol", trend_value if trend_value is not None else "QQQ")

    symbols: List[str] = []
    for risk_assets in _asset_sets_from_config(risk_value, DEFAULT_RISK_ASSETS):
        symbols.extend(str(symbol).upper() for symbol in risk_assets)
    symbols.extend(str(symbol).upper() for symbol in _as_list(defensive_value))
    symbols.extend(str(symbol).upper() for symbol in _as_list(trend_value))
    return list(dict.fromkeys(symbols))


def _rotation_required_symbols_from_config(config: Dict[str, Any]) -> List[str]:
    if not _is_rotation_strategy_name(config.get("strategy_name")):
        return []
    params = _strategy_params(config)
    grid = config.get("parameter_grid")
    trend_value = params.get("trend_filter_symbol")
    if isinstance(grid, dict):
        trend_value = trend_value if trend_value is not None else grid.get("trend_filter_symbol", grid.get("trend_filter_symbols"))
    trend_value = trend_value if trend_value is not None else "QQQ"
    required = [
        str(symbol).upper()
        for symbol in _as_list(trend_value)
        if str(symbol).upper() != CASH_SYMBOL
    ]
    return list(dict.fromkeys(required))


def _rotation_asset_status(config: Dict[str, Any], data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    symbols = _rotation_symbols_from_config(config)
    if not symbols:
        return None
    skipped = []
    available = []
    for symbol in symbols:
        if symbol == CASH_SYMBOL or symbol in data.columns:
            available.append(symbol)
        else:
            skipped.append({"symbol": symbol, "reason": "missing"})
    return {
        "requested_symbols": symbols,
        "available_symbols": available,
        "skipped_symbols": skipped,
    }


def _rebound_grid_kwargs(grid_config: Dict[str, Any]) -> Dict[str, List[Any]]:
    return {
        "use_rebound_options": [
            bool(v)
            for v in _grid_values(
                grid_config,
                "use_rebound_module",
                grid_config.get("use_rebound_options", (False,)),
            )
        ],
        "rolling_high_windows": [
            int(v)
            for v in _grid_values(
                grid_config,
                "rolling_high_window",
                grid_config.get("rolling_high_windows", (126,)),
            )
        ],
        "drawdown_triggers": [
            float(v)
            for v in _grid_values(
                grid_config,
                "drawdown_trigger",
                grid_config.get("drawdown_triggers", (-0.20,)),
            )
        ],
        "rebound_momentum_windows": [
            int(v)
            for v in _grid_values(
                grid_config,
                "rebound_momentum_window",
                grid_config.get("rebound_momentum_windows", (10,)),
            )
        ],
        "rebound_momentum_thresholds": [
            float(v)
            for v in _grid_values(
                grid_config,
                "rebound_momentum_threshold",
                grid_config.get("rebound_momentum_thresholds", (0.05,)),
            )
        ],
        "reclaim_ma_windows": [
            int(v)
            for v in _grid_values(
                grid_config,
                "reclaim_ma_window",
                grid_config.get("reclaim_ma_windows", (20,)),
            )
        ],
        "rebound_positions": [
            float(v)
            for v in _grid_values(
                grid_config,
                "rebound_position",
                grid_config.get("rebound_positions", (1.0,)),
            )
        ],
        "rebound_hold_days_options": [
            int(v)
            for v in _grid_values(
                grid_config,
                "rebound_hold_days",
                grid_config.get("rebound_hold_days_options", (10,)),
            )
        ],
        "extreme_crash_vol_windows": [
            int(v)
            for v in _grid_values(
                grid_config,
                "extreme_crash_vol_window",
                grid_config.get("extreme_crash_vol_windows", (20,)),
            )
        ],
        "extreme_crash_vol_thresholds": [
            float(v)
            for v in _grid_values(
                grid_config,
                "extreme_crash_vol_threshold",
                grid_config.get("extreme_crash_vol_thresholds", (0.05,)),
            )
        ],
    }


def _rebound_params_from_mapping(params: Dict[str, Any]) -> ReboundParams:
    return ReboundParams(
        use_rebound_module=bool(params.get("use_rebound_module", False)),
        rolling_high_window=int(params.get("rolling_high_window", 126)),
        drawdown_trigger=float(params.get("drawdown_trigger", -0.20)),
        rebound_momentum_window=int(params.get("rebound_momentum_window", 10)),
        rebound_momentum_threshold=float(params.get("rebound_momentum_threshold", 0.05)),
        reclaim_ma_window=int(params.get("reclaim_ma_window", 20)),
        rebound_position=float(params.get("rebound_position", 1.0)),
        rebound_hold_days=int(params.get("rebound_hold_days", 10)),
        extreme_crash_vol_window=int(params.get("extreme_crash_vol_window", 20)),
        extreme_crash_vol_threshold=float(params.get("extreme_crash_vol_threshold", 0.05)),
    )


def _governor_grid_kwargs(grid_config: Dict[str, Any]) -> Dict[str, List[Any]]:
    return {
        "use_drawdown_governor_options": [
            bool(v)
            for v in _grid_values(
                grid_config,
                "use_drawdown_governor",
                grid_config.get("use_drawdown_governor_options", (False,)),
            )
        ],
        "portfolio_dd_triggers": [
            float(v)
            for v in _grid_values(
                grid_config,
                "portfolio_dd_trigger",
                grid_config.get("portfolio_dd_triggers", (-0.30,)),
            )
        ],
        "qqq_dd_triggers": [
            float(v)
            for v in _grid_values(
                grid_config,
                "qqq_dd_trigger",
                grid_config.get("qqq_dd_triggers", (-0.15,)),
            )
        ],
        "reduced_exposures": [
            float(v)
            for v in _grid_values(
                grid_config,
                "reduced_exposure",
                grid_config.get("reduced_exposures", (0.50,)),
            )
        ],
        "recovery_ma_windows": [
            int(v)
            for v in _grid_values(
                grid_config,
                "recovery_ma_window",
                grid_config.get("recovery_ma_windows", (20,)),
            )
        ],
        "recovery_momentum_windows": [
            int(v)
            for v in _grid_values(
                grid_config,
                "recovery_momentum_window",
                grid_config.get("recovery_momentum_windows", (10,)),
            )
        ],
        "recovery_momentum_thresholds": [
            float(v)
            for v in _grid_values(
                grid_config,
                "recovery_momentum_threshold",
                grid_config.get("recovery_momentum_thresholds", (0.05,)),
            )
        ],
        "max_days_reduced_options": [
            int(v)
            for v in _grid_values(
                grid_config,
                "max_days_reduced",
                grid_config.get("max_days_reduced_options", (40,)),
            )
        ],
    }


def _governor_params_from_mapping(params: Dict[str, Any]) -> DrawdownGovernorParams:
    return DrawdownGovernorParams(
        use_drawdown_governor=bool(params.get("use_drawdown_governor", False)),
        portfolio_dd_trigger=float(params.get("portfolio_dd_trigger", -0.30)),
        qqq_dd_trigger=float(params.get("qqq_dd_trigger", -0.15)),
        reduced_exposure=float(params.get("reduced_exposure", 0.50)),
        recovery_ma_window=int(params.get("recovery_ma_window", 20)),
        recovery_momentum_window=int(params.get("recovery_momentum_window", 10)),
        recovery_momentum_threshold=float(params.get("recovery_momentum_threshold", 0.05)),
        max_days_reduced=int(params.get("max_days_reduced", 40)),
    )


def _risk_off_grid_kwargs(
    grid_config: Dict[str, Any],
    default_symbol: str = "CASH",
    default_weight: Any = 0.0,
) -> Dict[str, List[Any]]:
    symbols = [
        str(v).upper()
        for v in _grid_values(
            grid_config,
            "risk_off_symbol",
            grid_config.get("risk_off_symbols", (default_symbol,)),
        )
    ]
    weights = _grid_values(
        grid_config,
        "risk_off_weight",
        grid_config.get("risk_off_weights", (default_weight,)),
    )
    parsed_weights = [
        None if value is None else float(value)
        for value in weights
    ]
    return {
        "risk_off_symbols": symbols,
        "risk_off_weights": parsed_weights,
    }


def _risk_off_params_from_mapping(
    params: Dict[str, Any],
    default_symbol: str = "CASH",
    default_weight: Any = 0.0,
) -> Dict[str, Any]:
    weight = params.get("risk_off_weight", default_weight)
    return {
        "risk_off_symbol": str(params.get("risk_off_symbol", default_symbol)).upper(),
        "risk_off_weight": None if weight is None else float(weight),
    }


def _load_price_data(config: Dict[str, Any]) -> pd.DataFrame:
    use_synthetic = bool(config.get("use_synthetic_leverage", False))
    symbols = [str(symbol).upper() for symbol in _as_list(config["symbols"])]
    synthetic_base_symbol = str(config.get("synthetic_base_symbol", "QQQ")).upper()
    synthetic_target_symbol = str(config.get("synthetic_target_symbol", "TQQQ")).upper()
    real_leveraged_symbol = str(config.get("real_leveraged_symbol", synthetic_target_symbol)).upper()
    start_date = pd.Timestamp(config["start_date"])
    end_value = config.get("end_date")
    end_date = pd.Timestamp(end_value) if end_value else None
    load_symbols = list(dict.fromkeys(symbols))
    required_symbols = {str(config["benchmark_symbol"]).upper()}
    for symbol in _asset_config_symbols_from_config(config):
        if symbol not in load_symbols:
            load_symbols.append(symbol)
        if symbol != CASH_SYMBOL:
            required_symbols.add(symbol)
    for symbol in _risk_off_symbols_from_config(config):
        if symbol not in load_symbols:
            load_symbols.append(symbol)
        if symbol != CASH_SYMBOL:
            required_symbols.add(symbol)
    market_symbols = _market_internals_symbols_from_config(config)
    for symbol in market_symbols:
        if symbol not in load_symbols:
            load_symbols.append(symbol)
    rotation_symbols = _rotation_symbols_from_config(config)
    for symbol in rotation_symbols:
        if symbol not in load_symbols:
            load_symbols.append(symbol)
    for symbol in _rotation_required_symbols_from_config(config):
        required_symbols.add(symbol)
    if use_synthetic:
        for symbol in (synthetic_base_symbol, real_leveraged_symbol):
            if symbol not in load_symbols:
                load_symbols.append(symbol)
            required_symbols.add(symbol)
        required_symbols.discard(synthetic_target_symbol)
        required_symbols.discard(real_leveraged_symbol)
        required_symbols.add(synthetic_base_symbol)
    required_symbols.discard(CASH_SYMBOL)
    optional_symbols = {
        symbol
        for symbol in load_symbols
        if symbol != CASH_SYMBOL and symbol not in required_symbols
    }

    data_csv = config.get("data_csv")
    include_ohlc = bool(config.get("download_ohlc", True))
    if data_csv:
        csv_path = Path(data_csv)
        data = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
        missing = required_symbols - set(data.columns)
        if missing:
            raise ValueError(f"data_csv is missing required symbols: {sorted(missing)}")
        columns = []
        for symbol in load_symbols:
            if symbol in data.columns:
                columns.append(symbol)
            columns.extend(column for column in ohlc_columns(symbol) if column in data.columns)
        columns = list(dict.fromkeys(columns))
        data = data[columns].sort_index()
        if CASH_SYMBOL in load_symbols:
            data = add_cash_series(data, include_ohlc=include_ohlc)
            columns = []
            for symbol in load_symbols:
                if symbol in data.columns:
                    columns.append(symbol)
                columns.extend(column for column in ohlc_columns(symbol) if column in data.columns)
            data = data[list(dict.fromkeys(columns))]
        if not use_synthetic:
            data = data.dropna(subset=[symbol for symbol in required_symbols if symbol in data.columns])
    else:
        data = load_prices(
            start=str(config["start_date"]),
            end=str(end_value) if end_value else None,
            symbols=load_symbols,
            use_csv_if_exists=bool(config.get("use_csv_if_exists", True)),
            cache_dir=str(config.get("cache_dir", "./price_cache")),
            dropna=not use_synthetic and not (market_symbols or rotation_symbols or optional_symbols),
            allow_missing_symbols=bool(market_symbols or rotation_symbols or optional_symbols),
            include_ohlc=include_ohlc,
        )
        missing = required_symbols - set(data.columns)
        if missing:
            raise ValueError(f"Downloaded price data is missing required symbols: {sorted(missing)}")
        if not use_synthetic and (market_symbols or rotation_symbols):
            data = data.dropna(subset=[symbol for symbol in required_symbols if symbol in data.columns])

    if use_synthetic and not data_csv:
        base_history = load_prices(
            start=str(config["start_date"]),
            end=str(end_value) if end_value else None,
            symbols=[synthetic_base_symbol],
            use_csv_if_exists=bool(config.get("use_csv_if_exists", True)),
            cache_dir=str(config.get("cache_dir", "./price_cache")),
            dropna=True,
            allow_missing_symbols=False,
            include_ohlc=include_ohlc,
        )
        union_index = data.index.union(base_history.index)
        data = data.reindex(union_index).sort_index()
        for column in base_history.columns:
            if column in data.columns:
                data[column] = data[column].combine_first(base_history[column])
            else:
                data[column] = base_history[column].reindex(data.index)

    data = data.loc[data.index >= start_date].copy()
    if end_date is not None:
        data = data.loc[data.index <= end_date].copy()
    if use_synthetic:
        data = _apply_synthetic_leverage(
            data=data,
            base_symbol=synthetic_base_symbol,
            target_symbol=synthetic_target_symbol,
            real_symbol=real_leveraged_symbol,
            leverage=float(config.get("synthetic_leverage", 3.0)),
            expense_ratio=float(config.get("synthetic_expense_ratio", 0.0095)),
            financing_spread=float(config.get("synthetic_financing_spread", 0.0)),
        )
    elif CASH_SYMBOL in load_symbols and CASH_SYMBOL not in data.columns:
        data = add_cash_series(data, include_ohlc=include_ohlc)
        columns = []
        for symbol in load_symbols:
            if symbol in data.columns:
                columns.append(symbol)
            columns.extend(column for column in ohlc_columns(symbol) if column in data.columns)
        data = data[list(dict.fromkeys(columns))]
    if data.empty:
        raise ValueError("No price data available after applying start_date/end_date filters.")
    return data


def _apply_synthetic_leverage(
    data: pd.DataFrame,
    base_symbol: str,
    target_symbol: str,
    real_symbol: str,
    leverage: float,
    expense_ratio: float,
    financing_spread: float,
) -> pd.DataFrame:
    if base_symbol not in data.columns:
        raise ValueError(f"Synthetic base symbol is missing from price data: {base_symbol}")

    out = data.copy()
    if real_symbol in out.columns:
        out[f"REAL_{real_symbol}"] = out[real_symbol]

    synthetic_price = make_synthetic_leveraged_price(
        base_prices=out[base_symbol],
        leverage=leverage,
        annual_expense_ratio=expense_ratio,
        annual_financing_spread=financing_spread,
    )
    out[target_symbol] = synthetic_price.reindex(out.index)
    out = out.dropna(subset=[base_symbol, target_symbol]).copy()
    return out


def _ma_grid(config: Dict[str, Any]) -> List[Params]:
    asset_config = _asset_config(config)
    grid_config = config.get("parameter_grid")
    if grid_config is not None:
        if not isinstance(grid_config, dict):
            raise ValueError("parameter_grid must be a mapping.")
        return make_param_grid(
            short_range=[int(v) for v in _grid_values(grid_config, "short_range", [])],
            long_range=[int(v) for v in _grid_values(grid_config, "long_range", [])],
            ma_types=[str(v) for v in _grid_values(grid_config, "ma_types", ("sma", "ema"))],
            signal_assets=[str(v) for v in _grid_values(grid_config, "signal_assets", ("TQQQ", "QQQ"))],
            thresholds=[float(v) for v in _grid_values(grid_config, "thresholds", (0.0,))],
            cooldown_days_options=[
                int(v) for v in _grid_values(grid_config, "cooldown_days_options", (0,))
            ],
            use_qqq_filter_options=[
                bool(v)
                for v in _grid_values(
                    grid_config,
                    "qqq_filter_options",
                    grid_config.get("use_qqq_filter_options", (False, True)),
                )
            ],
            qqq_filter_windows=[
                int(v) for v in _grid_values(grid_config, "qqq_filter_windows", (200,))
            ],
            **_risk_off_grid_kwargs(grid_config, default_symbol="CASH", default_weight=0.0),
            asset_config=asset_config,
        )

    params = _strategy_params(config)
    risk_off = _risk_off_params_from_mapping(params, default_symbol="CASH", default_weight=0.0)
    return [
        Params(
            short=int(params["short"]),
            long=int(params["long"]),
            ma_type=str(params["ma_type"]),
            signal_asset=str(params.get("signal_asset", "TQQQ")),
            threshold=float(params.get("threshold", 0.0)),
            cooldown_days=int(params.get("cooldown_days", 0)),
            qqq_filter=bool(params.get("qqq_filter", False)),
            qqq_filter_window=int(params.get("qqq_filter_window", 200)),
            risk_off_symbol=risk_off["risk_off_symbol"],
            risk_off_weight=float(risk_off["risk_off_weight"]),
            asset_config=asset_config,
        )
    ]


def _regime_grid(config: Dict[str, Any]) -> List[RegimeParams]:
    asset_config = _asset_config(config)
    grid_config = config.get("parameter_grid")
    if grid_config is not None:
        if not isinstance(grid_config, dict):
            raise ValueError("parameter_grid must be a mapping.")
        return make_regime_param_grid(
            trend_windows=[int(v) for v in _grid_values(grid_config, "trend_windows", [])],
            momentum_windows=[int(v) for v in _grid_values(grid_config, "momentum_windows", [])],
            vol_windows=[int(v) for v in _grid_values(grid_config, "vol_windows", [])],
            trend_on_levels=[float(v) for v in _grid_values(grid_config, "trend_on_levels", [])],
            trend_off_levels=[float(v) for v in _grid_values(grid_config, "trend_off_levels", [])],
            mom_on_levels=[float(v) for v in _grid_values(grid_config, "mom_on_levels", [])],
            mom_off_levels=[float(v) for v in _grid_values(grid_config, "mom_off_levels", [])],
            vol_caps=[float(v) for v in _grid_values(grid_config, "vol_caps", [])],
            risk_on_leverages=[
                float(v) for v in _grid_values(grid_config, "risk_on_leverages", (1.0,))
            ],
            risk_off_qqq_positions=[
                float(v) for v in _grid_values(grid_config, "risk_off_qqq_positions", (0.0,))
            ],
            transition_positions=[
                float(v) for v in _grid_values(grid_config, "transition_positions", (0.35,))
            ],
            min_hold_days_options=[
                int(v) for v in _grid_values(grid_config, "min_hold_days_options", (0,))
            ],
            cooldown_days_options=[
                int(v) for v in _grid_values(grid_config, "cooldown_days_options", (2,))
            ],
            **_rebound_grid_kwargs(grid_config),
            **_governor_grid_kwargs(grid_config),
            **_risk_off_grid_kwargs(grid_config, default_symbol="QQQ", default_weight=None),
            market_internals_options=_market_internals_grid(grid_config),
            asset_config=asset_config,
        )

    params = _strategy_params(config)
    risk_off = _risk_off_params_from_mapping(params, default_symbol="QQQ", default_weight=None)
    return [
        RegimeParams(
            trend_window=int(params["trend_window"]),
            momentum_window=int(params["momentum_window"]),
            vol_window=int(params["vol_window"]),
            trend_on=float(params["trend_on"]),
            trend_off=float(params["trend_off"]),
            mom_on=float(params["mom_on"]),
            mom_off=float(params["mom_off"]),
            vol_cap=float(params["vol_cap"]),
            risk_on_leverage=float(params.get("risk_on_leverage", 1.0)),
            risk_off_qqq_position=float(params.get("risk_off_qqq_position", 0.0)),
            transition_position=float(params.get("transition_position", 0.35)),
            min_hold_days=int(params.get("min_hold_days", 0)),
            cooldown_days=int(params.get("cooldown_days", 2)),
            rebound=_rebound_params_from_mapping(params),
            governor=_governor_params_from_mapping(params),
            risk_off_symbol=risk_off["risk_off_symbol"],
            risk_off_weight=risk_off["risk_off_weight"],
            market_internals=_market_internals_params_from_mapping(params),
            asset_config=asset_config,
        )
    ]


def _core_overlay_grid(config: Dict[str, Any]) -> List[CoreOverlayParams]:
    asset_config = _asset_config(config)
    grid_config = config.get("parameter_grid")
    if grid_config is not None:
        if not isinstance(grid_config, dict):
            raise ValueError("parameter_grid must be a mapping.")
        return make_core_overlay_param_grid(
            core_exposures=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "core_exposure",
                    grid_config.get("core_exposures", []),
                )
            ],
            overlay_maxes=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "overlay_max",
                    grid_config.get("overlay_maxes", []),
                )
            ],
            max_total_exposures=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "max_total_exposure",
                    grid_config.get("max_total_exposures", []),
                )
            ],
            trend_windows=[int(v) for v in _grid_values(grid_config, "trend_window", grid_config.get("trend_windows", []))],
            fast_trend_windows=[
                int(v)
                for v in _grid_values(
                    grid_config,
                    "fast_trend_window",
                    grid_config.get("fast_trend_windows", []),
                )
            ],
            momentum_windows=[
                int(v)
                for v in _grid_values(
                    grid_config,
                    "momentum_window",
                    grid_config.get("momentum_windows", []),
                )
            ],
            vol_windows=[int(v) for v in _grid_values(grid_config, "vol_window", grid_config.get("vol_windows", []))],
            vol_caps=[float(v) for v in _grid_values(grid_config, "vol_cap", grid_config.get("vol_caps", []))],
            crash_cut_exposures=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "crash_cut_exposure",
                    grid_config.get("crash_cut_exposures", []),
                )
            ],
            rebound_boost_options=[
                bool(v)
                for v in _grid_values(
                    grid_config,
                    "rebound_boost",
                    grid_config.get("rebound_boost_options", []),
                )
            ],
            **_rebound_grid_kwargs(grid_config),
            **_governor_grid_kwargs(grid_config),
            **_risk_off_grid_kwargs(grid_config, default_symbol="CASH", default_weight=0.0),
            market_internals_options=_market_internals_grid(grid_config),
            asset_config=asset_config,
        )

    params = _strategy_params(config)
    risk_off = _risk_off_params_from_mapping(params, default_symbol="CASH", default_weight=0.0)
    return [
        CoreOverlayParams(
            core_exposure=float(params["core_exposure"]),
            overlay_max=float(params["overlay_max"]),
            max_total_exposure=float(params["max_total_exposure"]),
            trend_window=int(params["trend_window"]),
            fast_trend_window=int(params["fast_trend_window"]),
            momentum_window=int(params["momentum_window"]),
            vol_window=int(params["vol_window"]),
            vol_cap=float(params["vol_cap"]),
            crash_cut_exposure=float(params["crash_cut_exposure"]),
            rebound_boost=bool(params.get("rebound_boost", False)),
            rebound=_rebound_params_from_mapping(params),
            governor=_governor_params_from_mapping(params),
            risk_off_symbol=risk_off["risk_off_symbol"],
            risk_off_weight=float(risk_off["risk_off_weight"]),
            market_internals=_market_internals_params_from_mapping(params),
            asset_config=asset_config,
        )
    ]


def _vol_target_grid(config: Dict[str, Any]) -> List[VolTargetParams]:
    asset_config = _asset_config(config)
    grid_config = config.get("parameter_grid")
    if grid_config is not None:
        if not isinstance(grid_config, dict):
            raise ValueError("parameter_grid must be a mapping.")
        return make_vol_target_param_grid(
            target_ann_vols=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "target_ann_vol",
                    grid_config.get("target_ann_vols", []),
                )
            ],
            realized_vol_windows=[
                int(v)
                for v in _grid_values(
                    grid_config,
                    "realized_vol_window",
                    grid_config.get("realized_vol_windows", []),
                )
            ],
            min_exposures=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "min_exposure",
                    grid_config.get("min_exposures", []),
                )
            ],
            max_exposures=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "max_exposure",
                    grid_config.get("max_exposures", []),
                )
            ],
            trend_windows=[
                int(v)
                for v in _grid_values(
                    grid_config,
                    "trend_window",
                    grid_config.get("trend_windows", []),
                )
            ],
            momentum_windows=[
                int(v)
                for v in _grid_values(
                    grid_config,
                    "momentum_window",
                    grid_config.get("momentum_windows", []),
                )
            ],
            trend_multiplier_below_ma_options=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "trend_multiplier_below_ma",
                    grid_config.get("trend_multiplier_below_ma_options", []),
                )
            ],
            momentum_boosts=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "momentum_boost",
                    grid_config.get("momentum_boosts", []),
                )
            ],
            crash_vol_cutoffs=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "crash_vol_cutoff",
                    grid_config.get("crash_vol_cutoffs", []),
                )
            ],
            crash_exposures=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "crash_exposure",
                    grid_config.get("crash_exposures", []),
                )
            ],
            **_rebound_grid_kwargs(grid_config),
            **_governor_grid_kwargs(grid_config),
            **_risk_off_grid_kwargs(grid_config, default_symbol="CASH", default_weight=0.0),
            market_internals_options=_market_internals_grid(grid_config),
            asset_config=asset_config,
        )

    params = _strategy_params(config)
    risk_off = _risk_off_params_from_mapping(params, default_symbol="CASH", default_weight=0.0)
    return [
        VolTargetParams(
            target_ann_vol=float(params["target_ann_vol"]),
            realized_vol_window=int(params["realized_vol_window"]),
            min_exposure=float(params["min_exposure"]),
            max_exposure=float(params["max_exposure"]),
            trend_window=int(params["trend_window"]),
            momentum_window=int(params["momentum_window"]),
            trend_multiplier_below_ma=float(params["trend_multiplier_below_ma"]),
            momentum_boost=float(params["momentum_boost"]),
            crash_vol_cutoff=float(params["crash_vol_cutoff"]),
            crash_exposure=float(params["crash_exposure"]),
            rebound=_rebound_params_from_mapping(params),
            governor=_governor_params_from_mapping(params),
            risk_off_symbol=risk_off["risk_off_symbol"],
            risk_off_weight=float(risk_off["risk_off_weight"]),
            market_internals=_market_internals_params_from_mapping(params),
            asset_config=asset_config,
        )
    ]


def _rotation_grid(config: Dict[str, Any]) -> List[RotationParams]:
    asset_config = _asset_config(config)
    grid_config = config.get("parameter_grid")
    if grid_config is not None:
        if not isinstance(grid_config, dict):
            raise ValueError("parameter_grid must be a mapping.")
        risk_asset_sets = _asset_sets_from_config(
            grid_config.get("risk_assets", grid_config.get("risk_asset_sets")),
            DEFAULT_RISK_ASSETS,
        )
        return make_rotation_param_grid(
            risk_asset_sets=risk_asset_sets,
            rebalance_frequencies=[
                str(v).lower()
                for v in _grid_values(
                    grid_config,
                    "rebalance_frequency",
                    grid_config.get("rebalance_frequencies", ("monthly",)),
                )
            ],
            top_ns=[
                int(v)
                for v in _grid_values(
                    grid_config,
                    "top_n",
                    grid_config.get("top_ns", (1,)),
                )
            ],
            max_asset_weights=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "max_asset_weight",
                    grid_config.get("max_asset_weights", (1.0,)),
                )
            ],
            max_total_leveraged_exposures=[
                float(v)
                for v in _grid_values(
                    grid_config,
                    "max_total_leveraged_exposure",
                    grid_config.get("max_total_leveraged_exposures", (1.0,)),
                )
            ],
            trend_filter_symbols=[
                str(v).upper()
                for v in _grid_values(
                    grid_config,
                    "trend_filter_symbol",
                    grid_config.get("trend_filter_symbols", ("QQQ",)),
                )
            ],
            trend_filter_windows=[
                int(v)
                for v in _grid_values(
                    grid_config,
                    "trend_filter_window",
                    grid_config.get("trend_filter_windows", (200,)),
                )
            ],
            defensive_assets=[
                str(v).upper()
                for v in _grid_values(
                    grid_config,
                    "defensive_asset",
                    grid_config.get("defensive_assets", ("CASH",)),
                )
            ],
            require_positive_momentum_options=[
                bool(v)
                for v in _grid_values(
                    grid_config,
                    "require_positive_momentum",
                    grid_config.get("require_positive_momentum_options", (True,)),
                )
            ],
            asset_config=asset_config,
        )

    params = _strategy_params(config)
    risk_assets = _asset_sets_from_config(params.get("risk_assets"), DEFAULT_RISK_ASSETS)[0]
    return [
        RotationParams(
            risk_assets=tuple(str(symbol).upper() for symbol in risk_assets),
            rebalance_frequency=str(params.get("rebalance_frequency", "monthly")).lower(),
            top_n=int(params.get("top_n", 1)),
            max_asset_weight=float(params.get("max_asset_weight", 1.0)),
            max_total_leveraged_exposure=float(params.get("max_total_leveraged_exposure", 1.0)),
            trend_filter_symbol=str(params.get("trend_filter_symbol", "QQQ")).upper(),
            trend_filter_window=int(params.get("trend_filter_window", 200)),
            defensive_asset=str(params.get("defensive_asset", "CASH")).upper(),
            require_positive_momentum=bool(params.get("require_positive_momentum", True)),
            asset_config=asset_config,
        )
    ]


def _run_walk_forward(config: Dict[str, Any], data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    data = data.copy()
    requested_execution_model = normalize_execution_model(
        config.get("execution_model", CLOSE_TO_CLOSE_SHIFTED)
    )
    data.attrs["requested_execution_model"] = requested_execution_model
    data.attrs["execution_model"] = requested_execution_model
    strategy_name = str(config["strategy_name"]).lower()
    objective = str(config["objective"])
    transaction_cost_bps = float(config["transaction_cost_bps"])
    train_years = int(config["train_years"])
    test_years = int(config["test_years"])
    n_jobs = int(config.get("n_jobs", 1))
    window_mode = str(config.get("walk_forward_mode", "rolling"))

    if strategy_name in {"buy_and_hold", "buy-hold", "buyhold"}:
        return _run_buy_and_hold_walk_forward(config=config, data=data)

    if strategy_name in {"ma", "moving_average", "moving-average"}:
        grid = _ma_grid(config)
        if bool(config.get("use_ensemble_wf", False)):
            wf_table, stitched = walk_forward_search_ensemble(
                data=data,
                grid=grid,
                train_objective=objective,
                transaction_cost_bps=transaction_cost_bps,
                train_years=train_years,
                test_years=test_years,
                n_jobs=n_jobs,
                top_k=int(config.get("ensemble_top_k", 5)),
                weight_power=float(config.get("ensemble_weight_power", 2.0)),
                window_mode=window_mode,
            )
        else:
            wf_table, stitched = walk_forward_search(
                data=data,
                grid=grid,
                train_objective=objective,
                transaction_cost_bps=transaction_cost_bps,
                train_years=train_years,
                test_years=test_years,
                n_jobs=n_jobs,
                window_mode=window_mode,
            )
        return wf_table, stitched, len(grid)

    if strategy_name == "regime":
        grid = _regime_grid(config)
        wf_table, stitched = walk_forward_regime_search(
            data=data,
            grid=grid,
            train_objective=objective,
            transaction_cost_bps=transaction_cost_bps,
            train_years=train_years,
            test_years=test_years,
            n_jobs=n_jobs,
            recent_days=int(config.get("recent_days", 252)),
            window_mode=window_mode,
        )
        return wf_table, stitched, len(grid)

    if strategy_name in {"core_overlay", "core-overlay", "core_overlay_tqqq", "coreoverlaytqqqstrategy"}:
        grid = _core_overlay_grid(config)
        wf_table, stitched = walk_forward_core_overlay_search(
            data=data,
            grid=grid,
            train_objective=objective,
            transaction_cost_bps=transaction_cost_bps,
            train_years=train_years,
            test_years=test_years,
            n_jobs=n_jobs,
            window_mode=window_mode,
        )
        return wf_table, stitched, len(grid)

    if strategy_name in {"vol_target", "vol-target", "voltargettqqqstrategy", "vol_target_tqqq"}:
        grid = _vol_target_grid(config)
        wf_table, stitched = walk_forward_vol_target_search(
            data=data,
            grid=grid,
            train_objective=objective,
            transaction_cost_bps=transaction_cost_bps,
            train_years=train_years,
            test_years=test_years,
            n_jobs=n_jobs,
            window_mode=window_mode,
        )
        return wf_table, stitched, len(grid)

    if _is_rotation_strategy_name(strategy_name):
        grid = _rotation_grid(config)
        wf_table, stitched = walk_forward_rotation_search(
            data=data,
            grid=grid,
            train_objective=objective,
            transaction_cost_bps=transaction_cost_bps,
            train_years=train_years,
            test_years=test_years,
            n_jobs=n_jobs,
            window_mode=window_mode,
        )
        return wf_table, stitched, len(grid)

    raise ValueError(f"Unsupported strategy_name: {config['strategy_name']}")


def _run_buy_and_hold_walk_forward(
    config: Dict[str, Any],
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    data = data.copy()
    requested_execution_model = normalize_execution_model(
        config.get("execution_model", CLOSE_TO_CLOSE_SHIFTED)
    )
    data.attrs["requested_execution_model"] = requested_execution_model
    data.attrs["execution_model"] = requested_execution_model
    trade_symbol = str(config.get("trade_symbol", config["benchmark_symbol"])).upper()
    benchmark_symbol = str(config["benchmark_symbol"]).upper()
    if trade_symbol not in data.columns:
        raise ValueError(f"Buy-and-hold trade symbol is missing from data: {trade_symbol}")
    if benchmark_symbol not in data.columns:
        raise ValueError(f"Benchmark symbol is missing from data: {benchmark_symbol}")

    train_years = int(config["train_years"])
    test_years = int(config["test_years"])
    transaction_cost_bps = float(config["transaction_cost_bps"])
    windows = walk_forward_windows(
        data.index,
        train_years=train_years,
        test_years=test_years,
        mode=str(config.get("walk_forward_mode", "rolling")),
    )
    effective_execution_model, execution_warning = resolve_execution_model(
        data,
        requested_execution_model,
        [trade_symbol],
    )
    full_trade_ret = asset_returns(data, trade_symbol, effective_execution_model)
    full_benchmark_ret = data[benchmark_symbol].pct_change().fillna(0.0)

    wf_rows: List[Dict[str, Any]] = []
    stitched_parts: List[pd.DataFrame] = []
    prev_position = 0.0

    for train_start, train_end, test_start, test_end in windows:
        test_ret = full_trade_ret.loc[test_start:test_end]
        benchmark_ret = full_benchmark_ret.loc[test_start:test_end]
        if len(test_ret) < 1:
            continue

        position = pd.Series(1.0, index=test_ret.index, name="position")
        turnover = position.diff().abs().fillna(0.0)
        turnover.iloc[0] = abs(float(position.iloc[0]) - prev_position)
        cost = turnover * (transaction_cost_bps / 10000.0)
        ret = test_ret - cost
        piece = pd.DataFrame(
            {
                "daily_ret": test_ret,
                "ret": ret,
                "position": position,
                "turnover": turnover,
                "requested_execution_model": requested_execution_model,
                "execution_model": effective_execution_model,
                "execution_model_warning": execution_warning,
            },
            index=test_ret.index,
        )

        metrics = summarize_performance(piece, benchmark_ret)
        wf_rows.append(
            {
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "strategy_name": "buy_and_hold",
                "trade_symbol": trade_symbol,
                "benchmark_symbol": benchmark_symbol,
                "test_cagr": float(metrics["cagr"]),
                "test_sharpe": float(metrics["sharpe"]),
                "test_max_dd": float(metrics["max_dd"]),
                "test_calmar": float(metrics["calmar"]),
                "test_final_equity": float(metrics["final_equity"]),
                "test_final_equity_ratio": float(metrics["final_equity_ratio"]),
            }
        )
        stitched_parts.append(piece)
        prev_position = float(position.iloc[-1])

    wf_table = pd.DataFrame(wf_rows)
    if not stitched_parts:
        return wf_table, pd.DataFrame(), 1

    stitched = pd.concat(stitched_parts).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")].copy()
    stitched["equity"] = (1.0 + stitched["ret"]).cumprod()
    return wf_table, stitched, 1


def _disable_rebound_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    disabled = copy.deepcopy(config)
    params = (
        disabled.get("strategy_params")
        or disabled.get("strategy_parameters")
        or disabled.get("parameters")
    )
    if isinstance(params, dict):
        params["use_rebound_module"] = False

    grid = disabled.get("parameter_grid")
    if isinstance(grid, dict):
        grid["use_rebound_module"] = [False]
        grid.pop("use_rebound_options", None)

    return disabled


def _enable_rebound_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    enabled = copy.deepcopy(config)
    params = (
        enabled.get("strategy_params")
        or enabled.get("strategy_parameters")
        or enabled.get("parameters")
    )
    if isinstance(params, dict):
        params["use_rebound_module"] = True

    grid = enabled.get("parameter_grid")
    if isinstance(grid, dict):
        grid["use_rebound_module"] = [True]
        grid.pop("use_rebound_options", None)

    return enabled


def _disable_governor_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    disabled = copy.deepcopy(config)
    params = (
        disabled.get("strategy_params")
        or disabled.get("strategy_parameters")
        or disabled.get("parameters")
    )
    if isinstance(params, dict):
        params["use_drawdown_governor"] = False

    grid = disabled.get("parameter_grid")
    if isinstance(grid, dict):
        grid["use_drawdown_governor"] = [False]
        grid.pop("use_drawdown_governor_options", None)

    return disabled


def _year_strategy_return(yearly_returns: pd.DataFrame, year: int) -> float:
    if yearly_returns.empty or "year" not in yearly_returns.columns:
        return float("nan")
    match = yearly_returns.loc[yearly_returns["year"] == year, "strategy_return"]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def _worst_year_strategy_return(yearly_returns: pd.DataFrame) -> float:
    if yearly_returns.empty or "strategy_return" not in yearly_returns.columns:
        return float("nan")
    return float(yearly_returns["strategy_return"].min())


def _selected_risk_off_value(wf_table: pd.DataFrame, column: str, default: Any) -> Any:
    if column not in wf_table.columns or wf_table.empty:
        return default
    values = wf_table[column].dropna().astype(str).unique().tolist()
    if not values:
        return default
    return values[0] if len(values) == 1 else ";".join(values)


def _market_internals_run_status(config: Dict[str, Any], data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if not _market_internals_enabled_from_config(config):
        return None
    return market_internals_availability(
        data,
        max_required_window=_market_internals_max_required_window(config),
        requested_symbols=_market_internals_symbols_from_config(config),
    )


def _save_market_internals_daily(output_dir: Path, stitched: pd.DataFrame) -> None:
    market_cols = [
        column
        for column in stitched.columns
        if column.startswith("market_internals_") or column.startswith("mi_")
    ]
    if not market_cols:
        return
    stitched[market_cols].to_csv(output_dir / "market_internals_daily.csv")


def _save_rebound_ablation(
    output_dir: Path,
    without_summary: pd.DataFrame,
    without_yearly: pd.DataFrame,
    with_summary: pd.DataFrame,
    with_yearly: pd.DataFrame,
) -> None:
    without_row = without_summary.iloc[0]
    with_row = with_summary.iloc[0]

    def build_row(label: str, summary_row: pd.Series, yearly: pd.DataFrame) -> Dict[str, Any]:
        return {
            "variant": label,
            "final_equity_ratio": float(summary_row["final_equity_ratio"]),
            "excess_cagr": float(summary_row["excess_cagr"]),
            "return_2019": _year_strategy_return(yearly, 2019),
            "return_2020": _year_strategy_return(yearly, 2020),
            "return_2023": _year_strategy_return(yearly, 2023),
        }

    without_values = build_row("without_rebound", without_row, without_yearly)
    with_values = build_row("with_rebound", with_row, with_yearly)
    diff_values = {
        "variant": "difference",
        "final_equity_ratio": with_values["final_equity_ratio"] - without_values["final_equity_ratio"],
        "excess_cagr": with_values["excess_cagr"] - without_values["excess_cagr"],
        "return_2019": with_values["return_2019"] - without_values["return_2019"],
        "return_2020": with_values["return_2020"] - without_values["return_2020"],
        "return_2023": with_values["return_2023"] - without_values["return_2023"],
    }

    pd.DataFrame([without_values, with_values, diff_values]).to_csv(
        output_dir / "rebound_ablation_summary.csv",
        index=False,
    )


def _maybe_save_rebound_ablation(
    config: Dict[str, Any],
    data: pd.DataFrame,
    output_dir: Path,
    wf_table: pd.DataFrame,
    with_summary: pd.DataFrame,
    with_yearly: pd.DataFrame,
) -> None:
    strategy_name = str(config["strategy_name"]).lower()
    if strategy_name not in {
        "regime",
        "core_overlay",
        "core-overlay",
        "core_overlay_tqqq",
        "coreoverlaytqqqstrategy",
        "vol_target",
        "vol-target",
        "voltargettqqqstrategy",
        "vol_target_tqqq",
    }:
        return
    if "use_rebound_module" not in wf_table.columns:
        return
    if int(wf_table["use_rebound_module"].fillna(0).astype(int).max()) <= 0:
        return

    disabled_config = _disable_rebound_in_config(config)
    _, without_stitched, _ = _run_walk_forward(disabled_config, data)
    if without_stitched.empty:
        return

    benchmark_symbol = str(config["benchmark_symbol"]).upper()
    without_summary, without_yearly = compare_to_benchmark(
        without_stitched,
        data,
        benchmark_symbol=benchmark_symbol,
    )
    _save_rebound_ablation(
        output_dir=output_dir,
        without_summary=without_summary,
        without_yearly=without_yearly,
        with_summary=with_summary,
        with_yearly=with_yearly,
    )


def _ablation_row(
    variant: str,
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    stitched: pd.DataFrame,
) -> Dict[str, Any]:
    row = summary.iloc[0]
    return {
        "variant": variant,
        "raw_outperformance_pass": bool(row["raw_outperformance_pass"]),
        "final_equity_ratio": float(row["final_equity_ratio"]),
        "excess_cagr": float(row["excess_cagr"]),
        "strategy_final_equity": float(row["strategy_final_equity"]),
        "benchmark_final_equity": float(row["benchmark_final_equity"]),
        "strategy_cagr": float(row["strategy_cagr"]),
        "benchmark_cagr": float(row["benchmark_cagr"]),
        "strategy_max_dd": float(row["strategy_max_dd"]),
        "benchmark_max_dd": float(row["benchmark_max_dd"]),
        "strategy_calmar": float(row["strategy_calmar"]),
        "benchmark_calmar": float(row["benchmark_calmar"]),
        "avg_exposure": float(stitched["position"].mean()) if "position" in stitched.columns else float("nan"),
        "trades": int((stitched["turnover"] > 0).sum()) if "turnover" in stitched.columns else 0,
        "return_2019": _year_strategy_return(yearly, 2019),
        "return_2020": _year_strategy_return(yearly, 2020),
        "return_2023": _year_strategy_return(yearly, 2023),
    }


def _save_drawdown_governor_ablation(
    output_dir: Path,
    results: List[Dict[str, Any]],
    yearly_parts: List[pd.DataFrame],
) -> None:
    pd.DataFrame(results).to_csv(output_dir / "drawdown_governor_ablation_summary.csv", index=False)
    if yearly_parts:
        pd.concat(yearly_parts, ignore_index=True).to_csv(
            output_dir / "drawdown_governor_ablation_yearly_returns.csv",
            index=False,
        )


def _maybe_save_drawdown_governor_ablation(
    config: Dict[str, Any],
    data: pd.DataFrame,
    output_dir: Path,
    wf_table: pd.DataFrame,
) -> None:
    strategy_name = str(config["strategy_name"]).lower()
    if strategy_name not in {
        "regime",
        "core_overlay",
        "core-overlay",
        "core_overlay_tqqq",
        "coreoverlaytqqqstrategy",
        "vol_target",
        "vol-target",
        "voltargettqqqstrategy",
        "vol_target_tqqq",
    }:
        return
    if "use_drawdown_governor" not in wf_table.columns:
        return
    if int(wf_table["use_drawdown_governor"].fillna(0).astype(int).max()) <= 0:
        return

    benchmark_symbol = str(config["benchmark_symbol"]).upper()
    variants = [
        ("base_strategy", _disable_governor_in_config(_disable_rebound_in_config(config))),
        ("base_governor", _disable_rebound_in_config(config)),
        ("base_governor_rebound", _enable_rebound_in_config(config)),
    ]

    results: List[Dict[str, Any]] = []
    yearly_parts: List[pd.DataFrame] = []

    for variant, variant_config in variants:
        _, stitched, _ = _run_walk_forward(variant_config, data)
        if stitched.empty:
            continue
        summary, yearly = compare_to_benchmark(
            stitched,
            data,
            benchmark_symbol=benchmark_symbol,
        )
        results.append(_ablation_row(variant, summary, yearly, stitched))
        yearly_variant = yearly.copy()
        yearly_variant.insert(0, "variant", variant)
        yearly_parts.append(yearly_variant)

    if results:
        _save_drawdown_governor_ablation(output_dir, results, yearly_parts)


def _is_vol_target_strategy(config: Dict[str, Any]) -> bool:
    strategy_name = str(config["strategy_name"]).lower()
    return strategy_name in {"vol_target", "vol-target", "voltargettqqqstrategy", "vol_target_tqqq"}


def _is_rotation_strategy(config: Dict[str, Any]) -> bool:
    return _is_rotation_strategy_name(config["strategy_name"])


def _save_rotation_outputs(output_dir: Path, stitched: pd.DataFrame) -> None:
    selected_cols = [
        col
        for col in [
            "selected_assets",
            "selected_score",
            "leveraged_exposure",
            "defensive_weight",
            "position",
            "rebalance_signal",
            "trend_filter_allows_risk",
            "available_risk_assets",
        ]
        if col in stitched.columns
    ]
    if selected_cols:
        stitched[selected_cols].to_csv(output_dir / "selected_assets.csv")

    weight_cols = [column for column in stitched.columns if column.startswith("weight_")]
    if weight_cols:
        stitched[weight_cols].to_csv(output_dir / "weights_by_date.csv")

    turnover_cols = [column for column in stitched.columns if column.startswith("turnover_")]
    summary = pd.DataFrame(
        [
            {
                "total_turnover": float(stitched["turnover"].sum()) if "turnover" in stitched.columns else float("nan"),
                "average_daily_turnover": float(stitched["turnover"].mean()) if "turnover" in stitched.columns else float("nan"),
                "trade_days": int((stitched["turnover"] > 0.0).sum()) if "turnover" in stitched.columns else 0,
                "rebalance_signals": int(stitched["rebalance_signal"].sum()) if "rebalance_signal" in stitched.columns else 0,
                "total_transaction_cost": float(stitched["cost"].sum()) if "cost" in stitched.columns else float("nan"),
                "avg_leveraged_exposure": float(stitched["leveraged_exposure"].mean()) if "leveraged_exposure" in stitched.columns else float("nan"),
                "avg_defensive_weight": float(stitched["defensive_weight"].mean()) if "defensive_weight" in stitched.columns else float("nan"),
                "max_gross_position": float(stitched["position"].max()) if "position" in stitched.columns else float("nan"),
            }
        ]
    )
    if turnover_cols:
        for column in turnover_cols:
            summary[column.replace("turnover_", "total_turnover_")] = float(stitched[column].sum())
    summary.to_csv(output_dir / "turnover_summary.csv", index=False)


def _soxl_benchmark_symbols(config: Dict[str, Any], data: pd.DataFrame) -> List[str]:
    asset_config = _asset_config(config)
    is_soxl = bool(config.get("soxl_experiment", False)) or asset_config.trade_asset == "SOXL"
    if not is_soxl:
        return []
    requested = ["SOXL", "TQQQ"]
    requested.extend(str(symbol).upper() for symbol in _as_list(config.get("benchmark_comparison_symbols", ["SOXX", "SMH"])))
    return list(dict.fromkeys(symbol for symbol in requested if symbol in data.columns))


def _save_soxl_outputs(
    output_dir: Path,
    config: Dict[str, Any],
    stitched: pd.DataFrame,
    data: pd.DataFrame,
    wf_table: pd.DataFrame,
) -> None:
    symbols = _soxl_benchmark_symbols(config, data)
    if not symbols:
        return

    yearly_parts: List[pd.DataFrame] = []
    for symbol in symbols:
        summary, yearly = compare_to_benchmark(stitched, data, benchmark_symbol=symbol)
        lower = symbol.replace("^", "").lower()
        summary.to_csv(output_dir / f"soxl_vs_{lower}_summary.csv", index=False)
        yearly_with_symbol = yearly.copy()
        yearly_with_symbol.insert(0, "benchmark_symbol", symbol)
        yearly_with_symbol.to_csv(output_dir / f"soxl_vs_{lower}_yearly_returns.csv", index=False)
        yearly_parts.append(yearly_with_symbol)

    if yearly_parts:
        pd.concat(yearly_parts, ignore_index=True).to_csv(
            output_dir / "soxl_yearly_returns.csv",
            index=False,
        )
    wf_table.to_csv(output_dir / "soxl_selected_params.csv", index=False)


def _save_drawdown_chart_data(
    output_dir: Path,
    stitched: pd.DataFrame,
    data: pd.DataFrame,
    benchmark_symbol: str,
) -> None:
    benchmark_prices = data.loc[stitched.index, benchmark_symbol].astype(float)
    benchmark_ret = benchmark_prices.pct_change().fillna(0.0)
    benchmark_equity = (1.0 + benchmark_ret).cumprod()
    strategy_equity = stitched["equity"].astype(float)
    chart = pd.DataFrame(
        {
            "strategy_equity": strategy_equity,
            "strategy_drawdown": strategy_equity / strategy_equity.cummax() - 1.0,
            "benchmark_equity": benchmark_equity,
            "benchmark_drawdown": benchmark_equity / benchmark_equity.cummax() - 1.0,
        },
        index=stitched.index,
    )
    chart.to_csv(output_dir / "drawdown_chart_data.csv")


def _save_vol_target_acceptance_summary(
    output_dir: Path,
    benchmark_summary: pd.DataFrame,
    stitched: pd.DataFrame,
    wf_table: pd.DataFrame,
) -> None:
    summary_row = benchmark_summary.iloc[0]
    raw_pass = bool(summary_row["raw_outperformance_pass"])
    total_cost = float(stitched["cost"].sum()) if "cost" in stitched.columns else float("nan")
    avg_target_exposure = (
        float(stitched["target_exposure"].mean())
        if "target_exposure" in stitched.columns
        else float("nan")
    )
    avg_realized_ann_vol = (
        float(stitched["realized_ann_vol"].mean())
        if "realized_ann_vol" in stitched.columns
        else float("nan")
    )
    trades = int((stitched["turnover"] > 0).sum()) if "turnover" in stitched.columns else 0
    selected_windows = int(len(wf_table))

    diagnostics = pd.DataFrame(
        [
            {
                "raw_outperformance_pass": raw_pass,
                "final_equity_ratio": float(summary_row["final_equity_ratio"]),
                "excess_cagr": float(summary_row["excess_cagr"]),
                "strategy_final_equity": float(summary_row["strategy_final_equity"]),
                "benchmark_final_equity": float(summary_row["benchmark_final_equity"]),
                "strategy_max_dd": float(summary_row["strategy_max_dd"]),
                "benchmark_max_dd": float(summary_row["benchmark_max_dd"]),
                "avg_exposure": float(stitched["position"].mean()),
                "avg_target_exposure": avg_target_exposure,
                "avg_realized_ann_vol": avg_realized_ann_vol,
                "trades": trades,
                "total_transaction_cost": total_cost,
                "walk_forward_windows": selected_windows,
                "diagnosis": (
                    "pass: strategy final equity exceeded same-period TQQQ"
                    if raw_pass
                    else "fail: strategy final equity did not exceed same-period TQQQ; inspect excess_cagr, exposure, drawdown, turnover, and total_transaction_cost"
                ),
            }
        ]
    )
    diagnostics.to_csv(output_dir / "vol_target_acceptance_summary.csv", index=False)

    print("\n=== VolTarget acceptance ===")
    if raw_pass:
        print("PASS: strategy final equity exceeded same-period TQQQ.")
    else:
        print("FAIL: strategy final equity did not exceed same-period TQQQ.")
        print(f"EXCESS CAGR: {float(summary_row['excess_cagr']):.6f}")
        print(f"AVG EXPOSURE: {float(stitched['position'].mean()):.6f}")
        print(f"AVG REALIZED ANN VOL: {avg_realized_ann_vol:.6f}")
        print(f"TOTAL TRANSACTION COST: {total_cost:.6f}")
        print(
            "MAX DD STRATEGY / BENCHMARK: "
            f"{float(summary_row['strategy_max_dd']):.6f} / {float(summary_row['benchmark_max_dd']):.6f}"
        )


def run_experiment_config(config_path: Path, objective_override: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml_file(config_path)
    if objective_override:
        config["objective"] = str(objective_override)
    return run_experiment(config, config_path=config_path)


def _experiment_dry_run_row(
    config: Dict[str, Any],
    *,
    config_path: Optional[Path] = None,
    include_full_experiment_overhead: bool = True,
) -> Dict[str, Any]:
    estimate = estimate_experiment_workload(
        config,
        include_full_experiment_overhead=include_full_experiment_overhead,
    )
    symbols = [str(symbol).upper() for symbol in _as_list(config.get("symbols"))]
    asset_config = None
    try:
        asset_config = _asset_config(config)
    except Exception:
        asset_config = None
    row = {
        "status": "dry_run",
        "experiment_name": str(config.get("experiment_name", "")),
        "strategy_name": str(config.get("strategy_name", "")),
        "config_path": str(config_path) if config_path is not None else "",
        "symbols": ";".join(symbols),
        "benchmark_symbol": str(config.get("benchmark_symbol", "")).upper(),
        "output_dir": str(config.get("output_dir", "")),
        "train_years": config.get("train_years", ""),
        "test_years": config.get("test_years", ""),
        "objective": str(config.get("objective", "")),
        "transaction_cost_bps": config.get("transaction_cost_bps", ""),
        "trade_asset": asset_config.trade_asset if asset_config is not None else "",
        "primary_signal_asset": asset_config.primary_signal_asset if asset_config is not None else "",
        "secondary_filter_asset": asset_config.secondary_filter_asset if asset_config is not None else "",
        "actual_runtime_seconds": 0.0,
    }
    row.update(estimate)
    return row


def dry_run_experiment_config(config_path: Path, objective_override: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml_file(config_path)
    if objective_override:
        config["objective"] = str(objective_override)
    row = _experiment_dry_run_row(config, config_path=config_path)
    print(pd.DataFrame([row]).to_string(index=False))
    return row


def run_experiment(config: Dict[str, Any], config_path: Optional[Path] = None) -> Dict[str, Any]:
    _require_experiment_fields(config)
    estimate = estimate_experiment_workload(config, include_full_experiment_overhead=True)
    started = time.perf_counter()

    data = _load_price_data(config)
    market_internals_status = _market_internals_run_status(config, data)
    rotation_asset_status = _rotation_asset_status(config, data)
    wf_table, stitched, grid_size = _run_walk_forward(config, data)
    if stitched.empty:
        raise ValueError(f"Experiment produced no stitched equity: {config['experiment_name']}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_symbol = str(config["benchmark_symbol"]).upper()
    benchmark_summary, yearly_returns = compare_to_benchmark(
        stitched,
        data,
        benchmark_symbol=benchmark_symbol,
    )
    synthetic_tracking_summary = None
    synthetic_tracking_daily = None
    if bool(config.get("use_synthetic_leverage", False)):
        synthetic_target_symbol = str(config.get("synthetic_target_symbol", "TQQQ")).upper()
        real_symbol = str(config.get("real_leveraged_symbol", synthetic_target_symbol)).upper()
        real_column = f"REAL_{real_symbol}"
        if real_column in data.columns and synthetic_target_symbol in data.columns:
            real_overlap = data[real_column].dropna()
            synthetic_overlap = data[synthetic_target_symbol].reindex(real_overlap.index).dropna()
            if not real_overlap.empty and not synthetic_overlap.empty:
                synthetic_tracking_summary, synthetic_tracking_daily = compare_real_to_synthetic(
                    real_prices=data[real_column],
                    synthetic_prices=data[synthetic_target_symbol],
                )

    stitched.to_csv(output_dir / "stitched_equity.csv")
    _save_market_internals_daily(output_dir, stitched)
    wf_table.to_csv(output_dir / "walk_forward_windows.csv", index=False)
    wf_table.to_csv(output_dir / "parameter_selection_by_window.csv", index=False)
    benchmark_summary.to_csv(output_dir / "same_period_benchmark_summary.csv", index=False)
    yearly_returns.to_csv(output_dir / "yearly_returns.csv", index=False)
    if _is_rotation_strategy(config):
        _save_rotation_outputs(output_dir, stitched)
    _save_soxl_outputs(
        output_dir=output_dir,
        config=config,
        stitched=stitched,
        data=data,
        wf_table=wf_table,
    )
    if _is_vol_target_strategy(config):
        _save_drawdown_chart_data(
            output_dir=output_dir,
            stitched=stitched,
            data=data,
            benchmark_symbol=benchmark_symbol,
        )
    _maybe_save_rebound_ablation(
        config=config,
        data=data,
        output_dir=output_dir,
        wf_table=wf_table,
        with_summary=benchmark_summary,
        with_yearly=yearly_returns,
    )
    _maybe_save_drawdown_governor_ablation(
        config=config,
        data=data,
        output_dir=output_dir,
        wf_table=wf_table,
    )
    if synthetic_tracking_summary is not None and synthetic_tracking_daily is not None:
        save_synthetic_tracking(
            out_dir=output_dir,
            summary=synthetic_tracking_summary,
            daily=synthetic_tracking_daily,
        )
    sensitivity_outputs = run_cost_and_execution_sensitivity(
        config=config,
        data=data,
        output_dir=output_dir,
        standard_wf_table=wf_table,
        standard_stitched=stitched,
        grid_size=grid_size,
        walk_forward_runner=_run_walk_forward,
    )
    run_anti_overfit_validation(
        config=config,
        data=data,
        output_dir=output_dir,
        standard_wf_table=wf_table,
        standard_stitched=stitched,
        standard_summary=benchmark_summary,
        grid_size=grid_size,
        walk_forward_runner=_run_walk_forward,
    )
    run_config_payload = {
        "command": "run-config",
        "config_path": str(config_path) if config_path is not None else None,
        "grid_size": grid_size,
        "execution_model": normalize_execution_model(config.get("execution_model", CLOSE_TO_CLOSE_SHIFTED)),
        **config,
    }
    execution_warnings = []
    if "execution_model_warning" in stitched.columns:
        execution_warnings.extend(
            sorted(
                {
                    str(value)
                    for value in stitched["execution_model_warning"].dropna().unique()
                    if str(value)
                }
            )
        )
    for sensitivity_name, sensitivity_df in sensitivity_outputs.items():
        if "execution_model_warning" in sensitivity_df.columns:
            execution_warnings.extend(
                str(value)
                for value in sensitivity_df["execution_model_warning"].dropna().unique()
                if str(value)
            )
    if execution_warnings:
        run_config_payload["execution_model_warnings"] = sorted(set(execution_warnings))
    if market_internals_status is not None:
        run_config_payload["market_internals_data_status"] = market_internals_status
    if rotation_asset_status is not None:
        run_config_payload["rotation_asset_status"] = rotation_asset_status
    runtime_profile = _runtime_profile_fields(estimate, actual_runtime_seconds=time.perf_counter() - started)
    run_config_payload["runtime_profile"] = runtime_profile
    save_run_config(output_dir, run_config_payload)

    print_benchmark_summary(benchmark_summary)
    if _is_vol_target_strategy(config):
        _save_vol_target_acceptance_summary(
            output_dir=output_dir,
            benchmark_summary=benchmark_summary,
            stitched=stitched,
            wf_table=wf_table,
        )
    generate_experiment_report(output_dir=output_dir, config=run_config_payload, price_data=data)

    row = benchmark_summary.iloc[0].to_dict()
    asset_config = _asset_config(config)
    row.update(
        {
            "experiment_name": config["experiment_name"],
            "strategy_name": config["strategy_name"],
            "objective": config["objective"],
            "output_dir": str(output_dir),
            "config_path": str(config_path) if config_path is not None else "",
            "grid_size": grid_size,
            "trade_asset": asset_config.trade_asset,
            "primary_signal_asset": asset_config.primary_signal_asset,
            "secondary_filter_asset": asset_config.secondary_filter_asset,
            "risk_off_symbol": _selected_risk_off_value(wf_table, "risk_off_symbol", ""),
            "risk_off_weight": _selected_risk_off_value(wf_table, "risk_off_weight", ""),
            "avg_market_internals_risk_score": float(stitched["market_internals_risk_score"].mean())
            if "market_internals_risk_score" in stitched.columns
            else float("nan"),
            "avg_leveraged_exposure": float(stitched["leveraged_exposure"].mean())
            if "leveraged_exposure" in stitched.columns
            else float("nan"),
            "avg_defensive_weight": float(stitched["defensive_weight"].mean())
            if "defensive_weight" in stitched.columns
            else float("nan"),
            "worst_year": _worst_year_strategy_return(yearly_returns),
            "return_2022": _year_strategy_return(yearly_returns, 2022),
            "return_2023": _year_strategy_return(yearly_returns, 2023),
            **runtime_profile,
        }
    )
    return row


def _resolve_config_path(batch_path: Path, value: Any) -> Path:
    candidate = Path(str(value))
    if candidate.exists():
        return candidate
    return batch_path.parent / candidate


def _batch_entry_config(batch_path: Path, entry: Any, objective_override: Optional[str]) -> tuple[Dict[str, Any], Optional[Path]]:
    if isinstance(entry, dict):
        config = dict(entry)
        if objective_override:
            config["objective"] = str(objective_override)
        return config, None
    config_path = _resolve_config_path(batch_path, entry)
    config = load_yaml_file(config_path)
    if objective_override:
        config["objective"] = str(objective_override)
    return config, config_path


def run_batch_config(
    batch_path: Path,
    objective_override: Optional[str] = None,
    *,
    max_configs: Optional[int] = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    batch_path = Path(batch_path)
    batch = load_yaml_file(batch_path)
    batch_name = str(batch.get("batch_name", batch_path.stem))
    entries = batch.get("configs") or batch.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Batch config must contain a non-empty configs list.")
    total_configs = len(entries)
    if max_configs is not None:
        entries = entries[: max(0, int(max_configs))]

    rows: List[Dict[str, Any]] = []
    output_dir = Path(batch.get("output_dir", Path("outputs") / batch_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_started = time.perf_counter()
    if dry_run:
        for entry in entries:
            config, config_path = _batch_entry_config(batch_path, entry, objective_override)
            rows.append(_experiment_dry_run_row(config, config_path=config_path))
        summary = pd.DataFrame(rows)
        summary["batch_total_configs"] = total_configs
        summary["batch_selected_configs"] = len(entries)
        summary["batch_dry_run"] = True
        summary.to_csv(output_dir / "batch_dry_run_summary.csv", index=False)
        print(summary.to_string(index=False))
        print(f"\nDry-run summary written to: {output_dir / 'batch_dry_run_summary.csv'}")
        return summary

    for entry in entries:
        config, config_path = _batch_entry_config(batch_path, entry, objective_override)
        rows.append(run_experiment(config, config_path=config_path))

    summary = pd.DataFrame(rows)
    summary["batch_total_configs"] = total_configs
    summary["batch_selected_configs"] = len(entries)
    summary["batch_actual_runtime_seconds"] = time.perf_counter() - batch_started
    if "estimated_runtime_seconds" in summary.columns:
        summary["batch_estimated_runtime_seconds"] = pd.to_numeric(
            summary["estimated_runtime_seconds"],
            errors="coerce",
        ).sum()

    summary.to_csv(output_dir / "batch_summary.csv", index=False)
    summary.sort_values("final_equity_ratio", ascending=False).to_csv(
        output_dir / "ranked_by_final_equity_ratio.csv",
        index=False,
    )
    summary.sort_values("excess_cagr", ascending=False).to_csv(
        output_dir / "ranked_by_excess_cagr.csv",
        index=False,
    )
    summary.sort_values("strategy_calmar", ascending=False).to_csv(
        output_dir / "ranked_by_calmar.csv",
        index=False,
    )
    if "risk_off_symbol" in summary.columns:
        risk_cols = [
            col
            for col in [
                "experiment_name",
                "strategy_name",
                "risk_off_symbol",
                "risk_off_weight",
                "start_date",
                "end_date",
                "final_equity_ratio",
                "strategy_cagr",
                "strategy_max_dd",
                "strategy_calmar",
                "worst_year",
                "return_2022",
                "return_2023",
                "raw_outperformance_pass",
                "output_dir",
            ]
            if col in summary.columns
        ]
        risk_off_comparison = summary[risk_cols].rename(
            columns={
                "strategy_cagr": "CAGR",
                "strategy_max_dd": "max_dd",
                "strategy_calmar": "Calmar",
            }
        )
        risk_off_comparison.sort_values("final_equity_ratio", ascending=False).to_csv(
            output_dir / "risk_off_comparison.csv",
            index=False,
        )
    return summary
