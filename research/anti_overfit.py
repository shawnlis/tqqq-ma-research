from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .reports import compare_to_benchmark

WalkForwardRunner = Callable[[Dict[str, Any], pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, int]]


VARIANT_COLUMNS = [
    "variant",
    "status",
    "train_years",
    "test_years",
    "walk_forward_mode",
    "transaction_cost_bps",
    "grid_size",
    "walk_forward_windows",
    "stitched_rows",
    "benchmark_symbol",
    "start_date",
    "end_date",
    "strategy_final_equity",
    "benchmark_final_equity",
    "strategy_cagr",
    "benchmark_cagr",
    "strategy_sharpe",
    "benchmark_sharpe",
    "strategy_max_dd",
    "benchmark_max_dd",
    "strategy_calmar",
    "benchmark_calmar",
    "final_equity_ratio",
    "excess_cagr",
    "raw_outperformance_pass",
    "error",
]

BOOTSTRAP_COLUMNS = [
    "source_variant",
    "block_size",
    "bootstrap_iterations",
    "probability_strategy_beats_benchmark",
    "median_final_equity_ratio",
    "p05_final_equity_ratio",
    "p95_final_equity_ratio",
    "status",
]

SUBPERIOD_COLUMNS = [
    "period",
    "status",
    "start_date",
    "end_date",
    "stitched_rows",
    "benchmark_symbol",
    "strategy_final_equity",
    "benchmark_final_equity",
    "strategy_cagr",
    "benchmark_cagr",
    "strategy_max_dd",
    "benchmark_max_dd",
    "final_equity_ratio",
    "excess_cagr",
    "raw_outperformance_pass",
    "error",
]

STABILITY_COLUMNS = [
    "window",
    "metric",
    "selected_metric",
    "neighborhood_count",
    "neighborhood_mean",
    "neighborhood_min",
    "neighborhood_max",
    "neighborhood_std",
    "isolated",
    "parameter_signature",
]

VALIDATION_COLUMNS = [
    "candidate_viable",
    "standard_final_equity_ratio",
    "standard_walk_forward_pass",
    "alternate_walk_forward_pass",
    "cost_sensitivity_pass",
    "parameter_neighborhood_not_isolated",
    "parameter_isolated_fraction",
    "parameter_min_neighborhood_count",
    "number_parameter_combinations_tested",
    "best_in_sample_result",
    "walk_forward_result",
    "degradation_from_in_sample_to_oos",
    "bootstrap_probability_20d",
    "bootstrap_probability_60d",
]


def _nan_row(columns: Iterable[str], **values: Any) -> Dict[str, Any]:
    row = {column: np.nan for column in columns}
    row.update(values)
    return row


def _variant_config(
    config: Dict[str, Any],
    *,
    train_years: int,
    test_years: int,
    walk_forward_mode: str,
    transaction_cost_bps: Optional[float] = None,
) -> Dict[str, Any]:
    variant = copy.deepcopy(config)
    variant["train_years"] = int(train_years)
    variant["test_years"] = int(test_years)
    variant["walk_forward_mode"] = str(walk_forward_mode)
    if transaction_cost_bps is not None:
        variant["transaction_cost_bps"] = float(transaction_cost_bps)
    return variant


def _same_variant(
    config: Dict[str, Any],
    *,
    train_years: int,
    test_years: int,
    walk_forward_mode: str,
    transaction_cost_bps: Optional[float],
) -> bool:
    configured_cost = float(config.get("transaction_cost_bps", 0.0))
    target_cost = configured_cost if transaction_cost_bps is None else float(transaction_cost_bps)
    return (
        int(config.get("train_years", 0)) == int(train_years)
        and int(config.get("test_years", 0)) == int(test_years)
        and str(config.get("walk_forward_mode", "rolling")).lower() == str(walk_forward_mode).lower()
        and configured_cost == target_cost
    )


def _variant_summary_row(
    *,
    variant: str,
    config: Dict[str, Any],
    stitched: pd.DataFrame,
    wf_table: pd.DataFrame,
    grid_size: int,
    price_data: pd.DataFrame,
    benchmark_symbol: str,
) -> Dict[str, Any]:
    metadata = {
        "variant": variant,
        "train_years": int(config.get("train_years", np.nan)),
        "test_years": int(config.get("test_years", np.nan)),
        "walk_forward_mode": str(config.get("walk_forward_mode", "rolling")),
        "transaction_cost_bps": float(config.get("transaction_cost_bps", np.nan)),
        "grid_size": int(grid_size),
        "walk_forward_windows": int(len(wf_table)),
        "benchmark_symbol": benchmark_symbol,
    }
    if stitched.empty:
        return _nan_row(
            VARIANT_COLUMNS,
            **metadata,
            status="no_result",
            stitched_rows=0,
            error="variant produced no stitched equity",
        )

    summary, _ = compare_to_benchmark(stitched, price_data, benchmark_symbol=benchmark_symbol)
    row = summary.iloc[0].to_dict()
    out = _nan_row(VARIANT_COLUMNS, **metadata, status="ok", error="")
    out.update({key: row.get(key, np.nan) for key in row.keys() if key in VARIANT_COLUMNS})
    out["variant"] = variant
    out["status"] = "ok"
    out["train_years"] = metadata["train_years"]
    out["test_years"] = metadata["test_years"]
    out["walk_forward_mode"] = metadata["walk_forward_mode"]
    out["transaction_cost_bps"] = metadata["transaction_cost_bps"]
    out["grid_size"] = metadata["grid_size"]
    out["walk_forward_windows"] = metadata["walk_forward_windows"]
    out["error"] = ""
    return out


def _run_variant(
    *,
    variant: str,
    config: Dict[str, Any],
    data: pd.DataFrame,
    price_data: pd.DataFrame,
    benchmark_symbol: str,
    walk_forward_runner: WalkForwardRunner,
    standard_wf_table: pd.DataFrame,
    standard_stitched: pd.DataFrame,
    standard_grid_size: int,
    train_years: int,
    test_years: int,
    walk_forward_mode: str,
    transaction_cost_bps: Optional[float] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, int]:
    variant_config = _variant_config(
        config,
        train_years=train_years,
        test_years=test_years,
        walk_forward_mode=walk_forward_mode,
        transaction_cost_bps=transaction_cost_bps,
    )
    try:
        if _same_variant(
            config,
            train_years=train_years,
            test_years=test_years,
            walk_forward_mode=walk_forward_mode,
            transaction_cost_bps=transaction_cost_bps,
        ):
            wf_table, stitched, grid_size = standard_wf_table, standard_stitched, standard_grid_size
        else:
            wf_table, stitched, grid_size = walk_forward_runner(variant_config, data)
        row = _variant_summary_row(
            variant=variant,
            config=variant_config,
            stitched=stitched,
            wf_table=wf_table,
            grid_size=grid_size,
            price_data=price_data,
            benchmark_symbol=benchmark_symbol,
        )
        return row, wf_table, stitched, grid_size
    except Exception as exc:
        row = _nan_row(
            VARIANT_COLUMNS,
            variant=variant,
            status="error",
            train_years=train_years,
            test_years=test_years,
            walk_forward_mode=walk_forward_mode,
            transaction_cost_bps=float(
                config.get("transaction_cost_bps", np.nan)
                if transaction_cost_bps is None
                else transaction_cost_bps
            ),
            grid_size=np.nan,
            walk_forward_windows=0,
            stitched_rows=0,
            benchmark_symbol=benchmark_symbol,
            error=str(exc),
        )
        return row, pd.DataFrame(), pd.DataFrame(), 0


def block_bootstrap_summary(
    stitched_df: pd.DataFrame,
    price_data: pd.DataFrame,
    benchmark_symbol: str,
    *,
    block_sizes: Iterable[int] = (20, 60),
    n_bootstrap: int = 200,
    seed: int = 42,
    source_variant: str = "standard_5y_1y",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    if stitched_df.empty or "ret" not in stitched_df.columns:
        for block_size in block_sizes:
            rows.append(
                _nan_row(
                    BOOTSTRAP_COLUMNS,
                    source_variant=source_variant,
                    block_size=int(block_size),
                    bootstrap_iterations=int(n_bootstrap),
                    status="no_data",
                )
            )
        return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)

    stitched = stitched_df.sort_index().copy()
    missing_dates = stitched.index.difference(price_data.index)
    if len(missing_dates) > 0:
        raise ValueError(f"price_data is missing stitched benchmark date: {missing_dates[0]}")

    benchmark_prices = price_data.loc[stitched.index, benchmark_symbol].astype(float)
    benchmark_ret = benchmark_prices.pct_change().fillna(0.0).to_numpy(dtype=float)
    strategy_ret = stitched["ret"].astype(float).to_numpy(dtype=float)
    n = len(strategy_ret)

    for block_size_raw in block_sizes:
        block_size = int(block_size_raw)
        if block_size <= 0 or n == 0 or n_bootstrap <= 0:
            rows.append(
                _nan_row(
                    BOOTSTRAP_COLUMNS,
                    source_variant=source_variant,
                    block_size=block_size,
                    bootstrap_iterations=int(n_bootstrap),
                    status="invalid_input",
                )
            )
            continue

        ratios = []
        strategy_beats = []
        max_start = max(n - block_size, 0)
        for _ in range(int(n_bootstrap)):
            sampled_positions: List[int] = []
            while len(sampled_positions) < n:
                start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                stop = min(start + block_size, n)
                sampled_positions.extend(range(start, stop))
            positions = np.asarray(sampled_positions[:n], dtype=int)
            strategy_final = float(np.prod(1.0 + strategy_ret[positions]))
            benchmark_final = float(np.prod(1.0 + benchmark_ret[positions]))
            ratio = strategy_final / benchmark_final if benchmark_final != 0.0 else np.nan
            ratios.append(ratio)
            strategy_beats.append(strategy_final > benchmark_final)

        ratio_series = pd.Series(ratios, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "source_variant": source_variant,
                "block_size": block_size,
                "bootstrap_iterations": int(n_bootstrap),
                "probability_strategy_beats_benchmark": float(np.mean(strategy_beats)),
                "median_final_equity_ratio": float(ratio_series.median()) if not ratio_series.empty else np.nan,
                "p05_final_equity_ratio": float(ratio_series.quantile(0.05)) if not ratio_series.empty else np.nan,
                "p95_final_equity_ratio": float(ratio_series.quantile(0.95)) if not ratio_series.empty else np.nan,
                "status": "ok",
            }
        )

    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)


def subperiod_summary(
    stitched_df: pd.DataFrame,
    price_data: pd.DataFrame,
    benchmark_symbol: str,
) -> pd.DataFrame:
    if stitched_df.empty:
        return pd.DataFrame(
            [
                _nan_row(
                    SUBPERIOD_COLUMNS,
                    period="all",
                    status="no_data",
                    benchmark_symbol=benchmark_symbol,
                    error="stitched_df is empty",
                )
            ],
            columns=SUBPERIOD_COLUMNS,
        )

    stitched = stitched_df.sort_index().copy()
    stitched_end = stitched.index.max()
    periods = [
        ("2011-2015", pd.Timestamp("2011-01-01"), pd.Timestamp("2015-12-31")),
        ("2016-2019", pd.Timestamp("2016-01-01"), pd.Timestamp("2019-12-31")),
        ("2020-2021", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
        ("2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
        ("2023-present", pd.Timestamp("2023-01-01"), stitched_end),
        ("synthetic_2000-2002", pd.Timestamp("2000-01-01"), pd.Timestamp("2002-12-31")),
        ("synthetic_2008", pd.Timestamp("2008-01-01"), pd.Timestamp("2008-12-31")),
    ]

    rows: List[Dict[str, Any]] = []
    for label, start, end in periods:
        sub = stitched.loc[start:end]
        if sub.empty:
            rows.append(
                _nan_row(
                    SUBPERIOD_COLUMNS,
                    period=label,
                    status="no_overlap",
                    start_date=start.date().isoformat(),
                    end_date=end.date().isoformat(),
                    stitched_rows=0,
                    benchmark_symbol=benchmark_symbol,
                    error="",
                )
            )
            continue

        try:
            summary, _ = compare_to_benchmark(sub, price_data, benchmark_symbol=benchmark_symbol)
            row = summary.iloc[0].to_dict()
            out = _nan_row(SUBPERIOD_COLUMNS, period=label, status="ok", error="")
            out.update({key: row.get(key, np.nan) for key in row.keys() if key in SUBPERIOD_COLUMNS})
            out["period"] = label
            out["status"] = "ok"
            out["error"] = ""
            rows.append(out)
        except Exception as exc:
            rows.append(
                _nan_row(
                    SUBPERIOD_COLUMNS,
                    period=label,
                    status="error",
                    start_date=sub.index[0].date().isoformat(),
                    end_date=sub.index[-1].date().isoformat(),
                    stitched_rows=int(len(sub)),
                    benchmark_symbol=benchmark_symbol,
                    error=str(exc),
                )
            )

    return pd.DataFrame(rows, columns=SUBPERIOD_COLUMNS)


def _is_metric_column(column: str) -> bool:
    exact = {
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "start_date",
        "end_date",
        "train_objective",
        "objective",
        "strategy_name",
        "benchmark_symbol",
        "output_dir",
        "config_path",
        "grid_size",
        "walk_forward_mode",
        "walk_forward_windows",
    }
    prefixes = (
        "train_best_",
        "train_topk_",
        "test_",
        "recent_",
        "strategy_",
        "benchmark_",
        "avg_",
        "return_",
        "wf_",
    )
    suffixes = ("_date",)
    return column in exact or column.startswith(prefixes) or column.endswith(suffixes)


def _parameter_columns(wf_table: pd.DataFrame) -> List[str]:
    columns = []
    for column in wf_table.columns:
        if _is_metric_column(str(column)):
            continue
        if pd.api.types.is_numeric_dtype(wf_table[column]) or pd.api.types.is_bool_dtype(wf_table[column]):
            columns.append(str(column))
        elif pd.api.types.is_object_dtype(wf_table[column]) or pd.api.types.is_string_dtype(wf_table[column]):
            columns.append(str(column))
    return columns


def _numeric_neighbor_mask(values: pd.Series, selected: float) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    if np.isnan(selected):
        return numeric_values.isna()
    if float(selected).is_integer():
        radius = max(1.0, abs(float(selected)) * 0.25)
    else:
        radius = max(0.05, abs(float(selected)) * 0.25)
    return (numeric_values - selected).abs() <= radius


def _parameter_signature(row: pd.Series, columns: Iterable[str]) -> str:
    return "|".join(f"{column}={row.get(column)}" for column in columns)


def parameter_stability_summary(
    wf_table: pd.DataFrame,
    *,
    preferred_metric: str = "test_final_equity_ratio",
) -> pd.DataFrame:
    if wf_table.empty:
        return pd.DataFrame(columns=STABILITY_COLUMNS)

    metric = preferred_metric
    if metric not in wf_table.columns:
        if "test_final_equity" in wf_table.columns:
            metric = "test_final_equity"
        elif "train_best_objective" in wf_table.columns:
            metric = "train_best_objective"
        else:
            return pd.DataFrame(columns=STABILITY_COLUMNS)

    table = wf_table.copy()
    table[metric] = pd.to_numeric(table[metric], errors="coerce")
    param_cols = _parameter_columns(table)
    if not param_cols:
        return pd.DataFrame(columns=STABILITY_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for idx, selected in table.iterrows():
        mask = pd.Series(True, index=table.index)
        for column in param_cols:
            selected_value = selected[column]
            if pd.api.types.is_numeric_dtype(table[column]) or pd.api.types.is_bool_dtype(table[column]):
                mask &= _numeric_neighbor_mask(table[column], float(selected_value))
            else:
                mask &= table[column].astype(str).fillna("") == str(selected_value)

        neighbors = table.loc[mask, metric].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "window": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "metric": metric,
                "selected_metric": float(selected[metric]) if pd.notna(selected[metric]) else np.nan,
                "neighborhood_count": int(len(neighbors)),
                "neighborhood_mean": float(neighbors.mean()) if not neighbors.empty else np.nan,
                "neighborhood_min": float(neighbors.min()) if not neighbors.empty else np.nan,
                "neighborhood_max": float(neighbors.max()) if not neighbors.empty else np.nan,
                "neighborhood_std": float(neighbors.std(ddof=0)) if len(neighbors) > 1 else 0.0,
                "isolated": bool(len(neighbors) <= 1),
                "parameter_signature": _parameter_signature(selected, param_cols),
            }
        )

    return pd.DataFrame(rows, columns=STABILITY_COLUMNS)


def _ratio_for_variant(variant_summary: pd.DataFrame, variant: str) -> float:
    if variant_summary.empty:
        return np.nan
    rows = variant_summary.loc[variant_summary["variant"] == variant]
    if rows.empty:
        return np.nan
    return float(pd.to_numeric(rows["final_equity_ratio"], errors="coerce").iloc[0])


def _passes_ratio(value: float) -> bool:
    return bool(pd.notna(value) and float(value) > 1.0)


def _best_in_sample_result(wf_table: pd.DataFrame) -> float:
    for column in ("train_best_final_equity_ratio", "train_best_objective"):
        if column in wf_table.columns:
            values = pd.to_numeric(wf_table[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not values.empty:
                return float(values.max())
    return np.nan


def validation_summary(
    *,
    variant_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    stability: pd.DataFrame,
    standard_wf_table: pd.DataFrame,
    number_parameter_combinations_tested: int,
) -> pd.DataFrame:
    standard_ratio = _ratio_for_variant(variant_summary, "standard_5y_1y")
    standard_pass = _passes_ratio(standard_ratio)
    alternate_pass = any(
        _passes_ratio(_ratio_for_variant(variant_summary, variant))
        for variant in ("alt_3y_1y", "alt_7y_1y", "anchored_expanding_1y")
    )
    cost_pass = all(
        _passes_ratio(_ratio_for_variant(variant_summary, variant))
        for variant in ("cost_25bps", "cost_50bps")
    )

    if stability.empty or "isolated" not in stability.columns:
        isolated_fraction = np.nan
        min_neighborhood_count = np.nan
        neighborhood_pass = False
    else:
        isolated = stability["isolated"].astype(bool)
        counts = pd.to_numeric(stability["neighborhood_count"], errors="coerce")
        isolated_fraction = float(isolated.mean()) if len(isolated) else np.nan
        min_neighborhood_count = float(counts.min()) if not counts.dropna().empty else np.nan
        neighborhood_pass = bool(len(isolated) > 0 and isolated_fraction < 0.5 and counts.max() > 1)

    best_in_sample = _best_in_sample_result(standard_wf_table)
    degradation = (
        best_in_sample - standard_ratio
        if pd.notna(best_in_sample) and pd.notna(standard_ratio)
        else np.nan
    )

    bootstrap_probabilities = {}
    if not bootstrap.empty:
        for block_size in (20, 60):
            rows = bootstrap.loc[bootstrap["block_size"] == block_size]
            bootstrap_probabilities[block_size] = (
                float(pd.to_numeric(rows["probability_strategy_beats_benchmark"], errors="coerce").iloc[0])
                if not rows.empty
                else np.nan
            )

    row = {
        "candidate_viable": bool(standard_pass and alternate_pass and cost_pass and neighborhood_pass),
        "standard_final_equity_ratio": standard_ratio,
        "standard_walk_forward_pass": bool(standard_pass),
        "alternate_walk_forward_pass": bool(alternate_pass),
        "cost_sensitivity_pass": bool(cost_pass),
        "parameter_neighborhood_not_isolated": bool(neighborhood_pass),
        "parameter_isolated_fraction": isolated_fraction,
        "parameter_min_neighborhood_count": min_neighborhood_count,
        "number_parameter_combinations_tested": int(number_parameter_combinations_tested),
        "best_in_sample_result": best_in_sample,
        "walk_forward_result": standard_ratio,
        "degradation_from_in_sample_to_oos": degradation,
        "bootstrap_probability_20d": bootstrap_probabilities.get(20, np.nan),
        "bootstrap_probability_60d": bootstrap_probabilities.get(60, np.nan),
    }
    return pd.DataFrame([row], columns=VALIDATION_COLUMNS)


def _validation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = config.get("anti_overfit_validation", {})
    if raw is False:
        return {"enabled": False}
    if raw is True or raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("anti_overfit_validation must be a mapping, boolean, or omitted.")
    return raw


def run_anti_overfit_validation(
    *,
    config: Dict[str, Any],
    data: pd.DataFrame,
    output_dir: Path,
    standard_wf_table: pd.DataFrame,
    standard_stitched: pd.DataFrame,
    standard_summary: pd.DataFrame,
    grid_size: int,
    walk_forward_runner: WalkForwardRunner,
) -> Dict[str, pd.DataFrame]:
    validation_config = _validation_config(config)
    if validation_config.get("enabled") is False:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_symbol = str(config["benchmark_symbol"]).upper()
    base_cost = float(config.get("transaction_cost_bps", 0.0))

    variant_specs = [
        ("standard_5y_1y", 5, 1, "rolling", base_cost),
        ("alt_3y_1y", 3, 1, "rolling", base_cost),
        ("alt_7y_1y", 7, 1, "rolling", base_cost),
        ("anchored_expanding_1y", 5, 1, "anchored_expanding", base_cost),
        ("cost_25bps", 5, 1, "rolling", 25.0),
        ("cost_50bps", 5, 1, "rolling", 50.0),
    ]

    variant_rows: List[Dict[str, Any]] = []
    standard_5_stitched = pd.DataFrame()
    standard_5_wf = pd.DataFrame()
    standard_5_grid_size = 0
    for variant, train_years, test_years, mode, cost in variant_specs:
        row, wf_table, stitched, variant_grid_size = _run_variant(
            variant=variant,
            config=config,
            data=data,
            price_data=data,
            benchmark_symbol=benchmark_symbol,
            walk_forward_runner=walk_forward_runner,
            standard_wf_table=standard_wf_table,
            standard_stitched=standard_stitched,
            standard_grid_size=grid_size,
            train_years=train_years,
            test_years=test_years,
            walk_forward_mode=mode,
            transaction_cost_bps=cost,
        )
        variant_rows.append(row)
        if variant == "standard_5y_1y":
            standard_5_stitched = stitched
            standard_5_wf = wf_table
            standard_5_grid_size = variant_grid_size

    variant_summary_df = pd.DataFrame(variant_rows, columns=VARIANT_COLUMNS)
    validation_stitched = standard_5_stitched if not standard_5_stitched.empty else standard_stitched
    validation_source = "standard_5y_1y" if not standard_5_stitched.empty else "configured_walk_forward"

    bootstrap_df = block_bootstrap_summary(
        validation_stitched,
        data,
        benchmark_symbol,
        block_sizes=validation_config.get("bootstrap_block_sizes", (20, 60)),
        n_bootstrap=int(validation_config.get("bootstrap_iterations", 200)),
        seed=int(validation_config.get("bootstrap_seed", 42)),
        source_variant=validation_source,
    )
    subperiod_df = subperiod_summary(validation_stitched, data, benchmark_symbol)
    stability_df = parameter_stability_summary(
        standard_5_wf if not standard_5_wf.empty else standard_wf_table
    )
    validation_df = validation_summary(
        variant_summary=variant_summary_df,
        bootstrap=bootstrap_df,
        stability=stability_df,
        standard_wf_table=standard_5_wf if not standard_5_wf.empty else standard_wf_table,
        number_parameter_combinations_tested=int(standard_5_grid_size or grid_size),
    )

    validation_df.to_csv(output_dir / "validation_summary.csv", index=False)
    variant_summary_df.to_csv(output_dir / "walk_forward_variant_summary.csv", index=False)
    bootstrap_df.to_csv(output_dir / "bootstrap_summary.csv", index=False)
    subperiod_df.to_csv(output_dir / "subperiod_summary.csv", index=False)
    stability_df.to_csv(output_dir / "parameter_stability.csv", index=False)

    return {
        "validation_summary": validation_df,
        "walk_forward_variant_summary": variant_summary_df,
        "bootstrap_summary": bootstrap_df,
        "subperiod_summary": subperiod_df,
        "parameter_stability": stability_df,
        "standard_summary": standard_summary,
    }
