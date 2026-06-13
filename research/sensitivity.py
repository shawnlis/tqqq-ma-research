from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .execution import CLOSE_TO_CLOSE_SHIFTED, EXECUTION_MODELS, normalize_execution_model
from .reports import compare_to_benchmark

WalkForwardRunner = Callable[[Dict[str, Any], pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, int]]

COST_SCENARIOS = (0.0, 10.0, 25.0, 50.0, 100.0)

SENSITIVITY_COLUMNS = [
    "scenario",
    "status",
    "transaction_cost_bps",
    "requested_execution_model",
    "execution_model",
    "execution_model_warning",
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
    "strategy_beats_benchmark_at_25bps",
    "zero_bps_only_outperformance",
    "error",
]


def _blank_row(**values: Any) -> Dict[str, Any]:
    row = {column: np.nan for column in SENSITIVITY_COLUMNS}
    row.update(values)
    return row


def _execution_metadata(stitched: pd.DataFrame, requested_model: str) -> tuple[str, str]:
    if stitched.empty:
        return requested_model, ""
    effective = (
        str(stitched["execution_model"].dropna().iloc[0])
        if "execution_model" in stitched.columns and not stitched["execution_model"].dropna().empty
        else requested_model
    )
    warning = (
        str(stitched["execution_model_warning"].dropna().iloc[0])
        if "execution_model_warning" in stitched.columns
        and not stitched["execution_model_warning"].dropna().empty
        and str(stitched["execution_model_warning"].dropna().iloc[0])
        else ""
    )
    return effective, warning


def _summary_row(
    *,
    scenario: str,
    config: Dict[str, Any],
    stitched: pd.DataFrame,
    wf_table: pd.DataFrame,
    grid_size: int,
    data: pd.DataFrame,
    benchmark_symbol: str,
) -> Dict[str, Any]:
    requested_model = normalize_execution_model(config.get("execution_model", CLOSE_TO_CLOSE_SHIFTED))
    effective_model, warning = _execution_metadata(stitched, requested_model)
    metadata = {
        "scenario": scenario,
        "status": "ok" if not stitched.empty else "no_result",
        "transaction_cost_bps": float(config.get("transaction_cost_bps", np.nan)),
        "requested_execution_model": requested_model,
        "execution_model": effective_model,
        "execution_model_warning": warning,
        "grid_size": int(grid_size),
        "walk_forward_windows": int(len(wf_table)),
        "stitched_rows": int(len(stitched)),
        "benchmark_symbol": benchmark_symbol,
        "error": "" if not stitched.empty else "scenario produced no stitched equity",
    }
    if stitched.empty:
        return _blank_row(**metadata)

    summary, _ = compare_to_benchmark(stitched, data, benchmark_symbol=benchmark_symbol)
    row = _blank_row(**metadata)
    for key, value in summary.iloc[0].to_dict().items():
        if key in row:
            row[key] = value
    row.update(metadata)
    return row


def _same_scenario(config: Dict[str, Any], *, cost: float, execution_model: str) -> bool:
    return (
        float(config.get("transaction_cost_bps", 0.0)) == float(cost)
        and normalize_execution_model(config.get("execution_model", CLOSE_TO_CLOSE_SHIFTED))
        == normalize_execution_model(execution_model)
    )


def _run_scenario(
    *,
    scenario: str,
    base_config: Dict[str, Any],
    data: pd.DataFrame,
    standard_wf_table: pd.DataFrame,
    standard_stitched: pd.DataFrame,
    standard_grid_size: int,
    walk_forward_runner: WalkForwardRunner,
    transaction_cost_bps: float,
    execution_model: str,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["transaction_cost_bps"] = float(transaction_cost_bps)
    config["execution_model"] = normalize_execution_model(execution_model)
    benchmark_symbol = str(config["benchmark_symbol"]).upper()

    try:
        if _same_scenario(base_config, cost=transaction_cost_bps, execution_model=execution_model):
            wf_table, stitched, grid_size = standard_wf_table, standard_stitched, standard_grid_size
        else:
            wf_table, stitched, grid_size = walk_forward_runner(config, data)
        return _summary_row(
            scenario=scenario,
            config=config,
            stitched=stitched,
            wf_table=wf_table,
            grid_size=grid_size,
            data=data,
            benchmark_symbol=benchmark_symbol,
        )
    except Exception as exc:
        return _blank_row(
            scenario=scenario,
            status="error",
            transaction_cost_bps=float(transaction_cost_bps),
            requested_execution_model=normalize_execution_model(execution_model),
            execution_model=np.nan,
            execution_model_warning="",
            grid_size=np.nan,
            walk_forward_windows=0,
            stitched_rows=0,
            benchmark_symbol=benchmark_symbol,
            error=str(exc),
        )


def _annotate_cost_acceptance(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
    if df.empty:
        return df

    ratios = pd.to_numeric(df["final_equity_ratio"], errors="coerce")
    zero_pass = bool(
        ((df["transaction_cost_bps"] == 0.0) & (ratios > 1.0)).any()
    )
    twenty_five_pass = bool(
        ((df["transaction_cost_bps"] == 25.0) & (ratios > 1.0)).any()
    )
    df["strategy_beats_benchmark_at_25bps"] = twenty_five_pass
    df["zero_bps_only_outperformance"] = bool(zero_pass and not twenty_five_pass)
    return df


def run_cost_and_execution_sensitivity(
    *,
    config: Dict[str, Any],
    data: pd.DataFrame,
    output_dir: Path,
    standard_wf_table: pd.DataFrame,
    standard_stitched: pd.DataFrame,
    grid_size: int,
    walk_forward_runner: WalkForwardRunner,
    cost_scenarios: Iterable[float] = COST_SCENARIOS,
    execution_models: Iterable[str] = EXECUTION_MODELS,
) -> Dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_execution_model = normalize_execution_model(config.get("execution_model", CLOSE_TO_CLOSE_SHIFTED))
    base_cost = float(config.get("transaction_cost_bps", 0.0))

    cost_rows = [
        _run_scenario(
            scenario=f"{float(cost):g}bps",
            base_config=config,
            data=data,
            standard_wf_table=standard_wf_table,
            standard_stitched=standard_stitched,
            standard_grid_size=grid_size,
            walk_forward_runner=walk_forward_runner,
            transaction_cost_bps=float(cost),
            execution_model=base_execution_model,
        )
        for cost in cost_scenarios
    ]
    cost_df = _annotate_cost_acceptance(cost_rows)

    execution_rows = [
        _run_scenario(
            scenario=normalize_execution_model(model),
            base_config=config,
            data=data,
            standard_wf_table=standard_wf_table,
            standard_stitched=standard_stitched,
            standard_grid_size=grid_size,
            walk_forward_runner=walk_forward_runner,
            transaction_cost_bps=base_cost,
            execution_model=normalize_execution_model(model),
        )
        for model in execution_models
    ]
    execution_df = pd.DataFrame(execution_rows, columns=SENSITIVITY_COLUMNS)

    cost_df.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    execution_df.to_csv(output_dir / "execution_model_sensitivity.csv", index=False)
    return {
        "cost_sensitivity": cost_df,
        "execution_model_sensitivity": execution_df,
    }
