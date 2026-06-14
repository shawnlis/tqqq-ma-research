from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .fairness import build_constant_leverage_benchmark_summary
from .reports import compare_to_benchmark


DEFAULT_PERIODS = [
    {
        "label": "real_tqqq_period",
        "start_date": "2011-01-01",
        "end_date": None,
        "use_synthetic": True,
        "train_years": 5,
        "test_years": 1,
    },
    {
        "label": "synthetic_full_qqq_history",
        "start_date": "1999-03-10",
        "end_date": None,
        "use_synthetic": True,
        "train_years": 5,
        "test_years": 1,
    },
    {
        "label": "synthetic_2000_2002",
        "start_date": "1999-03-10",
        "end_date": "2002-12-31",
        "use_synthetic": True,
        "train_years": 1,
        "test_years": 1,
    },
    {
        "label": "synthetic_2008",
        "start_date": "2003-01-01",
        "end_date": "2008-12-31",
        "use_synthetic": True,
        "train_years": 5,
        "test_years": 1,
    },
    {
        "label": "synthetic_2020_crash_rebound",
        "start_date": "2015-01-01",
        "end_date": "2020-12-31",
        "use_synthetic": True,
        "train_years": 5,
        "test_years": 1,
    },
    {
        "label": "synthetic_2022",
        "start_date": "2017-01-01",
        "end_date": "2022-12-31",
        "use_synthetic": True,
        "train_years": 5,
        "test_years": 1,
    },
]


def _as_list(value: Any, default: Iterable[Any] = ()) -> List[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _unique(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        symbol = str(value).upper()
        if symbol not in out:
            out.append(symbol)
    return out


def _safe_path_part(value: Any) -> str:
    text = str(value).strip().lower().replace("^", "")
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join("".join(chars).split("_")).strip("_") or "case"


def _single_value(config: Dict[str, Any], key: str, default: Any) -> Any:
    params = config.get("vol_target_params") or config.get("strategy_params") or {}
    grid = config.get("parameter_grid") or {}
    if isinstance(params, dict) and key in params:
        return params[key]
    if isinstance(grid, dict) and key in grid:
        values = _as_list(grid[key])
        return values[0] if values else default
    return config.get(key, default)


def _target_ann_vol_values(config: Dict[str, Any]) -> List[float]:
    values = (
        config.get("target_ann_vol_values")
        or config.get("target_ann_vols")
        or config.get("vol_target_ann_vol")
    )
    if values is None:
        grid = config.get("parameter_grid") or {}
        if isinstance(grid, dict) and "target_ann_vol" in grid:
            values = grid["target_ann_vol"]
    return [float(value) for value in _as_list(values, [0.60])]


def _max_exposure_values(config: Dict[str, Any]) -> List[float]:
    values = config.get("max_exposure_values") or config.get("adaptive_max_exposures")
    if values is None:
        grid = config.get("parameter_grid") or {}
        if isinstance(grid, dict) and "max_exposure" in grid:
            values = grid["max_exposure"]
    return [float(value) for value in _as_list(values, [1.0, 1.25, 1.5, 2.0])]


def _periods(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = config.get("synthetic_periods") or DEFAULT_PERIODS
    periods: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        merged = {
            "use_synthetic": True,
            "train_years": config.get("train_years", 5),
            "test_years": config.get("test_years", 1),
            **item,
        }
        periods.append(merged)
    return periods


def _case_config(
    base_config: Dict[str, Any],
    *,
    period: Dict[str, Any],
    target_ann_vol: float,
    max_exposure: float,
    output_dir: Path,
) -> Dict[str, Any]:
    synthetic_base_symbol = str(base_config.get("synthetic_base_symbol", "QQQ")).upper()
    synthetic_target_symbol = str(base_config.get("synthetic_target_symbol", "TQQQ")).upper()
    use_synthetic = bool(period.get("use_synthetic", True))
    case = {
        "experiment_name": (
            f"{base_config.get('experiment_name', 'voltarget_synthetic_history')}_"
            f"{period.get('label', 'period')}_max_{max_exposure:g}_vol_{target_ann_vol:g}"
        ),
        "symbols": _unique(
            base_config.get("symbols")
            or [synthetic_base_symbol, synthetic_target_symbol, "CASH"]
        ),
        "start_date": str(period.get("start_date", base_config.get("start_date", "1999-03-10"))),
        "end_date": period.get("end_date", base_config.get("end_date")),
        "strategy_name": "vol_target",
        "parameter_grid": {
            "target_ann_vol": [float(target_ann_vol)],
            "realized_vol_window": [int(_single_value(base_config, "realized_vol_window", 20))],
            "min_exposure": [float(_single_value(base_config, "min_exposure", 0.25))],
            "max_exposure": [float(max_exposure)],
            "trend_window": [int(_single_value(base_config, "trend_window", 150))],
            "momentum_window": [int(_single_value(base_config, "momentum_window", 63))],
            "trend_multiplier_below_ma": [float(_single_value(base_config, "trend_multiplier_below_ma", 0.50))],
            "momentum_boost": [float(_single_value(base_config, "momentum_boost", 1.15))],
            "crash_vol_cutoff": [float(_single_value(base_config, "crash_vol_cutoff", 0.050))],
            "crash_exposure": [float(_single_value(base_config, "crash_exposure", 0.25))],
            "risk_off_symbol": [str(_single_value(base_config, "risk_off_symbol", "CASH")).upper()],
            "risk_off_weight": [float(_single_value(base_config, "risk_off_weight", 0.0))],
        },
        "benchmark_symbol": synthetic_target_symbol,
        "transaction_cost_bps": float(base_config.get("transaction_cost_bps", 25.0)),
        "train_years": int(period.get("train_years", base_config.get("train_years", 5))),
        "test_years": int(period.get("test_years", base_config.get("test_years", 1))),
        "objective": str(base_config.get("objective", "objective_final_ratio")),
        "output_dir": str(output_dir),
        "use_synthetic_leverage": use_synthetic,
        "synthetic_base_symbol": synthetic_base_symbol,
        "synthetic_target_symbol": synthetic_target_symbol,
        "real_leveraged_symbol": str(base_config.get("real_leveraged_symbol", synthetic_target_symbol)).upper(),
        "synthetic_leverage": float(base_config.get("synthetic_leverage", 3.0)),
        "synthetic_expense_ratio": float(base_config.get("synthetic_expense_ratio", 0.0095)),
        "synthetic_financing_spread": float(base_config.get("synthetic_financing_spread", 0.0)),
        "anti_overfit_validation": {"enabled": False},
    }
    for optional_key in ("data_csv", "cache_dir", "use_csv_if_exists", "download_ohlc", "execution_model"):
        if optional_key in base_config:
            case[optional_key] = base_config[optional_key]
    return case


def _blank_row(**values: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "period_label": "",
        "status": "error",
        "error": "",
        "use_synthetic_leverage": np.nan,
        "benchmark_source": "",
        "period_start_date": "",
        "period_end_date": "",
        "start_date": "",
        "end_date": "",
        "train_years": np.nan,
        "test_years": np.nan,
        "target_ann_vol": np.nan,
        "max_exposure": np.nan,
        "walk_forward_windows": 0,
        "stitched_rows": 0,
        "benchmark_symbol": "",
        "strategy_final_equity": np.nan,
        "benchmark_final_equity": np.nan,
        "final_equity_ratio": np.nan,
        "strategy_cagr": np.nan,
        "benchmark_cagr": np.nan,
        "excess_cagr": np.nan,
        "strategy_sharpe": np.nan,
        "benchmark_sharpe": np.nan,
        "strategy_max_dd": np.nan,
        "benchmark_max_dd": np.nan,
        "raw_outperformance_pass": np.nan,
        "ratio_versus_1x_tqqq": np.nan,
        "ratio_versus_same_max_constant_tqqq": np.nan,
        "max_drawdown_difference_vs_same_max_constant_tqqq": np.nan,
        "cagr_difference_vs_same_max_constant_tqqq": np.nan,
        "sharpe_difference_vs_same_max_constant_tqqq": np.nan,
        "same_max_constant_final_equity": np.nan,
        "same_max_constant_max_dd": np.nan,
        "case_output_dir": "",
    }
    row.update(values)
    return row


def _run_case(
    *,
    base_config: Dict[str, Any],
    period: Dict[str, Any],
    target_ann_vol: float,
    max_exposure: float,
    output_dir: Path,
) -> Dict[str, Any]:
    from .experiments import _asset_config, _load_price_data, _run_walk_forward

    label = str(period.get("label", "period"))
    case_dir = output_dir / "case_outputs" / f"{_safe_path_part(label)}_max_{max_exposure:g}_vol_{target_ann_vol:g}"
    case_config = _case_config(
        base_config,
        period=period,
        target_ann_vol=target_ann_vol,
        max_exposure=max_exposure,
        output_dir=case_dir,
    )
    benchmark_source = (
        f"synthetic_{case_config['synthetic_leverage']:g}x_{case_config['synthetic_base_symbol']}"
        if bool(case_config.get("use_synthetic_leverage", False))
        else f"real_{case_config['benchmark_symbol']}"
    )
    metadata = {
        "period_label": label,
        "use_synthetic_leverage": bool(case_config.get("use_synthetic_leverage", False)),
        "benchmark_source": benchmark_source,
        "period_start_date": case_config["start_date"],
        "period_end_date": case_config.get("end_date") or "",
        "start_date": case_config["start_date"],
        "end_date": case_config.get("end_date") or "",
        "train_years": int(case_config["train_years"]),
        "test_years": int(case_config["test_years"]),
        "target_ann_vol": float(target_ann_vol),
        "max_exposure": float(max_exposure),
        "benchmark_symbol": str(case_config["benchmark_symbol"]).upper(),
        "case_output_dir": str(case_dir),
    }

    try:
        case_dir.mkdir(parents=True, exist_ok=True)
        data = _load_price_data(case_config)
        wf_table, stitched, grid_size = _run_walk_forward(case_config, data)
        if stitched.empty:
            return _blank_row(**metadata, status="no_result", error="stitched equity is empty")

        summary, yearly = compare_to_benchmark(
            stitched,
            data,
            benchmark_symbol=str(case_config["benchmark_symbol"]).upper(),
        )
        asset_config = _asset_config(case_config)
        fairness = build_constant_leverage_benchmark_summary(
            stitched=stitched,
            price_data=data,
            transaction_cost_bps=float(case_config["transaction_cost_bps"]),
            wf_table=wf_table,
            trade_asset=asset_config.trade_asset,
            config=case_config,
        )
        same_max = fairness.loc[fairness["is_same_max_exposure_benchmark"]].iloc[0]
        result = _blank_row(
            **metadata,
            status="ok",
            error="",
            walk_forward_windows=int(len(wf_table)),
            stitched_rows=int(len(stitched)),
            grid_size=int(grid_size),
        )
        result.update(summary.iloc[0].to_dict())
        result.update(
            {
                "benchmark_source": benchmark_source,
                "ratio_versus_1x_tqqq": float(same_max["ratio_versus_1x_tqqq"]),
                "ratio_versus_same_max_constant_tqqq": float(
                    same_max["ratio_versus_same_max_constant_tqqq"]
                ),
                "max_drawdown_difference_vs_same_max_constant_tqqq": float(
                    same_max["max_drawdown_difference_vs_same_max_constant_tqqq"]
                ),
                "cagr_difference_vs_same_max_constant_tqqq": float(
                    same_max["cagr_difference_vs_same_max_constant_tqqq"]
                ),
                "sharpe_difference_vs_same_max_constant_tqqq": float(
                    same_max["sharpe_difference_vs_same_max_constant_tqqq"]
                ),
                "same_max_constant_final_equity": float(same_max["constant_final_equity"]),
                "same_max_constant_max_dd": float(same_max["constant_max_dd"]),
            }
        )

        if bool(base_config.get("save_case_outputs", False)):
            stitched.to_csv(case_dir / "stitched_equity.csv")
            wf_table.to_csv(case_dir / "walk_forward_windows.csv", index=False)
            summary.to_csv(case_dir / "same_period_benchmark_summary.csv", index=False)
            yearly.to_csv(case_dir / "yearly_returns.csv", index=False)
            fairness.to_csv(case_dir / "constant_leverage_benchmark_summary.csv", index=False)
        return result
    except Exception as exc:
        return _blank_row(**metadata, status="error", error=str(exc))


def run_voltarget_synthetic_history(
    config: Dict[str, Any],
    *,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(str(config.get("output_dir", "outputs/voltarget_synthetic_history")))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for period in _periods(config):
        for target_ann_vol in _target_ann_vol_values(config):
            for max_exposure in _max_exposure_values(config):
                rows.append(
                    _run_case(
                        base_config=config,
                        period=period,
                        target_ann_vol=target_ann_vol,
                        max_exposure=max_exposure,
                        output_dir=output_dir,
                    )
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "voltarget_synthetic_history.csv", index=False)
    _write_report(output_dir / "voltarget_synthetic_history_report.md", summary, config)
    run_config = {
        "command": "run-config",
        "experiment_type": "voltarget_synthetic_history",
        "config_path": str(config_path) if config_path else None,
        "actual_runtime_seconds": time.perf_counter() - started,
        **config,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Saved VolTarget synthetic-history outputs to {output_dir}")
    return {
        "experiment_name": str(config.get("experiment_name", "voltarget_synthetic_history")),
        "strategy_name": "vol_target",
        "status": "ok" if not summary.empty and (summary["status"] == "ok").any() else "error",
        "output_dir": str(output_dir),
        "rows": int(len(summary)),
        "successful_rows": int((summary["status"] == "ok").sum()) if "status" in summary.columns else 0,
    }


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(number):
        return ""
    return f"{number:.6f}"


def _write_report(path: Path, summary: pd.DataFrame, config: Dict[str, Any]) -> None:
    ok = summary[summary["status"] == "ok"].copy() if "status" in summary.columns else pd.DataFrame()
    best = ok.sort_values("final_equity_ratio", ascending=False).head(1)
    best_line = "No successful VolTarget synthetic-history rows completed."
    if not best.empty:
        row = best.iloc[0]
        best_line = (
            f"Best row: `{row['period_label']}` max_exposure `{float(row['max_exposure']):g}`, "
            f"target_ann_vol `{float(row['target_ann_vol']):g}`, "
            f"final_equity_ratio `{_fmt(row['final_equity_ratio'])}`."
        )

    lines = [
        "# VolTarget Synthetic-History Validation",
        "",
        "This diagnostic tests VolTarget against synthetic 3x QQQ buy-and-hold across real-period and pre-TQQQ synthetic periods.",
        "It is not an investment recommendation.",
        "",
        f"- Synthetic base symbol: `{config.get('synthetic_base_symbol', 'QQQ')}`",
        f"- Synthetic leverage: `{config.get('synthetic_leverage', 3.0)}`",
        f"- Benchmark: `synthetic 3x QQQ` when `use_synthetic_leverage` is true",
        f"- {best_line}",
        "",
        "## Full-History And Crash-Period Rows",
        "",
        "| Period | Max Exposure | Target Vol | Status | Final Equity Ratio | Same-Max Constant Ratio | Strategy Max DD | Benchmark Max DD |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            f"{row.get('period_label', '')} | "
            f"{_fmt(row.get('max_exposure'))} | "
            f"{_fmt(row.get('target_ann_vol'))} | "
            f"{row.get('status', '')} | "
            f"{_fmt(row.get('final_equity_ratio'))} | "
            f"{_fmt(row.get('ratio_versus_same_max_constant_tqqq'))} | "
            f"{_fmt(row.get('strategy_max_dd'))} | "
            f"{_fmt(row.get('benchmark_max_dd'))} |"
        )
    if not ok.empty:
        rejected = ok[ok["final_equity_ratio"] <= 1.0]
        verdict = (
            "At least one synthetic-history row failed to beat synthetic 3x QQQ buy-and-hold."
            if not rejected.empty
            else "All successful synthetic-history rows beat synthetic 3x QQQ buy-and-hold."
        )
    else:
        verdict = "No successful rows were available for a verdict."
    lines.extend(["", "## Verdict", "", verdict, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
