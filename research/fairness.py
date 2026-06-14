from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import _portfolio_from_trade_weight
from .execution import CLOSE_TO_CLOSE_SHIFTED, normalize_execution_model
from .metrics import annualized_return, calmar_ratio, max_drawdown, sharpe_ratio
from .strategies.constant_exposure_strategy import (
    ConstantExposureParams,
    build_constant_exposure_position,
)

CONSTANT_TQQQ_EXPOSURES: Tuple[float, ...] = (1.0, 1.25, 1.5, 2.0)
CONSTANT_TQQQ_LABELS: Dict[float, str] = {
    1.0: "buy_and_hold_tqqq_1.0",
    1.25: "constant_tqqq_1.25",
    1.5: "constant_tqqq_1.5",
    2.0: "constant_tqqq_2.0",
}


def _format_exposure(value: float) -> str:
    return f"{float(value):g}"


def constant_exposure_label(exposure: float) -> str:
    rounded = round(float(exposure), 8)
    return CONSTANT_TQQQ_LABELS.get(rounded, f"constant_tqqq_{_format_exposure(rounded)}")


def _coerce_stitched(stitched: pd.DataFrame) -> pd.DataFrame:
    if stitched.empty:
        raise ValueError("stitched equity is empty; constant leverage fairness cannot be computed.")
    if "ret" not in stitched.columns:
        raise ValueError("stitched equity must contain a ret column.")
    out = stitched.sort_index().copy()
    if "equity" not in out.columns:
        out["equity"] = (1.0 + out["ret"].astype(float)).cumprod()
    return out


def constant_exposure_backtest(
    price_data: pd.DataFrame,
    *,
    dates: Optional[pd.Index] = None,
    exposure: float = 1.0,
    trade_asset: str = "TQQQ",
    transaction_cost_bps: float = 0.0,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    trade_asset = str(trade_asset).upper()
    if trade_asset not in price_data.columns:
        raise ValueError(f"price_data is missing trade asset column: {trade_asset}")

    if dates is None:
        data = price_data.sort_index().copy()
    else:
        date_index = pd.Index(dates)
        missing_dates = date_index.difference(price_data.index)
        if len(missing_dates) > 0:
            raise ValueError(f"price_data is missing constant exposure date: {missing_dates[0]}")
        data = price_data.loc[date_index].copy()

    data.attrs.update(getattr(price_data, "attrs", {}))
    execution_model = normalize_execution_model(execution_model or CLOSE_TO_CLOSE_SHIFTED)
    position = build_constant_exposure_position(
        data.index,
        ConstantExposureParams(exposure=float(exposure), trade_asset=trade_asset),
    )
    bt = _portfolio_from_trade_weight(
        data=data,
        trade_weight=position,
        transaction_cost_bps=float(transaction_cost_bps),
        risk_off_symbol="CASH",
        risk_off_weight=0.0,
        trade_asset=trade_asset,
        execution_model=execution_model,
    )
    bt["equity"] = (1.0 + bt["ret"].astype(float)).cumprod()
    bt["constant_exposure"] = float(exposure)
    bt["benchmark_name"] = constant_exposure_label(float(exposure))
    return bt


def _performance_from_returns(bt: pd.DataFrame) -> Dict[str, float]:
    ret = bt["ret"].astype(float)
    equity = bt["equity"].astype(float) if "equity" in bt.columns else (1.0 + ret).cumprod()
    cagr = annualized_return(equity)
    dd = max_drawdown(equity)
    return {
        "final_equity": float(equity.iloc[-1]),
        "cagr": cagr,
        "sharpe": sharpe_ratio(ret),
        "max_dd": dd,
        "calmar": calmar_ratio(cagr, dd),
        "total_turnover": float(bt["turnover"].sum()) if "turnover" in bt.columns else np.nan,
        "total_transaction_cost": float(bt["cost"].sum()) if "cost" in bt.columns else np.nan,
    }


def infer_selected_max_exposure(
    wf_table: Optional[pd.DataFrame] = None,
    stitched: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    if wf_table is not None and not wf_table.empty and "max_exposure" in wf_table.columns:
        values = pd.to_numeric(wf_table["max_exposure"], errors="coerce").dropna()
        if not values.empty:
            return float(values.max())

    if stitched is not None and not stitched.empty:
        for column in ("target_exposure", "position"):
            if column in stitched.columns:
                values = pd.to_numeric(stitched[column], errors="coerce").dropna()
                if not values.empty:
                    return float(values.max())

    if config:
        params = config.get("strategy_params") or config.get("parameter_grid") or {}
        if isinstance(params, dict) and "max_exposure" in params:
            value = params["max_exposure"]
            if isinstance(value, (list, tuple)):
                numeric = pd.to_numeric(pd.Series(value), errors="coerce").dropna()
                if not numeric.empty:
                    return float(numeric.max())
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return 1.0


def _nearest_exposure(value: float, exposures: Iterable[float]) -> float:
    choices = [float(exposure) for exposure in exposures]
    if not choices:
        raise ValueError("At least one constant exposure benchmark is required.")
    return min(choices, key=lambda exposure: abs(exposure - float(value)))


def build_constant_leverage_benchmark_summary(
    *,
    stitched: pd.DataFrame,
    price_data: pd.DataFrame,
    transaction_cost_bps: float,
    wf_table: Optional[pd.DataFrame] = None,
    selected_max_exposure: Optional[float] = None,
    trade_asset: str = "TQQQ",
    execution_model: Optional[str] = None,
    exposures: Iterable[float] = CONSTANT_TQQQ_EXPOSURES,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    stitched = _coerce_stitched(stitched)
    exposures = tuple(float(exposure) for exposure in exposures)
    if not exposures:
        raise ValueError("At least one constant exposure benchmark is required.")

    selected_max = (
        float(selected_max_exposure)
        if selected_max_exposure is not None
        else infer_selected_max_exposure(wf_table=wf_table, stitched=stitched, config=config)
    )
    same_max_exposure = _nearest_exposure(selected_max, exposures)
    same_max_match_type = "exact" if np.isclose(selected_max, same_max_exposure) else "nearest_available"

    strategy_bt = pd.DataFrame(
        {
            "ret": stitched["ret"].astype(float),
            "equity": stitched["equity"].astype(float),
            "turnover": stitched["turnover"].astype(float) if "turnover" in stitched.columns else 0.0,
            "cost": stitched["cost"].astype(float) if "cost" in stitched.columns else 0.0,
        },
        index=stitched.index,
    )
    strategy_metrics = _performance_from_returns(strategy_bt)

    constant_backtests: Dict[float, pd.DataFrame] = {}
    constant_metrics: Dict[float, Dict[str, float]] = {}
    for exposure in exposures:
        bt = constant_exposure_backtest(
            price_data,
            dates=stitched.index,
            exposure=exposure,
            trade_asset=trade_asset,
            transaction_cost_bps=float(transaction_cost_bps),
            execution_model=execution_model,
        )
        constant_backtests[exposure] = bt
        constant_metrics[exposure] = _performance_from_returns(bt)

    one_x_exposure = _nearest_exposure(1.0, exposures)
    one_x_metrics = constant_metrics[one_x_exposure]
    same_max_metrics = constant_metrics[same_max_exposure]
    ratio_vs_1x = (
        strategy_metrics["final_equity"] / one_x_metrics["final_equity"]
        if one_x_metrics["final_equity"] != 0.0
        else np.nan
    )
    ratio_vs_same_max = (
        strategy_metrics["final_equity"] / same_max_metrics["final_equity"]
        if same_max_metrics["final_equity"] != 0.0
        else np.nan
    )
    max_dd_diff = strategy_metrics["max_dd"] - same_max_metrics["max_dd"]
    cagr_diff = strategy_metrics["cagr"] - same_max_metrics["cagr"]
    sharpe_diff = strategy_metrics["sharpe"] - same_max_metrics["sharpe"]

    rows = []
    for exposure in exposures:
        metrics = constant_metrics[exposure]
        rows.append(
            {
                "benchmark_name": constant_exposure_label(exposure),
                "constant_exposure": exposure,
                "selected_max_exposure": selected_max,
                "same_max_exposure": same_max_exposure,
                "same_max_exposure_match_type": same_max_match_type,
                "is_1x_tqqq_benchmark": bool(np.isclose(exposure, one_x_exposure)),
                "is_same_max_exposure_benchmark": bool(np.isclose(exposure, same_max_exposure)),
                "start_date": stitched.index[0].date().isoformat(),
                "end_date": stitched.index[-1].date().isoformat(),
                "stitched_rows": int(len(stitched)),
                "transaction_cost_bps": float(transaction_cost_bps),
                "strategy_final_equity": strategy_metrics["final_equity"],
                "strategy_cagr": strategy_metrics["cagr"],
                "strategy_sharpe": strategy_metrics["sharpe"],
                "strategy_max_dd": strategy_metrics["max_dd"],
                "strategy_calmar": strategy_metrics["calmar"],
                "strategy_total_turnover": strategy_metrics["total_turnover"],
                "strategy_total_transaction_cost": strategy_metrics["total_transaction_cost"],
                "constant_final_equity": metrics["final_equity"],
                "constant_cagr": metrics["cagr"],
                "constant_sharpe": metrics["sharpe"],
                "constant_max_dd": metrics["max_dd"],
                "constant_calmar": metrics["calmar"],
                "constant_total_turnover": metrics["total_turnover"],
                "constant_total_transaction_cost": metrics["total_transaction_cost"],
                "ratio_versus_1x_tqqq": ratio_vs_1x,
                "ratio_versus_same_max_constant_tqqq": ratio_vs_same_max,
                "max_drawdown_difference_vs_same_max_constant_tqqq": max_dd_diff,
                "cagr_difference_vs_same_max_constant_tqqq": cagr_diff,
                "sharpe_difference_vs_same_max_constant_tqqq": sharpe_diff,
            }
        )

    return pd.DataFrame(rows)


def write_voltarget_fairness_outputs(
    *,
    output_dir: Path,
    stitched: pd.DataFrame,
    price_data: pd.DataFrame,
    transaction_cost_bps: float,
    wf_table: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    trade_asset: str = "TQQQ",
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_constant_leverage_benchmark_summary(
        stitched=stitched,
        price_data=price_data,
        transaction_cost_bps=transaction_cost_bps,
        wf_table=wf_table,
        trade_asset=trade_asset,
        execution_model=execution_model,
        config=config,
    )
    summary.to_csv(output_dir / "constant_leverage_benchmark_summary.csv", index=False)
    _write_voltarget_fairness_report(output_dir / "voltarget_fairness_report.md", summary)
    return summary


def _write_voltarget_fairness_report(path: Path, summary: pd.DataFrame) -> None:
    same_max = summary[summary["is_same_max_exposure_benchmark"]]
    one_x = summary[summary["is_1x_tqqq_benchmark"]]
    same_row = same_max.iloc[0] if not same_max.empty else summary.iloc[0]
    one_x_row = one_x.iloc[0] if not one_x.empty else summary.iloc[0]

    lines = [
        "# VolTarget Fairness Report",
        "",
        "This diagnostic compares the VolTarget candidate against constant TQQQ exposure baselines over the exact stitched dates.",
        "The same-max-exposure benchmark is the constant TQQQ exposure closest to the selected VolTarget max_exposure.",
        "",
        "## Key Comparisons",
        "",
        f"- Ratio versus 1.0x TQQQ: `{float(one_x_row['ratio_versus_1x_tqqq']):.6f}`",
        (
            "- Ratio versus same-max-exposure benchmark "
            f"({float(same_row['same_max_exposure']):g}x): "
            f"`{float(same_row['ratio_versus_same_max_constant_tqqq']):.6f}`"
        ),
        (
            "- Max drawdown difference versus same-max-exposure benchmark: "
            f"`{float(same_row['max_drawdown_difference_vs_same_max_constant_tqqq']):.6f}`"
        ),
        (
            "- CAGR difference versus same-max-exposure benchmark: "
            f"`{float(same_row['cagr_difference_vs_same_max_constant_tqqq']):.6f}`"
        ),
        (
            "- Sharpe difference versus same-max-exposure benchmark: "
            f"`{float(same_row['sharpe_difference_vs_same_max_constant_tqqq']):.6f}`"
        ),
        "",
        "## Constant Exposure Benchmarks",
        "",
        "| Benchmark | Exposure | Final Equity | CAGR | Sharpe | Max Drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            f"{row['benchmark_name']} | "
            f"{float(row['constant_exposure']):g} | "
            f"{float(row['constant_final_equity']):.6f} | "
            f"{float(row['constant_cagr']):.6f} | "
            f"{float(row['constant_sharpe']):.6f} | "
            f"{float(row['constant_max_dd']):.6f} |"
        )
    lines.extend(
        [
            "",
            "This is not an investment recommendation. It is a leverage fairness check for empirical research.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
