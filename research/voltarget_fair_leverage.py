from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .assets import asset_config_from_row
from .backtest import make_vol_target_eval_piece
from .data import CASH_SYMBOL, add_cash_series, load_prices
from .execution import CLOSE_TO_CLOSE_SHIFTED, normalize_execution_model, ohlc_columns
from .fairness import constant_exposure_backtest, infer_selected_max_exposure
from .metrics import annualized_return, calmar_ratio, max_drawdown, sharpe_ratio
from .strategies.drawdown_governor import DrawdownGovernorParams
from .strategies.market_internals import MarketInternalsParams
from .strategies.rebound import ReboundParams
from .strategies.vol_target_strategy import VolTargetParams


VOLTARGET_CATEGORIES = {"tqqq_voltarget", "tqqq_voltarget_governor"}
DEFAULT_CONSTANT_EXPOSURES = (1.0, 1.25, 1.5, 2.0)
SIMPLE_NO_TREND_CRASH_VOL_CUTOFF = 1.0e9

FAIR_LEVERAGE_COLUMNS = [
    "family",
    "category",
    "classification",
    "baseline_output_dir",
    "start_date",
    "end_date",
    "date_count",
    "transaction_cost_bps",
    "execution_model",
    "trade_asset",
    "selected_max_exposure",
    "realized_average_exposure",
    "strategy_final_equity",
    "ratio_vs_1x_tqqq",
    "ratio_vs_same_max_constant_tqqq",
    "ratio_vs_same_avg_exposure_constant_tqqq",
    "ratio_vs_simple_voltarget_no_trend",
    "cagr_diff_vs_1x_tqqq",
    "cagr_diff_vs_same_max_constant_tqqq",
    "cagr_diff_vs_same_avg_exposure_constant_tqqq",
    "cagr_diff_vs_simple_voltarget_no_trend",
    "max_dd_diff_vs_1x_tqqq",
    "max_dd_diff_vs_same_max_constant_tqqq",
    "max_dd_diff_vs_same_avg_exposure_constant_tqqq",
    "max_dd_diff_vs_simple_voltarget_no_trend",
    "sharpe_diff_vs_1x_tqqq",
    "sharpe_diff_vs_same_max_constant_tqqq",
    "sharpe_diff_vs_same_avg_exposure_constant_tqqq",
    "sharpe_diff_vs_simple_voltarget_no_trend",
    "annual_turnover_diff_vs_1x_tqqq",
    "annual_turnover_diff_vs_same_max_constant_tqqq",
    "annual_turnover_diff_vs_same_avg_exposure_constant_tqqq",
    "annual_turnover_diff_vs_simple_voltarget_no_trend",
    "transaction_cost_drag_vs_1x_tqqq",
    "transaction_cost_drag_vs_same_max_constant_tqqq",
    "transaction_cost_drag_vs_same_avg_exposure_constant_tqqq",
    "transaction_cost_drag_vs_simple_voltarget_no_trend",
    "ex_2022_ratio_vs_1x_tqqq",
    "ex_2022_ratio_vs_same_max_constant_tqqq",
    "ex_2022_ratio_vs_same_avg_exposure_constant_tqqq",
    "ex_2022_ratio_vs_simple_voltarget_no_trend",
    "ratio_2022_only_vs_1x_tqqq",
    "ratio_2022_only_vs_same_max_constant_tqqq",
    "ratio_2022_only_vs_same_avg_exposure_constant_tqqq",
    "ratio_2022_only_vs_simple_voltarget_no_trend",
    "stage2_final_equity_ratio",
    "stage2_final_equity_ratio_ex_2022",
    "stage2_min_leave_one_year_out_ratio",
    "stage2_max_drawdown_improvement_ex_2022",
    "stage2_one_year_dominated",
    "stage2_largest_contribution_year",
    "stage2_largest_single_year_contribution_share",
    "governor_adds_value_outside_2022",
]


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _years_from_index(index: pd.Index) -> float:
    if len(index) < 2:
        return np.nan
    return max((pd.Timestamp(index[-1]) - pd.Timestamp(index[0])).days / 365.25, 1.0 / 252.0)


def _performance_from_frame(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty or "ret" not in frame.columns:
        return {
            "final_equity": np.nan,
            "cagr": np.nan,
            "sharpe": np.nan,
            "max_dd": np.nan,
            "calmar": np.nan,
            "annual_turnover": np.nan,
            "total_transaction_cost": np.nan,
        }
    ret = frame["ret"].astype(float)
    equity = frame["equity"].astype(float) if "equity" in frame.columns else (1.0 + ret).cumprod()
    cagr = annualized_return(equity)
    dd = max_drawdown(equity)
    years = _years_from_index(frame.index)
    turnover = frame["turnover"].astype(float).sum() if "turnover" in frame.columns else np.nan
    cost = frame["cost"].astype(float).sum() if "cost" in frame.columns else np.nan
    return {
        "final_equity": float(equity.iloc[-1]),
        "cagr": cagr,
        "sharpe": sharpe_ratio(ret),
        "max_dd": dd,
        "calmar": calmar_ratio(cagr, dd),
        "annual_turnover": float(turnover / years) if pd.notna(turnover) and years > 0 else np.nan,
        "total_transaction_cost": float(cost) if pd.notna(cost) else np.nan,
    }


def realized_average_exposure(stitched: pd.DataFrame) -> float:
    """Average realized trade-asset exposure over stitched dates."""
    for column in ("trade_weight", "tqqq_weight", "position"):
        if column in stitched.columns:
            values = pd.to_numeric(stitched[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.mean())
    return np.nan


def simple_voltarget_no_trend_params_from_row(row: pd.Series) -> VolTargetParams:
    """Reuse selected VolTarget sizing params while removing trend, momentum, governor, and internals controls."""
    return VolTargetParams(
        target_ann_vol=float(row["target_ann_vol"]),
        realized_vol_window=int(row["realized_vol_window"]),
        min_exposure=float(row["min_exposure"]),
        max_exposure=float(row["max_exposure"]),
        trend_window=int(row.get("trend_window", 200)),
        momentum_window=int(row.get("momentum_window", 63)),
        trend_multiplier_below_ma=1.0,
        momentum_boost=1.0,
        crash_vol_cutoff=SIMPLE_NO_TREND_CRASH_VOL_CUTOFF,
        crash_exposure=float(row.get("crash_exposure", 0.0)),
        rebound=ReboundParams(use_rebound_module=False),
        governor=DrawdownGovernorParams(use_drawdown_governor=False),
        risk_off_symbol=str(row.get("risk_off_symbol", "CASH")).upper(),
        risk_off_weight=float(row.get("risk_off_weight", 0.0)),
        market_internals=MarketInternalsParams(use_market_internals=False),
        asset_config=asset_config_from_row(row),
    )


def _load_stitched(output_dir: Path) -> pd.DataFrame:
    stitched = _read_csv(output_dir / "stitched_equity.csv", index_col=0, parse_dates=True)
    if not stitched.empty:
        stitched.index.name = "Date"
        if "equity" not in stitched.columns and "ret" in stitched.columns:
            stitched["equity"] = (1.0 + stitched["ret"].astype(float)).cumprod()
    return stitched


def _load_price_data_for_run(run_config: Dict[str, Any], wf_table: pd.DataFrame) -> pd.DataFrame:
    data_csv = run_config.get("data_csv")
    include_ohlc = bool(run_config.get("download_ohlc", True))
    symbols = [str(symbol).upper() for symbol in run_config.get("symbols", ["TQQQ", "QQQ"])]
    for symbol in (
        run_config.get("benchmark_symbol", "TQQQ"),
        run_config.get("asset_config", {}).get("trade_asset", "TQQQ") if isinstance(run_config.get("asset_config"), dict) else "TQQQ",
        run_config.get("asset_config", {}).get("primary_signal_asset", "QQQ") if isinstance(run_config.get("asset_config"), dict) else "QQQ",
        run_config.get("asset_config", {}).get("secondary_filter_asset", "QQQ") if isinstance(run_config.get("asset_config"), dict) else "QQQ",
    ):
        if symbol:
            symbols.append(str(symbol).upper())
    if not wf_table.empty:
        for column in ("trade_asset", "primary_signal_asset", "secondary_filter_asset", "benchmark_symbol", "risk_off_symbol"):
            if column in wf_table.columns:
                symbols.extend(str(value).upper() for value in wf_table[column].dropna().unique())
    symbols = list(dict.fromkeys(symbols))

    if data_csv:
        data = pd.read_csv(Path(str(data_csv)), parse_dates=["Date"]).set_index("Date").sort_index()
        if CASH_SYMBOL in symbols:
            data = add_cash_series(data, include_ohlc=include_ohlc)
        columns: list[str] = []
        for symbol in symbols:
            if symbol in data.columns:
                columns.append(symbol)
            columns.extend(column for column in ohlc_columns(symbol) if column in data.columns)
        return data[list(dict.fromkeys(columns))].sort_index()

    return load_prices(
        start=str(run_config.get("start_date", "2011-01-01")),
        end=str(run_config.get("end_date")) if run_config.get("end_date") else None,
        symbols=symbols,
        use_csv_if_exists=bool(run_config.get("use_csv_if_exists", True)),
        cache_dir=str(run_config.get("cache_dir", "./price_cache")),
        dropna=False,
        allow_missing_symbols=True,
        include_ohlc=include_ohlc,
    )


def replay_simple_voltarget_no_trend(
    *,
    price_data: pd.DataFrame,
    wf_table: pd.DataFrame,
    transaction_cost_bps: float,
    execution_model: Optional[str] = None,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    prev_position = 0.0
    prev_risk_off_weight = 0.0
    for _, row in wf_table.iterrows():
        params = simple_voltarget_no_trend_params_from_row(row)
        piece = make_vol_target_eval_piece(
            full_data=price_data,
            params=params,
            eval_start=pd.Timestamp(row["test_start"]),
            eval_end=pd.Timestamp(row["test_end"]),
            transaction_cost_bps=float(transaction_cost_bps),
            prev_position=prev_position,
            prev_risk_off_weight=prev_risk_off_weight,
            execution_model=execution_model,
        )
        pieces.append(piece)
        prev_position = float(piece["tqqq_weight"].iloc[-1]) if "tqqq_weight" in piece.columns else float(piece["position"].iloc[-1])
        prev_risk_off_weight = float(piece["risk_off_weight"].iloc[-1]) if "risk_off_weight" in piece.columns else 0.0
    if not pieces:
        return pd.DataFrame()
    stitched = pd.concat(pieces).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    stitched["equity"] = (1.0 + stitched["ret"].astype(float)).cumprod()
    stitched["drawdown"] = stitched["equity"] / stitched["equity"].cummax() - 1.0
    return stitched


def _slice(frame: pd.DataFrame, mask: Iterable[bool]) -> pd.DataFrame:
    if frame.empty:
        return frame
    aligned = pd.Series(mask, index=frame.index).reindex(frame.index).fillna(False).astype(bool)
    return frame.loc[aligned].copy()


def _ratio_for_slice(strategy: pd.DataFrame, benchmark: pd.DataFrame, mask: Optional[pd.Series] = None) -> float:
    left = strategy.copy()
    right = benchmark.copy()
    if mask is not None:
        left = _slice(left, mask)
        right = _slice(right, mask)
    common = left.index.intersection(right.index)
    if len(common) == 0:
        return np.nan
    left_ret = left.loc[common, "ret"].astype(float)
    right_ret = right.loc[common, "ret"].astype(float)
    strategy_final = float((1.0 + left_ret).cumprod().iloc[-1])
    benchmark_final = float((1.0 + right_ret).cumprod().iloc[-1])
    return strategy_final / benchmark_final if benchmark_final != 0.0 else np.nan


def _comparison_fields(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    suffix: str,
) -> Dict[str, float]:
    common = strategy.index.intersection(benchmark.index)
    if len(common) == 0:
        return {
            f"ratio_vs_{suffix}": np.nan,
            f"cagr_diff_vs_{suffix}": np.nan,
            f"max_dd_diff_vs_{suffix}": np.nan,
            f"sharpe_diff_vs_{suffix}": np.nan,
            f"annual_turnover_diff_vs_{suffix}": np.nan,
            f"transaction_cost_drag_vs_{suffix}": np.nan,
            f"ex_2022_ratio_vs_{suffix}": np.nan,
            f"ratio_2022_only_vs_{suffix}": np.nan,
        }
    left = strategy.loc[common].copy()
    right = benchmark.loc[common].copy()
    left_metrics = _performance_from_frame(left)
    right_metrics = _performance_from_frame(right)
    ex_2022 = pd.Series(left.index.year != 2022, index=left.index)
    only_2022 = pd.Series(left.index.year == 2022, index=left.index)
    return {
        f"ratio_vs_{suffix}": left_metrics["final_equity"] / right_metrics["final_equity"]
        if right_metrics["final_equity"] != 0.0
        else np.nan,
        f"cagr_diff_vs_{suffix}": left_metrics["cagr"] - right_metrics["cagr"],
        f"max_dd_diff_vs_{suffix}": left_metrics["max_dd"] - right_metrics["max_dd"],
        f"sharpe_diff_vs_{suffix}": left_metrics["sharpe"] - right_metrics["sharpe"],
        f"annual_turnover_diff_vs_{suffix}": left_metrics["annual_turnover"] - right_metrics["annual_turnover"],
        f"transaction_cost_drag_vs_{suffix}": left_metrics["total_transaction_cost"]
        - right_metrics["total_transaction_cost"],
        f"ex_2022_ratio_vs_{suffix}": _ratio_for_slice(left, right, ex_2022),
        f"ratio_2022_only_vs_{suffix}": _ratio_for_slice(left, right, only_2022),
    }


def classify_fair_leverage_candidate(row: Dict[str, Any]) -> str:
    full_ratio = _safe_float(row.get("stage2_final_equity_ratio", row.get("ratio_vs_1x_tqqq")))
    ex_2022 = _safe_float(row.get("stage2_final_equity_ratio_ex_2022"))
    min_leave = _safe_float(row.get("stage2_min_leave_one_year_out_ratio"))
    same_avg = _safe_float(row.get("ratio_vs_same_avg_exposure_constant_tqqq"))
    simple = _safe_float(row.get("ratio_vs_simple_voltarget_no_trend"))
    same_max = _safe_float(row.get("ratio_vs_same_max_constant_tqqq"))
    one_x = _safe_float(row.get("ratio_vs_1x_tqqq"))
    dd_improvement = _safe_float(row.get("stage2_max_drawdown_improvement_ex_2022"))
    category = str(row.get("category", ""))
    governor_adds = _truthy(row.get("governor_adds_value_outside_2022"))

    if (
        category == "tqqq_voltarget_governor"
        and pd.notna(full_ratio)
        and full_ratio > 1.0
        and pd.notna(ex_2022)
        and ex_2022 <= 1.0
        and not governor_adds
    ):
        return "governor_overfit_to_2022"
    if (
        pd.notna(full_ratio)
        and full_ratio > 1.0
        and pd.notna(ex_2022)
        and ex_2022 > 1.0
        and pd.notna(min_leave)
        and min_leave > 1.0
        and pd.notna(same_avg)
        and same_avg > 1.0
        and pd.notna(simple)
        and simple > 1.0
    ):
        return "persistent_candidate"
    if (
        pd.notna(full_ratio)
        and full_ratio > 1.0
        and pd.notna(ex_2022)
        and 0.95 <= ex_2022 <= 1.0
        and pd.notna(dd_improvement)
        and dd_improvement > 0.20
    ):
        return "crash_control_overlay"
    if pd.notna(one_x) and one_x > 1.0 and (
        (pd.notna(same_max) and same_max <= 1.0)
        or (pd.notna(same_avg) and same_avg <= 1.0)
    ):
        return "leverage_only"
    return "reject"


def _candidate_rows(output_dir: Path) -> pd.DataFrame:
    summary = _read_csv(output_dir / "voltarget_stage2_summary.csv")
    if summary.empty:
        raise FileNotFoundError(f"Missing voltarget_stage2_summary.csv in {output_dir}")
    category = summary.get("category", pd.Series(dtype=str)).astype(str)
    status = summary.get("status", pd.Series("ok", index=summary.index)).astype(str)
    return summary[category.isin(VOLTARGET_CATEGORIES) & (status == "ok")].copy()


def build_stage2_fair_leverage_summary(output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows: list[Dict[str, Any]] = []
    for _, candidate in _candidate_rows(output_dir).iterrows():
        candidate_dir = Path(str(candidate.get("baseline_output_dir", "")))
        stitched = _load_stitched(candidate_dir)
        wf_table = _read_csv(candidate_dir / "walk_forward_windows.csv")
        if stitched.empty or wf_table.empty:
            continue

        run_config = _read_json(candidate_dir / "run_config.json")
        price_data = _load_price_data_for_run(run_config, wf_table)
        transaction_cost_bps = _safe_float(run_config.get("transaction_cost_bps"), 0.0)
        execution_model = normalize_execution_model(run_config.get("execution_model", CLOSE_TO_CLOSE_SHIFTED))
        trade_asset = str(wf_table.get("trade_asset", pd.Series(["TQQQ"])).dropna().iloc[0]).upper()
        selected_max = infer_selected_max_exposure(wf_table=wf_table, stitched=stitched, config=run_config)
        avg_exposure = realized_average_exposure(stitched)

        constants = {
            "1x_tqqq": constant_exposure_backtest(
                price_data,
                dates=stitched.index,
                exposure=1.0,
                trade_asset=trade_asset,
                transaction_cost_bps=transaction_cost_bps,
                execution_model=execution_model,
            ),
            "same_max_constant_tqqq": constant_exposure_backtest(
                price_data,
                dates=stitched.index,
                exposure=selected_max,
                trade_asset=trade_asset,
                transaction_cost_bps=transaction_cost_bps,
                execution_model=execution_model,
            ),
            "same_avg_exposure_constant_tqqq": constant_exposure_backtest(
                price_data,
                dates=stitched.index,
                exposure=avg_exposure,
                trade_asset=trade_asset,
                transaction_cost_bps=transaction_cost_bps,
                execution_model=execution_model,
            ),
        }
        simple_no_trend = replay_simple_voltarget_no_trend(
            price_data=price_data,
            wf_table=wf_table,
            transaction_cost_bps=transaction_cost_bps,
            execution_model=execution_model,
        ).reindex(stitched.index)

        strategy_metrics = _performance_from_frame(stitched)
        row: Dict[str, Any] = {
            "family": str(candidate.get("family", "")),
            "category": str(candidate.get("category", "")),
            "classification": "",
            "baseline_output_dir": str(candidate_dir),
            "start_date": stitched.index[0].date().isoformat(),
            "end_date": stitched.index[-1].date().isoformat(),
            "date_count": int(len(stitched)),
            "transaction_cost_bps": transaction_cost_bps,
            "execution_model": execution_model,
            "trade_asset": trade_asset,
            "selected_max_exposure": selected_max,
            "realized_average_exposure": avg_exposure,
            "strategy_final_equity": strategy_metrics["final_equity"],
            "stage2_final_equity_ratio": _safe_float(candidate.get("final_equity_ratio")),
            "stage2_final_equity_ratio_ex_2022": _safe_float(candidate.get("final_equity_ratio_ex_2022")),
            "stage2_min_leave_one_year_out_ratio": _safe_float(candidate.get("min_leave_one_year_out_ratio")),
            "stage2_max_drawdown_improvement_ex_2022": _safe_float(
                candidate.get("max_drawdown_improvement_ex_2022")
            ),
            "stage2_one_year_dominated": _truthy(candidate.get("one_year_dominated")),
            "stage2_largest_contribution_year": candidate.get("largest_contribution_year", ""),
            "stage2_largest_single_year_contribution_share": _safe_float(
                candidate.get("largest_single_year_contribution_share")
            ),
            "governor_adds_value_outside_2022": candidate.get("governor_adds_value_outside_2022", ""),
        }
        for suffix, benchmark in constants.items():
            row.update(_comparison_fields(stitched, benchmark, suffix))
        row.update(_comparison_fields(stitched, simple_no_trend, "simple_voltarget_no_trend"))
        row["classification"] = classify_fair_leverage_candidate(row)
        rows.append(row)

    return pd.DataFrame(rows, columns=FAIR_LEVERAGE_COLUMNS)


def _answer_from_row(row: Optional[pd.Series], column: str, threshold: float = 1.0) -> str:
    if row is None:
        return "inconclusive"
    value = _safe_float(row.get(column))
    if not pd.notna(value):
        return "inconclusive"
    return "yes" if value > threshold else "no"


def _ex_2022_interest_answer(row: Optional[pd.Series]) -> str:
    if row is None:
        return "inconclusive"
    stage2 = _safe_float(row.get("stage2_final_equity_ratio_ex_2022"))
    fair_avg = _safe_float(row.get("ex_2022_ratio_vs_same_avg_exposure_constant_tqqq"))
    fair_simple = _safe_float(row.get("ex_2022_ratio_vs_simple_voltarget_no_trend"))
    if pd.notna(stage2) and stage2 > 1.0:
        if pd.notna(fair_avg) and pd.notna(fair_simple) and fair_avg > 1.0 and fair_simple > 1.0:
            return "yes versus TQQQ and fair leverage/no-trend baselines"
        return "yes versus 1.0x TQQQ, but not versus fair leverage/no-trend baselines ex-2022"
    if pd.notna(stage2):
        return "no versus 1.0x TQQQ ex-2022"
    return "inconclusive"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    table = df[[column for column in columns if column in df.columns]].copy()
    table = table.where(pd.notna(table), "")
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in table.columns) + " |")
    return "\n".join(lines)


def write_stage2_fair_leverage_report(path: Path, summary: pd.DataFrame) -> None:
    plain_rows = summary[summary["category"].astype(str) == "tqqq_voltarget"] if not summary.empty else pd.DataFrame()
    governor_rows = (
        summary[summary["category"].astype(str) == "tqqq_voltarget_governor"] if not summary.empty else pd.DataFrame()
    )
    plain = plain_rows.iloc[0] if not plain_rows.empty else None
    governor = governor_rows.iloc[0] if not governor_rows.empty else None

    if plain is not None:
        if _safe_float(plain.get("ratio_vs_simple_voltarget_no_trend")) > 1.0:
            attribution = "volatility scaling plus trend/crash timing"
        elif _safe_float(plain.get("ratio_vs_same_avg_exposure_constant_tqqq")) > 1.0:
            attribution = "volatility scaling, not proven trend timing"
        elif _safe_float(plain.get("ratio_vs_1x_tqqq")) > 1.0:
            attribution = "mostly leverage/risk-budget exposure"
        else:
            attribution = "not enough evidence of a persistent edge"
    else:
        attribution = "inconclusive"

    lines = [
        "# Stage 2 Fair Leverage Benchmark Report",
        "",
        "This attribution report compares completed Stage 2 VolTarget candidates against tradable constant-exposure and simple volatility-scaling baselines over the exact same stitched dates.",
        "It is not an investment recommendation.",
        "",
        "## Direct Answers",
        f"- Does plain VolTarget beat same-max constant TQQQ? {_answer_from_row(plain, 'ratio_vs_same_max_constant_tqqq')}.",
        f"- Does plain VolTarget beat same-average-exposure constant TQQQ? {_answer_from_row(plain, 'ratio_vs_same_avg_exposure_constant_tqqq')}.",
        f"- Does plain VolTarget beat SimpleVolTargetNoTrend? {_answer_from_row(plain, 'ratio_vs_simple_voltarget_no_trend')}.",
        f"- Does the governor add value outside 2022? {'yes' if governor is not None and _truthy(governor.get('governor_adds_value_outside_2022')) else 'no'}.",
        f"- Is the result mainly leverage, volatility scaling, trend timing, or crash avoidance? {attribution}.",
        f"- Would the strategy still be interesting if 2022 were excluded? {_ex_2022_interest_answer(plain)}.",
        "",
        "## Method",
        "- `same-max constant TQQQ` holds the candidate's selected maximum exposure as one fixed daily TQQQ weight.",
        "- `same-average-exposure constant TQQQ` holds the candidate's realized average `trade_weight` over stitched dates.",
        "- `SimpleVolTargetNoTrend` replays the selected VolTarget walk-forward windows with the same target volatility, realized-volatility window, minimum exposure, and maximum exposure, while disabling trend filter, momentum boost, rebound, drawdown governor, market internals, and crash-vol/trend override.",
        "- Constant exposure benchmarks include the configured initial transaction cost on the first move from zero exposure to the constant target weight.",
        "- Ex-2022 and 2022-only ratios are recomputed from daily returns on identical dates; no re-optimization is performed.",
        "",
        "## Candidate Summary",
        _markdown_table(
            summary,
            [
                "family",
                "classification",
                "ratio_vs_1x_tqqq",
                "ratio_vs_same_max_constant_tqqq",
                "ratio_vs_same_avg_exposure_constant_tqqq",
                "ratio_vs_simple_voltarget_no_trend",
                "ex_2022_ratio_vs_same_avg_exposure_constant_tqqq",
                "ex_2022_ratio_vs_simple_voltarget_no_trend",
                "ratio_2022_only_vs_simple_voltarget_no_trend",
                "stage2_one_year_dominated",
            ],
        ),
        "",
        "## Attribution Detail",
        _markdown_table(
            summary,
            [
                "family",
                "selected_max_exposure",
                "realized_average_exposure",
                "cagr_diff_vs_same_avg_exposure_constant_tqqq",
                "cagr_diff_vs_simple_voltarget_no_trend",
                "max_dd_diff_vs_same_avg_exposure_constant_tqqq",
                "max_dd_diff_vs_simple_voltarget_no_trend",
                "sharpe_diff_vs_same_avg_exposure_constant_tqqq",
                "sharpe_diff_vs_simple_voltarget_no_trend",
                "annual_turnover_diff_vs_simple_voltarget_no_trend",
                "transaction_cost_drag_vs_simple_voltarget_no_trend",
            ],
        ),
        "",
        "## Classification Rules",
        "- `persistent_candidate`: full ratio, ex-2022 ratio, leave-one-year ratio, same-average constant ratio, and SimpleVolTargetNoTrend ratio all exceed 1.0.",
        "- `crash_control_overlay`: full ratio exceeds 1.0, ex-2022 ratio is 0.95 to 1.0, and ex-2022 drawdown improvement is large.",
        "- `leverage_only`: beats 1.0x TQQQ but loses to same-max or same-average constant exposure.",
        "- `governor_overfit_to_2022`: governor variant beats the full period but fails ex-2022 and does not beat plain VolTarget outside 2022.",
        "- `reject`: none of the above classifications is supported.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stage2_fair_leverage_outputs(output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_stage2_fair_leverage_summary(output_dir)
    summary.to_csv(output_dir / "fair_leverage_benchmark_summary.csv", index=False)
    write_stage2_fair_leverage_report(output_dir / "fair_leverage_benchmark_report.md", summary)
    print(summary.to_string(index=False))
    print(f"\nFair leverage benchmark summary written to: {output_dir / 'fair_leverage_benchmark_summary.csv'}")
    return summary
